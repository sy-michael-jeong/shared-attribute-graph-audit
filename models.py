# Model definitions and shared helpers (seeding, metrics, aggregation).
import os
import statistics

import numpy as np
import torch

from common import seed_all  # noqa: F401  (re-exported for train.py)
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score



def pick_device(name="auto"):
    if name in ("cpu", "cuda"):
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def metrics(y_true, y_pred):
    vals, cnts = np.unique(y_true, return_counts=True)
    minority = [vals[np.argmin(cnts)]]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "minority_f1": float(f1_score(y_true, y_pred, labels=minority,
                                      average="macro", zero_division=0)),
        "minority_recall": float(recall_score(y_true, y_pred, labels=minority,
                                              average="macro", zero_division=0)),
    }


def agg(vals):
    """Seed-wise summary, carrying both definitions of the spread.

    `std` is the population standard deviation and is what the paper's plus
    or minus reports. `std_sample` is the sample standard deviation. Over
    five seeds the two differ by as much as twelve percent: HAN on ISCX-VPN
    gives 0.0336 and 0.0375. Keeping only one of them leaves a reader who
    recomputes the spread with a different tool unable to tell which number
    is wrong.
    """
    ok = [v for v in vals if v >= 0]
    if not ok:
        return {"mean": -1.0, "std": 0.0, "std_sample": 0.0, "n": 0}
    return {"mean": round(statistics.fmean(ok), 4),
            "std": round(float(np.std(ok)), 4) if len(ok) > 1 else 0.0,
            "std_sample": round(float(np.std(ok, ddof=1)), 4)
            if len(ok) > 1 else 0.0,
            "n": len(ok), "values": [round(v, 4) for v in ok]}


def class_weights(y, num_classes, device):
    cw = np.bincount(y, minlength=num_classes).astype(np.float32)
    cw = (cw.sum() / np.maximum(cw, 1)) / num_classes
    return torch.tensor(cw, device=device, dtype=torch.float32)


class HAN(nn.Module):
    # Wang et al., WWW 2019. Node-level + semantic-level attention via
    # PyG HANConv, with a learnable residual on the input features for
    # training stability.
    def __init__(self, in_dim, hidden, num_classes, metadata,
                 num_layers=2, heads=4, dropout=0.5, residual=True):
        super().__init__()
        from torch_geometric.nn import HANConv
        self.residual = residual
        self.input_proj = nn.Linear(in_dim, hidden)
        self.input_drop = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            HANConv(in_channels=hidden, out_channels=hidden,
                    metadata=metadata, heads=heads, dropout=dropout)
            for _ in range(num_layers))
        if residual:
            self.alpha = nn.Parameter(torch.zeros(num_layers))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x_dict, edge_index_dict):
        x = self.input_drop(F.gelu(self.input_proj(x_dict["flow"])))
        skip = x
        h = {"flow": x}
        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict).get("flow")
            if out is None:
                out = h["flow"]
            if self.residual:
                a = torch.sigmoid(self.alpha[i])
                out = a * skip + (1.0 - a) * out
            h = {"flow": out}
        return self.head(h["flow"])


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)

class EGraphSAGE(nn.Module):
    # Endpoint graph baseline (Lo et al., NOMS 2022): nodes are endpoints,
    # flows are edges, and each flow is classified from the concatenated
    # embeddings of its two endpoint nodes.
    #
    # That construction has a path from a flow to its own prediction that does
    # not pass through any neighbour. The flow is an edge, its own features are
    # the message on that edge, and the message is aggregated into exactly the
    # two nodes the classifier then reads. So part of what looks like a graph
    # gain can be the flow reading itself.
    #
    # `exclude_self` removes that path at the last layer: the message the flow
    # itself contributed is subtracted from each of its endpoints, and the mean
    # is retaken over the remaining degree. An endpoint carrying only this one
    # flow has nothing left, and keeps the node state instead of a zero, since a
    # zero would be a third thing rather than "no neighbour".
    #
    # The earlier layers are left alone, so the flow's features still reach its
    # endpoints through layer one. **The control therefore removes only part of
    # the path and the number it produces is a lower bound on the bypass.**
    def __init__(self, edge_dim, hidden, num_classes, num_layers=2, dropout=0.5,
                 exclude_self=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.exclude_self = exclude_self
        self.msg = nn.ModuleList()
        self.apl = nn.ModuleList()
        in_dim = edge_dim  # node init = all-ones(edge_dim), as in the original design
        for _ in range(num_layers):
            self.msg.append(nn.Linear(in_dim + edge_dim, hidden))
            self.apl.append(nn.Linear(in_dim + hidden, hidden))
            in_dim = hidden
        self.cls = nn.Linear(2 * hidden, num_classes)

    def forward(self, e_feat, src, dst, num_nodes):
        h = torch.ones(num_nodes, e_feat.shape[1], device=e_feat.device)
        s2 = torch.cat([src, dst])
        d2 = torch.cat([dst, src])
        e2 = torch.cat([e_feat, e_feat], dim=0)
        deg = torch.zeros(num_nodes, 1, device=e_feat.device)
        deg.index_add_(0, d2, torch.ones(len(d2), 1, device=e_feat.device))
        deg = deg.clamp(min=1.0)
        n_e = e_feat.shape[0]
        for k in range(self.num_layers):
            m = F.relu(self.msg[k](torch.cat([h[s2], e2], dim=1)))
            summed = torch.zeros(num_nodes, m.shape[1], device=e_feat.device)
            summed.index_add_(0, d2, m)
            last = (k == self.num_layers - 1)
            if self.exclude_self and last:
                # 엣지 e 의 두 방향 메시지는 m[:n_e](dst 로) 와 m[n_e:](src 로) 다.
                # 각 끝점의 합에서 그 하나를 빼고 남은 차수로 다시 평균 낸다.
                d_rest = (deg - 1.0)
                a_dst = (summed[dst] - m[:n_e]) / d_rest[dst].clamp(min=1.0)
                a_src = (summed[src] - m[n_e:]) / d_rest[src].clamp(min=1.0)
                # 이웃이 남지 않은 끝점은 노드 상태를 그대로 쓴다.
                alone_d = (d_rest[dst] <= 0).float()
                alone_s = (d_rest[src] <= 0).float()
                h_dst = F.relu(self.apl[k](torch.cat([h[dst], a_dst], dim=1)))
                h_src = F.relu(self.apl[k](torch.cat([h[src], a_src], dim=1)))
                keep_d = F.relu(self.apl[k](torch.cat(
                    [h[dst], torch.zeros_like(a_dst)], dim=1)))
                keep_s = F.relu(self.apl[k](torch.cat(
                    [h[src], torch.zeros_like(a_src)], dim=1)))
                h_dst = alone_d * keep_d + (1 - alone_d) * h_dst
                h_src = alone_s * keep_s + (1 - alone_s) * h_src
                h_dst = F.dropout(h_dst, p=self.dropout, training=self.training)
                h_src = F.dropout(h_src, p=self.dropout, training=self.training)
                return self.cls(torch.cat([h_src, h_dst], dim=1))
            aggr = summed / deg
            h = F.relu(self.apl[k](torch.cat([h, aggr], dim=1)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = torch.cat([h[src], h[dst]], dim=1)
        return self.cls(z)
