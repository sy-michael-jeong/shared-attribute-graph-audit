# Beam search over relation subsets (paper 5.2). Each configuration is
# trained once and scored by its validation macro-F1; these scores guide
# the beam and form the saturation curves. Evaluate selected
# configurations with train.py --model han (five seeds) before drawing
# conclusions.
import argparse
import json
from pathlib import Path

import yaml

from train import run_han


def beam_search(name, data_dir, cfg, pool, beam_k, seed):
    results = []
    stages = {}

    def evaluate(rels, stage):
        print("  trying %s" % rels)
        try:
            m = run_han(data_dir, cfg, rels, seed)
            score = float(m.get("best_val", -1.0))
        except Exception as e:
            print("  [error] %s" % e)
            m, score = {}, -1.0
        # 실현된 관계를 함께 남긴다. 엣지를 만들지 못한 관계가 섞이면
        # 요청 집합이 달라도 그래프가 같다. select_relations.py 와 make_fig1.py 가
        # 중복을 없앨 때 이 목록을 쓴다. 없으면 조용히 요청 집합으로
        # 되돌아가 같은 그래프를 여럿으로 센다.
        r = {"stage": stage, "metapaths": list(rels),
             "metapaths_used": list(m.get("relations", rels)),
             "score": score, "val_macro_f1": score,
             "test_macro_f1": float(m.get("macro_f1", -1.0)),
             "minority_f1": float(m.get("minority_f1", 0.0)),
             "weighted_f1": float(m.get("weighted_f1", 0.0))}
        results.append(r)
        return r

    print("== stage 1: single relations")
    stage = [evaluate([mp], 1) for mp in pool]
    stage.sort(key=lambda r: r["score"], reverse=True)
    stages[1] = stage
    beam = [r["metapaths"] for r in stage[:beam_k]]

    for k in range(2, len(pool) + 1):
        print("== stage %d" % k)
        seen, stage = set(), []
        for base in beam:
            for mp in pool:
                if mp in base:
                    continue
                cand = tuple(sorted(base + [mp]))
                if cand in seen:
                    continue
                seen.add(cand)
                stage.append(evaluate(list(cand), k))
        stage.sort(key=lambda r: r["score"], reverse=True)
        stages[k] = stage
        beam = [r["metapaths"] for r in stage[:beam_k]]

    results.sort(key=lambda r: r["score"], reverse=True)
    # `all_results` carries every configuration the beam visited, not only the
    # ten best. select_relations.py re-trains that set, so dropping it here would leave
    # the selection step with nothing to read.
    return {"dataset": name, "beam_k": beam_k, "pool": pool,
            "stages": stages, "all_results": results,
            "top_overall": results[:10]}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--beam-k", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    pool = cfg["hin"]["metapaths"]
    runs_root = Path(args.runs)
    runs_root.mkdir(parents=True, exist_ok=True)
    out_path = runs_root / "combinatorial_grand_summary.json"
    grand = json.load(open(out_path)) if out_path.exists() else {}

    for name in args.datasets:
        print("### beam search (K=%d) - %s" % (args.beam_k, name))
        grand[name] = beam_search(name, Path(args.data) / name, cfg, pool,
                                  args.beam_k, cfg["seed"])
        json.dump(grand, open(out_path, "w"), indent=2)
        print("saved %s" % out_path)


if __name__ == "__main__":
    main()
