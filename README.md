# Auditing what a TLS relation graph carries — reproducibility artifact

Every number the paper reports comes from a file in `results/`, and every file
in `results/` is produced by a command below. One check enforces that:

    python collect.py --results results --check     # every shipped file has a source

A file that fails either check does not ship.

## Layout

    *.py                 experiment and analysis scripts
    config.yaml          model, graph budget, seeds
    requirements.txt     pinned environment
    results/             the files the paper cites
    collect.py           the join between run outputs and results/

## What runs where

The training scripts write one file per model into whatever `--runs` directory
they are given. `results/` is arranged by experiment instead, and some files are
renamed so a reader can find them from a section number. `collect.py` holds that
mapping, one line per shipped file. Analysis scripts that write a single file
take `--out` and write into `results/` directly.

So there are two shapes of command below:

    training     python train.py ... --runs runs/<tag>      then collect.py
    analysis     python <script>.py ... --out results/<path>

## Environment

    pip install -r requirements.txt

Zeek 8.2.0 for the packet-capture datasets. A CUDA device is used when present;
`config.yaml` sets `device: auto`.

## Data

The five corpora are public and are not redistributed here. `extract_pcap.py`
turns the captures into the flow table, `build_graph.py` writes the splits and
the edges.

    python build_graph.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --raw data/raw --out data/processed_deg2 --config config.yaml

That writes, per dataset, `X_*.npy` `y_*.npy` `meta_*.csv` `hin_edges_*.npy`
`hin_summary.json` `feature_names.json`.

The relations that are **requested** differ by dataset. `config.yaml` holds the
pool under `metapaths` and the per-dataset request under `metapaths_by_dataset`;
BCCC-DoH and CIC-AndMal request the seven TLS relations only. `hin_summary.json`
records `pool`, `requested` and `not_requested` alongside `used` and `skipped`,
so a relation that carries no edge can be told apart from one that was never
asked for.

Edge construction is deterministic: the shuffle seed of a value group is
`blake2b(value) XOR seed`, not Python's `hash`. So the graph can be rebuilt and
compared byte for byte, and that is the check that the scripts here are the ones
that produced the shipped results:

    python verify_graph.py --data data/processed_deg2 \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --config config.yaml --work data/_verify_graph \
        --out results/reproducibility/graph_rebuild.json

It copies the metadata, rebuilds only the edges, and compares. Of the 44
requested relations 35 are realised and all 35 come back byte-identical; the
other nine carry no edge in either build.

Six further roots are derived from it. Each is a copy with one thing changed,
and each is named for what changed.

    data/processed_deg2_masked          edges crossing a partition removed
    data/processed_deg2_perm<SEED>      values permuted among the flows that hold them
    data/processed_deg2_bccc10          BCCC-DoH with all ten relations
    data/processed_deg2_bccc10_nounk    the extractor's fill-in string returned to missing
    data/_random                        one relation of meaningless edges
    data/_k4, data/_tb60, data/_tb600   budget 4, time bin 60 s and 600 s

    python mask_edges.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out data/processed_deg2_masked \
        --stats results/masked/edge_statistics.json

    for S in 41 42 43 44 45; do
      python perm_relation_control.py --datasets bccc_dohbrw \
        --data data/processed_deg2 --out data/processed_deg2_perm$S \
        --fields cert_validity_bucket cert_subject --seed $S
      python build_graph.py --datasets bccc_dohbrw \
        --out data/processed_deg2_perm$S --config config.yaml
    done

The permuted column differs by dataset, because the relation the paper reports
differs. BCCC-DoH permutes `cert_validity_bucket` together with `cert_subject`,
ISCX-VPN `src_ip`, HIKARI `dst_ip`, VNAT `ts`, CIC-AndMal all seven TLS columns.

The last two edit the metadata in place rather than writing a second copy,
because the rebuild reads the same files it would have written. So the copy is
made first and the script is pointed at the copy. Both refuse to run on a root
another experiment reads.

    cp -r data/processed_deg2_bccc10 data/processed_deg2_bccc10_nounk
    python mask_synthetic_token.py --data data/processed_deg2_bccc10_nounk \
        --datasets bccc_dohbrw \
        --column cert_validity_bucket --tokens unknown
    python build_graph.py --datasets bccc_dohbrw \
        --out data/processed_deg2_bccc10_nounk --config config.yaml

    cp -r data/processed_deg2 data/_random
    python random_edge_control.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/_random --degree 2 --group-size 100

## Before training — what can be measured without a model

    python profile_relations.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 \
        --out results/profiling/relation_profiling.json

    python dataset_composition.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 \
        --out results/composition/composition_order_preserving.json

    python group_split_feasibility.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --restarts 1000 \
        --out results/group_split/group_split_feasibility.json

    python split_overlap_audit.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --protocol order_preserving \
        --out results/split_protocol/overlap/overlap_order_preserving.json
    python split_overlap_audit.py --datasets bccc_dohbrw iscx_vpn hikari vnat \
        --data data/processed_random_v2 --protocol random \
        --out results/split_protocol/overlap/overlap_random.json
    python split_overlap_audit.py --datasets cic_andmal \
        --data data/processed --protocol random \
        --out results/split_protocol/overlap/overlap_random_cic.json

    python graph_stats.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out results/structure

    python edge_homophily.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out results/homophily/edge_homophily.json
    python edge_homophily.py --datasets bccc_dohbrw --data data/processed_deg2_bccc10 \
        --out results/homophily/edge_homophily_bccc10.json
    python edge_homophily.py --datasets bccc_dohbrw --data data/processed_deg2_bccc10_nounk \
        --out results/homophily/edge_homophily_bccc10_nounk.json
    python edge_homophily.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/_random --out results/homophily/edge_homophily_random_control.json

Two reference lines. `availability_null.py` takes the maximum over fields and
compares it against a label permutation. `availability_rule.py` reports every
field separately, and every field on three scopes, because the score changes
between them and the difference disappears under rounding.

    python baselines.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out results/reference_lines

    python availability_null.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --permutations 1000 \
        --out results/reference_lines/availability_null.json
    python availability_null.py --datasets bccc_dohbrw \
        --data data/processed_deg2_bccc10_nounk --permutations 1000 \
        --out results/reference_lines/availability_null_nounk.json

    python availability_rule.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 \
        --out results/reference_lines/availability_rule.json
    python availability_rule.py --datasets bccc_dohbrw \
        --data data/processed_deg2_bccc10_nounk \
        --out results/reference_lines/availability_rule_nounk.json

    python check_index_feature.py --raw data/raw/hikari --split random \
        --out results/feature_audit/hikari_index_column_random.json

Sec. 7.5 asks the same two questions of every feature column in every dataset,
and both are answerable from the matrix alone: does the column carry the row's
identity rather than the flow's behaviour, and does it reproduce the timestamp
that orders the split. Asking a trained score instead is too late, because by
then the leaked column has been paid for.

    python audit.py features --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 \
        --out results/feature_audit/feature_matrix_audit.json

    python audit.py drift --csv data/raw/hikari/ALLFLOWMETER_HIKARI2021.csv \
        --out results/composition/hikari_drift.json

    python ctu13_coverage.py --survey \
        --out results/dataset_suitability/ctu13_tls_coverage.json

The extractor writes a string when it cannot read a field, and those strings are
counted as values. How much of each field's coverage is such a string decides
where the fifth control has anything to remove, and it can be counted without
training:

    python token_census.py --data data/processed_deg2_bccc10 \
        --datasets bccc_dohbrw \
        --out results/reference_lines/token_census.json
    python token_census.py --data data/processed_deg2 \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --out results/reference_lines/token_census_all.json

`--survey` adds one request per capture directory on the same server, which
takes a few minutes. Without it the five scenarios are still measured; only the
count of other captures carrying a TLS log is left out.

## The main table and the controls

The reported relation set differs by dataset. `common.SELECTED` holds it and
`train.py` reads it when `--sets` is not given.

    for DS in bccc_dohbrw iscx_vpn hikari cic_andmal vnat; do
      python train.py --model han --datasets $DS --data data/processed_deg2 \
        --runs runs/main/han/$DS
      python train.py --model gcn --datasets $DS --data data/processed_deg2 \
        --runs runs/main/gcn/$DS
    done
    python train.py --model mlp --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --runs runs/main/mlp
    python train.py --model egs --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --runs runs/main/egraphsage
    python same_info_matched.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out runs/main/tabular

Five controls. Each cuts one thing and holds the rest.

    # self loops: every relation reduced to a loop on every node
    python no_edge_han.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out runs/no_edge

    # cross-partition masking
    for DS in bccc_dohbrw iscx_vpn hikari cic_andmal vnat; do
      python train.py --model han --datasets $DS --data data/processed_deg2_masked \
        --runs runs/masked/han/$DS
    done

    # value permutation, five permutation seeds by five training seeds
    for DS in bccc_dohbrw iscx_vpn hikari cic_andmal vnat; do
      for S in 41 42 43 44 45; do
        python train.py --model han --datasets $DS --data data/processed_deg2_perm$S \
          --runs runs/permutation/$DS/seed_$S
      done
    done

    # meaningless edges at the same budget
    for DS in bccc_dohbrw cic_andmal hikari; do
      python train.py --model han --datasets $DS --data data/_random \
        --sets via_random --runs runs/random_control/$DS
    done
    python train.py --model han --datasets iscx_vpn vnat --data data/_random \
        --sets via_random --runs runs/random_control/iscx_vpn_vnat

    # the extractor's fill-in string returned to missing
    python train.py --model han --datasets bccc_dohbrw \
        --data data/processed_deg2_bccc10_nounk --sets via_cert_validity \
        --runs runs/single_relation/bccc_dohbrw_nounk

Each relation on its own, so that a set's score can be read against what its
members reach separately. BCCC-DoH is measured on the ten-relation root because
three of the four are host and time relations, which the reported pool omits.

    for REL in via_sni via_ja3 via_cert_subject via_alpn via_cert_issuer \
               via_tls_cipher_group via_cert_validity; do
      python train.py --model han --datasets cic_andmal --data data/processed_deg2 \
        --sets $REL --runs runs/single_relation/cic_andmal --seeds 41 42 43 44 45
    done
    for REL in via_src_host via_dst_host via_timebin via_cert_validity; do
      python train.py --model han --datasets bccc_dohbrw \
        --data data/processed_deg2_bccc10 --sets $REL \
        --runs runs/single_relation/bccc_dohbrw_with_host --seeds 41 42 43 44 45
    done

Two families against each other on the same root: the seven TLS relations, and
the two host relations. Whether the TLS graph is worth building is a comparison
against the host graph, not against no graph.

    for DS in iscx_vpn vnat bccc_dohbrw; do
      python train.py --model han --datasets $DS --data data/processed_deg2 \
        --sets "via_sni+via_ja3+via_cert_subject+via_alpn+via_cert_issuer+via_tls_cipher_group+via_cert_validity" \
        --runs runs/family/tls/$DS --seeds 41 42 43 44 45
      python train.py --model han --datasets $DS --data data/processed_deg2 \
        --sets "via_src_host+via_dst_host" \
        --runs runs/family/host/$DS --seeds 41 42 43 44 45
    done

Meaningless edges at three bucket sizes and three assignment seeds. A single
random graph says only that one budget was matched; the sweep says whether the
score follows the budget or the meaning.

    for G in 10 100 1000; do
      for SEED in 42 43 44; do
        cp -r data/processed_deg2 data/processed_rand_g${G}_s${SEED}
        python random_edge_control.py --datasets bccc_dohbrw iscx_vpn cic_andmal vnat \
          --data data/processed_rand_g${G}_s${SEED} --group-size $G --seed $SEED --degree 2
        python train.py --model han --datasets bccc_dohbrw iscx_vpn cic_andmal vnat \
          --data data/processed_rand_g${G}_s${SEED} --sets via_random \
          --runs runs/random_control/sweep/g${G}_s${SEED} --seeds 41 42 43 44 45
      done
    done

The random control above matches the edge budget but not the coverage: it gives
every flow a bucket, while a real relation only connects the flows that hold a
value. `--match-coverage-from` borrows which flows have a value from the
reversed CertValidity column and assigns buckets inside that set alone, so the
graph matches coverage, edge count and isolate structure and differs only in
whether the values mean anything.

    cp -r data/processed_deg2_bccc10_nounk data/_rand_match_bccc
    python random_edge_control.py --datasets bccc_dohbrw --data data/_rand_match_bccc \
        --match-coverage-from cert_validity_bucket --group-size 100 --degree 2
    python train.py --model han --datasets bccc_dohbrw --data data/_rand_match_bccc \
        --sets via_random --runs runs/random_match_bccc --seeds 41 42 43 44 45

Deployment. The graph is built from the training flows alone and new flows are
attached afterwards, in one direction only. `--datasets` takes one name, because
`--sets` differs by dataset and the five metadata tables carry the same columns,
so a list would run without error on the wrong relation.

    python test_inductive_attach.py          # 13 invariants, no GPU, no data

    python inductive_eval.py --datasets bccc_dohbrw --data data/processed_deg2 \
        --sets via_cert_validity --out runs/inductive
    python inductive_eval.py --datasets iscx_vpn --data data/processed_deg2 \
        --sets via_src_host --out runs/inductive
    python inductive_eval.py --datasets hikari --data data/processed_deg2 \
        --sets via_dst_host --out runs/inductive
    python inductive_eval.py --datasets vnat --data data/processed_deg2 \
        --sets via_timebin --out runs/inductive
    python inductive_eval.py --datasets cic_andmal --data data/processed_deg2 \
        --sets "via_sni+via_ja3+via_cert_subject+via_alpn+via_cert_issuer+via_tls_cipher_group+via_cert_validity" \
        --out runs/inductive

## Relation search, selection, and the full pool

    for DS in bccc_dohbrw iscx_vpn hikari vnat; do
      python search.py --datasets $DS --data data/processed_deg2 \
        --runs runs/saturation/$DS
      python select_relations.py --datasets $DS --reuse-scores \
        --summary runs/saturation/$DS/combinatorial_grand_summary.json \
        --out runs/selection
    done

`--reuse-scores` takes the scores the search already recorded. Without it every
configuration is trained a second time, which costs a GPU day and cannot change
the answer, since the search stored both the validation and the test score for
each one.

`select_relations.py` counts ties over distinct realized graphs, not over requested sets.
A configuration that names a relation producing no edges falls back to the
smaller graph, so two requested sets can be one graph. On BCCC-DoH the pool
lists seven TLS relations but SNI yields no edge, so the largest realized graph
holds six; on HIKARI the ten-relation pool realizes at most two.

    for DS in bccc_dohbrw iscx_vpn vnat; do
      python train.py --model han --datasets $DS --data data/processed_deg2 \
        --sets all --runs runs/typing_vs_pruning/han_full/$DS
    done

## Split protocol

The reported table uses the audit split. Three further roots exist because the
paper compares protocols, and a protocol is chosen when the graph is built, not
when the model is trained. Each is a separate build of the same captures.

    python build_graph.py --datasets bccc_dohbrw iscx_vpn hikari vnat \
        --out data/processed_random_v2 --config config.yaml --split-mode random
    python build_graph.py --datasets cic_andmal \
        --out data/processed --config config.yaml --split-mode random
    python build_graph.py --datasets bccc_dohbrw \
        --out data/_bccc_random --config config.yaml --split-mode random
    python build_graph.py --datasets iscx_vpn \
        --out data/_iscx_dstdisj --config config.yaml --split-mode dst_ip_disjoint

`processed_random_v2` and `_bccc_random` are the same protocol on the same
dataset, built at different times. They are kept apart rather than merged
because every result file records the root it read, and rewriting those strings
would make the record say something that was not run.

    python repeat_splits.py --datasets bccc_dohbrw iscx_vpn hikari vnat \
        --data data/processed_deg2 --runs runs/repeat

The summary it writes holds every seed, not only the mean, because the table in
Sec. 6.3 averages over variants and an average taken over four-place numbers can
land on a rounding boundary. To rebuild that summary from a run tree that
already exists, without training anything:

    python repeat_splits.py --datasets bccc_dohbrw iscx_vpn hikari vnat \
        --runs runs/repeat --rebuild-summary

    python train.py --model han --datasets bccc_dohbrw --data data/_bccc_random \
        --runs runs/split_protocol/random/han/bccc_dohbrw

The audit split separates time, not identifiers. On the destination-disjoint
root no address appears on both sides, and that is where the ordering between
the tabular baseline and the graph models reverses. All four models are run on
the same root so the comparison is between models and not between splits.

    python train.py --model hgb --datasets iscx_vpn --data data/_iscx_dstdisj \
        --runs runs/dstdisj/hgb --seeds 41 42 43 44 45
    python train.py --model mlp --datasets iscx_vpn --data data/_iscx_dstdisj \
        --runs runs/dstdisj/mlp --seeds 41 42 43 44 45
    python train.py --model han --datasets iscx_vpn --data data/_iscx_dstdisj \
        --sets via_src_host --runs runs/dstdisj/han --seeds 41 42 43 44 45
    python no_edge_han.py --datasets iscx_vpn --data data/_iscx_dstdisj \
        --sets via_src_host --out runs/dstdisj/noedge --seeds 41 42 43 44 45

## The decomposition under repeated splits

The share the edges carry is a ratio, and a ratio measured on one split says
nothing about how much the split chose it. `--models decomp` runs the four
models the decomposition needs on every variant, self-loop control included.

    python repeat_splits.py --datasets iscx_vpn --data data/processed_deg2 \
        --raw data/raw --runs runs/repeat_decomp_iscx --models decomp \
        --seeds 41 42 43 44 45

BCCC-DoH is rebuilt from the captures, so each variant costs an extraction of
13,754 files. `--check` takes one variant end to end first, which is also the
check that the corpus can be rebuilt at all: five of those captures are
truncated and Zeek discards them, and the count still comes back at 505,040,
so the same files were discarded when the reported table was made.

    python repeat_splits.py --datasets bccc_dohbrw --data data/processed_deg2 \
        --raw data/raw --runs runs/_check_bccc --models decomp --check \
        --seeds 41 42 43 44 45

Then the full set of eleven variants. The summary is rewritten after every
variant, so an interrupted run keeps what it finished and picks up from there.
Do not narrow the arguments to run only what is left: `variants()` draws the
random split seeds and the edge seeds from one generator in sequence, so
lowering `--n-random` changes which edge seeds come out and the tags no longer
match the earlier half. Keep the arguments and let it skip. `--fresh` forces a
rebuild from zero.

    python repeat_splits.py --datasets bccc_dohbrw --data data/processed_deg2 \
        --raw data/raw --runs runs/repeat_decomp_bccc --models decomp \
        --n-random 5 --cutoffs 0.15 0.20 0.25 --n-edge 3 \
        --seeds 41 42 43 44 45

This one runs for hours. Detach it (`nohup ... &`, `tmux`) rather than leaving it
in an editor terminal, which takes the run down with the window.

CIC-AndMal is the dataset where the edges carry nothing to begin with, so it is
the case that says whether the protocol moves a share or manufactures one. It
takes the same three families of variant as the others.

    python repeat_splits.py --datasets cic_andmal --data data/processed_deg2 \
        --raw data/raw --runs runs/repeat_decomp_cic --models decomp \
        --n-random 5 --cutoffs 0.15 0.20 0.25 --n-edge 3 \
        --seeds 41 42 43 44 45

## Typing without pruning, at equal relation count

The reported HAN receives one relation on three of the five corpora while GCN
receives the whole pool, so their difference mixes relation typing with
pruning. On HIKARI the pool is two relations, so a HAN given the whole pool is
compared with a GCN given the same two — the counts match and only the typing
differs.

    python train.py --model han --datasets hikari --data data/processed_deg2 \
        --sets "via_src_host+via_dst_host" --runs runs/han_full_hikari \
        --seeds 41 42 43 44 45

The four single relations of BCCC-DoH in `results/single_relation/` were
measured on the ten-relation root, so their self-loop reference has to come
from the same root rather than from the reported one.

    python no_edge_han.py --datasets bccc_dohbrw --data data/processed_deg2_bccc10 \
        --sets "via_src_host+via_dst_host+via_timebin+via_cert_validity" \
        --out runs/noedge_bccc10 --seeds 41 42 43 44 45

## The literature configuration

Sec. 6.6 runs the same decomposition on an approximation of a configuration
taken from the literature: two relations, VersionCipher (the JA3 stand-in,
column `ja3`) and the certificate subject. The permutation control jointly
shuffles **those** two columns; its results are `results/reta_approx/permutation/`. Permuting the columns of the
reported configuration instead would leave the literature configuration's own
relation untouched and report a value-identity term of nearly zero for a
reason that has nothing to do with the literature.

    for S in 41 42 43 44 45; do
      python perm_relation_control.py --datasets bccc_dohbrw \
        --data data/processed_deg2 --out data/_perm_reta$S \
        --fields ja3 cert_subject --seed $S
      python build_graph.py --datasets bccc_dohbrw --out data/_perm_reta$S \
        --config config.yaml --skip-materialize
      python train.py --model han --datasets bccc_dohbrw --data data/_perm_reta$S \
        --sets "via_ja3+via_cert_subject" --runs runs/reta_perm_ja3/seed_$S \
        --seeds 41 42 43 44 45
    done

    python no_edge_han.py --datasets bccc_dohbrw --data data/processed_deg2 \
        --sets "via_ja3+via_cert_subject" --out runs/reta_noedge \
        --seeds 41 42 43 44 45

## Budget where the relations carry little

Sec. 6.6 claims the budget matters only where the relations carry something.
The claim needs the other side measured too. `--max-degree` writes the budget
into `hin_summary.json`, so the root is identified by its contents rather than
by its directory name.

    mkdir -p data/_k4_weak
    for DS in hikari cic_andmal vnat; do cp -r data/processed_deg2/$DS data/_k4_weak/$DS; done
    python build_graph.py --datasets hikari cic_andmal vnat --out data/_k4_weak \
        --config config.yaml --skip-materialize --max-degree 4
    python train.py --model han --datasets hikari --data data/_k4_weak \
        --sets "via_dst_host" --runs runs/degree4_weak/hikari --seeds 41 42 43 44 45
    python train.py --model han --datasets vnat --data data/_k4_weak \
        --sets "via_timebin" --runs runs/degree4_weak/vnat --seeds 41 42 43 44 45
    python train.py --model han --datasets cic_andmal --data data/_k4_weak \
        --sets "via_sni+via_ja3+via_cert_subject+via_alpn+via_cert_issuer+via_tls_cipher_group+via_cert_validity" \
        --runs runs/degree4_weak/cic_andmal --seeds 41 42 43 44 45
    cp data/_k4_weak/cic_andmal/hin_summary.json \
        results/sensitivity_weak/cic_andmal_hin_summary.json

The last of the three does not fit on a 24 GB device: seven relations at four
neighbours each is 1.6 M edges per relation. All five seeds are recorded as
failures, which is what `results/sensitivity_weak/cic_andmal.json` holds.

## Two further controls

A flow is an edge in the endpoint graph of E-GraphSAGE, and that edge carries
the flow's own features into exactly the two nodes the classifier then reads.
`--egs-exclude-self` subtracts that message at the last layer, so the flow no
longer counts as its own neighbour.

    python train.py --model egs \
        --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --egs-exclude-self \
        --runs runs/egs_no_self --seeds 41 42 43 44 45

The fifth control on CIC-AndMal, where three fields carry a fill-in string. The
copy is edited in place, so it has to be a copy, and the rebuild has to skip
materialisation or the edit is written over from the captures.

    cp -r data/processed_deg2 data/_nounk_cic
    python mask_synthetic_token.py --data data/_nounk_cic --datasets cic_andmal \
        --column cert_validity_bucket --tokens unknown
    python mask_synthetic_token.py --data data/_nounk_cic --datasets cic_andmal \
        --column cert_issuer_org --tokens PublicCA_Other Private_or_Unknown
    python mask_synthetic_token.py --data data/_nounk_cic --datasets cic_andmal \
        --column tls_cipher_group --tokens Unknown
    python build_graph.py --datasets cic_andmal --out data/_nounk_cic \
        --config config.yaml --skip-materialize
    python train.py --model han --datasets cic_andmal --data data/_nounk_cic \
        --sets "via_sni+via_ja3+via_cert_subject+via_alpn+via_cert_issuer+via_tls_cipher_group+via_cert_validity" \
        --runs runs/nounk_cic --seeds 41 42 43 44 45
    cp data/_nounk_cic/cic_andmal/hin_summary.json \
        results/token_reversal/cic_andmal_hin_summary.json

    for REL in via_sni via_ja3 via_cert_subject via_alpn via_cert_issuer \
               via_tls_cipher_group via_cert_validity via_src_host via_dst_host via_timebin; do
      for DS in iscx_vpn vnat; do
        python train.py --model han --datasets $DS --data data/processed_deg2 \
          --sets $REL --runs runs/relation_ranking/ts/$REL/$DS
      done
    done

## Sensitivity and the literature configuration

    python train.py --model han --datasets bccc_dohbrw --data data/_k4 \
        --runs runs/sensitivity/degree4_bccc
    python train.py --model han --datasets iscx_vpn --data data/_k4 \
        --runs runs/sensitivity/degree4_iscx
    python train.py --model han --datasets vnat --data data/_tb60 \
        --runs runs/sensitivity/tb60
    python train.py --model han --datasets vnat --data data/_tb600 \
        --runs runs/sensitivity/tb600

    python train.py --model han --datasets bccc_dohbrw --data data/processed_deg2 \
        --sets "via_ja3+via_cert_subject" --runs runs/reta/han
    python train.py --model han --datasets bccc_dohbrw --data data/processed_deg2_masked \
        --sets "via_ja3+via_cert_subject" --runs runs/reta/masked
    python no_edge_han.py --datasets bccc_dohbrw --data data/processed_deg2 \
        --sets "via_ja3+via_cert_subject" --out runs/reta/no_edge
    # Permutation for this configuration is NOT the reported-configuration
    # permutation (data/processed_deg2_perm$S shuffles CertValidity, which this
    # configuration does not use). Use the joint shuffle of via_ja3 and
    # via_cert_subject built in "The literature configuration" above
    # (data/_perm_reta$S -> runs/reta_perm_ja3 -> results/reta_approx/permutation/).

## Collect, aggregate, compare

    python collect.py --runs runs --results results

    python reaggregate.py --results results

`reaggregate.py` recomputes every summary block from `per_seed`. It stores the
population standard deviation under `std`, which is what the paper's plus or
minus reports, and the sample one under `std_sample`. The two differ by twelve
percent over five seeds. The `values` copy is rounded to four places, so
`per_seed` is the only correct input to the aggregate.

    python bootstrap_ci.py --results results --out results/model_comparison

    python rank_stability.py --results results --out results/rank_stability

    python make_fig1.py --summaries results/saturation --out fig1_saturation.pdf

## Checks

    python test_graph_invariants.py     # graph construction
    python test_inductive_attach.py     # attachment rule
    python collect.py --results results --check

    python verify_graph.py --datasets bccc_dohbrw iscx_vpn hikari cic_andmal vnat \
        --data data/processed_deg2 --out results/structure/verify_graph.json

Every run now writes a `graph` block beside its scores: edges per relation and
rows per partition. `data_root` is only the string the command was given, and a
directory can be rebuilt underneath a string that never changes — which is how
a table once came to name a graph it had not been run on.

The shipped result files predate that block, so they carry the string and not
the contents. What pins them instead is `results/structure/hin_summary_*.json`,
the builder's own record of every relation it wrote, and `verify_graph.py`,
which rebuilds the edges from that record and compares them byte for byte
against the shipped ones. Runs made from here on carry the fingerprint inside
the result file and do not need the detour.

## Per-dataset conditions

| dataset | relations available | reported relation | note |
|---|---|---|---|
| BCCC-DoH | 6 TLS (SNI yields no edge) | CertValidity | the extractor writes `unknown` when it cannot parse a certificate date |
| ISCX-VPN | 10 | SrcHost | 891 flows, 6 minority in test |
| HIKARI | 2 host only | DstHost | all seven TLS fields have coverage 0.000 |
| CIC-AndMal | 7 TLS | all seven | |
| VNAT | 10 | TimeBin | 2,197 flows, 20 minority in test |

HIKARI carries no TLS field at all, so a search over the ten-relation pool
scores 26 of its 163 configurations at -1.0: those are the configurations built
only from TLS relations, and they produce no edge to train on. They are not
failures, they are configurations this dataset cannot express.

## Notes

Seeds 41 to 45 for training, 42 for graph construction. `config.yaml` fixes
`max_degree_per_value: 2`, which bounds the neighbourhood of a flow rather than
the number of edges a value may create.

Scores are macro-F1 on the test partition unless a file says otherwise.
`availability_rule.py` is the exception and reports three scopes, because the
presence rule changes a lot between them.
