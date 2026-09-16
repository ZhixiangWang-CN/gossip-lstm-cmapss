# Serverless gossip training of LSTM failure detectors on NASA C-MAPSS

Code, data and results for the paper

> Y. Öztürk, E. Göktekin, B. Atlı, A. Öztürk, Z. Wang, U. Bagci. *Serverless Gossip Training of LSTM Networks for
> Imminent-Failure Detection: A Matched-Protocol Comparison with Federated and Centralized Learning on NASA C-MAPSS.*
> (under review)

The repository implements, in one PyTorch code base, four ways of training the same stacked LSTM (100/50 units,
dropout 0.2) to detect imminent failure (RUL ≤ 30 cycles) on C-MAPSS FD001:

| Method | Aggregation |
|---|---|
| `local` | none — each node trains on its own engines |
| `gossip` | synchronous ring gossip, Metropolis weights (1/3 self/left/right) after every local epoch |
| `fedavg` | FedAvg with sample-count weighting after every local epoch |
| `centralized` | single model on pooled data (reference) |

Variants: class-weighted loss (`*_weighted`), event-triggered gossip (`event_gossip*`), sample-mass-aware gossip
(`mass_gossip`). Network scenarios: message loss, node failure, server outage, delay, intermittent availability;
lifetime-sorted non-IID partitions; 5–40 nodes. Every run records its configuration, data hashes, training history,
consensus diagnostics, a per-message communication ledger and node-level confusion counts.

The full protocol (labels, splits, endpoints, statistics, and what is and is not modelled) is in
[`docs/protocol.md`](docs/protocol.md).

## Installation

Python 3.10–3.12, CPU only.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run_pdm.py doctor
```

Versions used for the paper: torch 2.14 (CPU), numpy 2.3, pandas 2.2, scipy 1.17, scikit-learn 1.8, matplotlib 3.10.

## Reproducing the paper

```bash
python run_pdm.py audit                                  # dataset checks: 93 terminal / 8,255 test windows
python run_pdm.py run --suite pilot                      # quick software check (5 epochs)
python run_pdm.py run --suite core --threads 4           # main study: 8 methods x 5 seeds, 50 epochs
python run_pdm.py verify                                 # recompute metrics/bytes from saved outputs
python scripts/run_extended.py output                    # exploratory scenarios (3 seeds)
python scripts/analyze.py --runs output --out analysis   # summary statistics and Figures 4-8
```

The analysis script also runs directly on the released results and reproduces `results/summary.json` exactly:

```bash
python scripts/analyze.py --runs results/runs --out analysis
```

Runs are resumable (`--max-jobs N` bounds a session; repeating a command skips completed jobs). A 50-epoch run takes
roughly 3–5 CPU minutes. `python -m pytest tests` runs the protocol unit tests (9 tests).

## Released results

`results/` contains the outputs used in the paper:

- `all_results.csv`, `seed_summary.csv`, `paired_tests.csv` — per-run metrics, seed-level summaries, exact paired
  sign-flip tests
- `tables.json`, `summary.json` — values reported in Tables 1–4 and the text
- `figures/` — Figures 4–8
- `runs/<dataset>_<method>_s<seed>_n<nodes>_<hash>/` — per-run `config.json`, `results.json`, `history.csv`,
  `node_metrics.csv`, `node_data.csv`, `data_audit.json`

Large per-run files (model checkpoints, per-window predictions, message ledgers) are not committed; they are regenerated
by the commands above. Each `config.json` records `software_sha256` (`7d748ee1…`), a hash of the experiment code at the
time the runs were produced; the released code differs from that snapshot only in report text, a code comment and the
dataset-audit command, not in data handling, training or evaluation.

## Data

`data/` holds the NASA C-MAPSS FD001 files; see [`data/README.md`](data/README.md) for source and citation.

## License

MIT (code). The C-MAPSS data are distributed by NASA under their original terms.
