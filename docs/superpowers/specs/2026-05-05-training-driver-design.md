# Training Driver — PoC Baseline + Ablations (UI-PRMD pickle)

**Date:** 2026-05-05
**Author:** Brainstorming session w/ Claude
**Status:** Approved, pending implementation plan

---

## 1. Context

Move-UP needs a credible baseline number on UI-PRMD before P3 (ST-GCN) starts in week 6. The dataset is already on disk (`model/data/uiprmd/uiprmd_all_exercises.pkl`, ~145 MB) and the loader (`model/uiprmd_pickle_dataset.py`) yields `(N=1908, seq=50, features=117)` clips with continuous quality scores.

We need a training driver that:
- Trains an LSTM regressor on Ex1 (deep squat) + Ex5 (sit-to-stand) end-to-end
- Reports MAE, MSE, Spearman ρ on a held-out test set
- Exposes CLI knobs to enable feature ablations (temporal length, joint subsets) by re-running with different flags

The driver and PyTorch model are **PoC / throwaway**: production stack per Master Plan §3-4 is RTMPose + MotionBERT + ST-GCN. We commit to PyTorch (not TF) so training-loop infrastructure transfers to P3 even though the LSTM model itself does not.

## 2. Goals & Non-Goals

### Goals
- One reusable training driver that produces comparable metrics across ablation runs.
- Stratified train/val/test split (val drives early stopping, test reported once).
- Per-run artifacts (config + metrics + history + weights) saved under `model/runs/<run-name>/`.
- Reproducibility: same seed + same flags + same code → same metrics (modulo MPS non-determinism by default).

### Non-Goals
- Cross-exercise generalization eval (deferred — needs a separate eval-on-other-split step).
- Per-frame baseline ablation (deferred — different model architecture, not LSTM).
- Production-grade tracking (MLflow / W&B / TensorBoard). Stdout + CSV + JSON only.
- Plotting. `history.csv` makes plots trivial to add later.
- Hyperparameter search.

## 3. Architecture

Three new files + edits to two existing files.

```
model/
├── lstm_torch.py             [new] PyTorch nn.Module LSTM regressor
├── training.py               [new] training loop + metrics + seed/device helpers
└── uiprmd_pickle_dataset.py  [edit] add train_val_test_split()

scripts/
└── train_baseline.py         [new] CLI + glue (load → split → model → fit → save)

pyproject.toml                [edit] add torch>=2.4, scipy>=1.11
docs/progression.md           [edit] per CLAUDE.md mandatory workflow
```

Each module has one job and is testable / importable in isolation. The training loop in `training.py` knows nothing about LSTM specifics — it works on any `nn.Module` with `forward(x) -> (batch, 1)`. This is the property that lets us reuse `fit()` for ST-GCN in P3 unchanged.

## 4. `model/lstm_torch.py` — model

```
Input:   (batch, seq_len=50, num_features=117)
   ↓ LSTM(input=num_features, hidden=64, batch_first=True) — last hidden
   ↓ Dropout(0.3)
   ↓ LSTM(input=64, hidden=32, batch_first=True) — last hidden
   ↓ Linear(32 → 16) + ReLU
   ↓ Linear(16 → 1) + Sigmoid
Output:  (batch, 1) ∈ [0, 1]
```

- Mirrors the architecture in `model/scoring_model.py` (TF) so direct comparison is meaningful if we ever re-run TF.
- Sigmoid output matches the score range in the pickle (~0.22 → 0.98).
- Configurable: `num_features`, `hidden_1`, `hidden_2`, `dropout`. `seq_len` is implicit in input shape.
- Loss: `nn.MSELoss`. Optimizer: `torch.optim.Adam(lr=1e-3)`. Both default; CLI-overridable.

## 5. `model/training.py` — training loop + metrics

Public surface:
```python
def set_seed(seed: int, strict: bool = False) -> None
def make_device() -> torch.device     # mps if available else cpu (cuda path included)
def train_one_epoch(model, loader, opt, loss_fn, device) -> float           # mean train loss
def evaluate(model, loader, device) -> dict                                  # {"mae","mse","spearman"}
def fit(model, train_loader, val_loader, *,
        epochs, lr, patience, device, log_every=1) -> dict                   # history + best_state
```

### Training loop
- Standard PyTorch loop, batched. `model.train()` / `model.eval()` toggled correctly.
- Validation runs every epoch.
- **Early stopping** on `val_mae` with configurable `patience` (default 10), best state-dict cached and restored at end. Mirrors `scoring_model.py:138` TF `EarlyStopping`.
- No LR scheduling for the baseline (lean). Easy to add later if a run plateaus visibly.

### Metrics
- **MAE / MSE** — `torch` ops, no extra deps.
- **Spearman ρ** — `scipy.stats.spearmanr` on concatenated predictions vs. labels.
- **Critical**: metrics computed on the *un*-batched concatenated tensor, NOT averaged across batches — averaging across batches gives wrong Spearman.

### Reproducibility
- `set_seed` seeds `random`, `numpy`, `torch.manual_seed`, `torch.mps.manual_seed` (if MPS).
- `strict=True` additionally sets `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG` (no-op on MPS but harmless). Default off — MPS doesn't fully support deterministic mode and we'd lose throughput.

## 6. `model/uiprmd_pickle_dataset.py` — `train_val_test_split()` extension

New function that calls the existing two-way `train_test_split()` twice:
1. Peel 20% test (stratified by exercise id).
2. Peel 20% val from the remaining 80% train pool (stratified, separate seed offset).

Result: **64% train / 16% val / 20% test**. Deterministic given seed. Returns six arrays: `X_train, X_val, X_test, y_train, y_val, y_test`. The existing `train_test_split` stays untouched and remains usable for two-way splits in notebooks.

## 7. `scripts/train_baseline.py` — CLI

```
usage: train_baseline.py [options]

Data:
  --exercises {squat,sit_to_stand,both}     default: both
  --seq-len INT                              default: 50  (slice from front of clip)
  --joint-subset {all,random,top-variance}   default: all
  --joint-subset-size INT                    default: 39  (used when subset != all)
  --joint-subset-seed INT                    default: 0   (for random subset)

Training:
  --epochs INT                               default: 80
  --batch-size INT                           default: 32
  --lr FLOAT                                 default: 1e-3
  --patience INT                             default: 10
  --hidden-1 INT                             default: 64
  --hidden-2 INT                             default: 32
  --dropout FLOAT                            default: 0.3

Run mgmt:
  --seed INT                                 default: 42
  --run-name STR                             default: auto-from-flags
  --out-dir PATH                             default: model/runs/
  --strict-deterministic                     default: off
  --no-save-weights                          default: save
```

### Flag → ablation mapping
| Ablation | Flags |
|---|---|
| Baseline | (defaults) |
| Temporal length | `--seq-len 25`, `--seq-len 10` |
| Random joint subset | `--joint-subset random --joint-subset-size 20` |
| Top-variance joint subset | `--joint-subset top-variance --joint-subset-size 20` |

### Auto-generated `--run-name`
Format: `<date>_<exercises>_seq<L>_j<subset>_h<H1>-<H2>_s<seed>`
Example: `2026-05-05_both_seq50_jall_h64-32_s42`. Makes `ls model/runs/` immediately legible.

### Joint subset implementation
- `all` (default): pass through.
- `random`: deterministic numpy permutation of 39 joints with `--joint-subset-seed`, take first `--joint-subset-size`. Reshape `(N, T, J, 3)` → `(N, T, S, 3)` → flatten last two dims → `(N, T, S*3)`. Pass `S*3` to `num_features`.
- `top-variance`: compute per-joint variance across the train pool only (no test leakage), rank, take top-S. Same reshape path.

## 8. Outputs

```
model/runs/<run-name>/
├── config.json          # CLI flags + git SHA + python/torch versions
├── metrics.json         # final test metrics + best-epoch val metrics
├── history.csv          # per-epoch: train_loss, val_mae, val_mse, val_spearman
└── weights.pt           # best-by-val-mae state dict (omitted if --no-save-weights)
```

**Stdout during training:** one line per epoch:
```
epoch  03 | train_loss=0.0162 val_mae=0.0598 val_mse=0.0061 val_spearman=0.412 *
```
(`*` marks new best val_mae.) Final test summary printed as a 4-line block.

No tqdm bar — single-line-per-epoch keeps logs grep-able.

## 9. Dependencies

Added to `pyproject.toml`:
```
torch>=2.4.0
scipy>=1.11
```
Not added: `pandas`, `tqdm`, `matplotlib`, `mlflow`, `wandb`. All optional for the PoC.

## 10. Acceptance criteria

The driver is "done" when:

1. `uv run python scripts/train_baseline.py` runs end-to-end on the default config without errors and produces a `model/runs/<auto-name>/` directory containing all four artifacts.
2. Final test metrics are printed to stdout and saved to `metrics.json`.
3. The same command rerun with `--seed 42 --strict-deterministic` produces bit-identical metrics on CPU. (MPS path: tiny float drift acceptable in non-strict mode; bit-identical not guaranteed.)
4. At least one ablation flag (e.g. `--seq-len 25`) produces a different `metrics.json` than baseline, confirming the knob is wired through.
5. `docs/progression.md` updated per CLAUDE.md mandatory workflow.

## 11. Deferred / future work

- **Cross-exercise generalization** eval (train Ex1, test Ex5): needs a separate `scripts/eval_cross.py` that loads weights and runs `evaluate()` on a held-out exercise subset.
- **Per-frame (no-temporal) baseline**: requires an MLP-on-mean architecture, not LSTM. Different model file.
- **LR scheduling**: `ReduceLROnPlateau` if a baseline plateau visible.
- **Plotting helper**: `scripts/plot_history.py` reading `history.csv`. One-off.
- **Production stack migration**: this whole driver is throwaway when ST-GCN training infra is built in P3. The `fit()` / `evaluate()` / split helpers are designed to transfer cleanly; the LSTM model is not.

## 12. Risks

- **MPS quirks**: occasional `nan` losses or hangs reported on Apple Silicon for some LSTM configs. Mitigation: `--strict-deterministic` falls back to a slower path; if MPS misbehaves we can force CPU via env var.
- **Score distribution skew**: Ex1 scores cluster near 0.95 (median ~0.92, only 5 samples below 0.66). MAE will be optimistic; Spearman ρ is the more honest metric for clinical fidelity. Document this in the metrics interpretation.
- **39 unnamed Vicon markers**: top-variance ablation will pick *some* subset but we can't claim "knee-related joints matter most" without joint-name metadata. Limits the interpretability of the ablation result. Acceptable for PoC.
