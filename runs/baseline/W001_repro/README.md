# W001_repro — rank1 euijin42 GOH30 reproduce

plan: `plans/plan-w-001-rank1-euijin42.md`
config: `notes/winners/rank1-euijin42/repo/GOH30_reproduce.py`
meta: `notes/winners/rank1-euijin42/meta.yaml`

## Run

```
CUDA_VISIBLE_DEVICES=1 DATA_DIR=./data MODELS_DIR=./models_goh30 \
N_EACH=10 GRU_ODE_EPOCHS=55 H_EPOCHS=12 \
SUBMISSION_OUT=./submission_GOH30.csv FROM_SCRATCH=True \
PYTHONPATH=. python notes/winners/rank1-euijin42/repo/GOH30_reproduce.py \
> runs/baseline/W001_repro/train.log 2>&1
```

device: L40S (cuda) · started 2026-06-05 04:14 KST · PID 17310

## Artifacts (gitignored, local-only)

- `models_goh30/phaseG_full_{0..9}.pt` (10 AttnGRU)
- `models_goh30/phaseODE_full_{0..9}.pt` (10 Neural ODE)
- `models_goh30/phaseH_full_{0..9}.pt` (10 HyperPhysics)
- `submission_GOH30.csv` (10000 row 등가중 30모델 blend)
- `train.log` (학습 로그)

## Verification

target: notebook self-LB private 0.7035 ± 0.005 (= 0.6985 ~ 0.7085).
LB Δ + final result → `plans/plan-w-001-rank1-euijin42.results.md`.
