#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export DDIB_DATA_DIR="${DDIB_DATA_DIR:-./data}"
export CPU_THREADS="${CPU_THREADS:-2}"
SEEDS="${SEEDS:-42 66 100}"
LOG_DIR="${LOG_DIR:-logs/mosei_3seed_all}"
EPOCHS="${EPOCHS:-300}"
STAGE_EPOCH="${STAGE_EPOCH:-100}"
mkdir -p "${LOG_DIR}"

for seed in ${SEEDS}; do
  python -u Train_DDIB.py \
    --dataset CMUMOSEI \
    --audio-feature wav2vec-large-c-UTT \
    --text-feature deberta-large-4-UTT \
    --video-feature manet_UTT \
    --gpu "${GPU:-0}" \
    --epochs "${EPOCHS}" \
    --stage_epoch "${STAGE_EPOCH}" \
    --batch-size "${BATCH_SIZE:-32}" \
    --hidden 256 \
    --depth 4 \
    --num_heads 2 \
    --drop_rate 0.5 \
    --lr 0.0001 \
    --l2 0.00001 \
    --seed "${seed}" \
    --test_condition all \
    --checkpoint-selection best_f1 \
    --ib-beta 0.0005 \
    --ib-std-offset -5 \
    --adaptive-multi-ib \
    --multi-ib-std-offset-t -2 \
    --multi-ib-std-offset-fusion -2 \
    --uni-aib-residual-alpha 0.20 \
    --stage1-joint-weight 1.0 \
    --restore-best-stage1-attention \
    --DDIB-uni \
    --DDIB-uni-beta 0.0005 \
    --DDIB-aux-weight 0.2 \
    --DDIB-fusion \
    --DDIB-fusion-beta 0.0002 \
    --DDIB-fusion-residual-mode gated_delta_lowrank_add \
    --DDIB-fusion-gate-init 0.05 \
    --DDIB-fusion-alpha-warmup 10 \
    --DDIB-fusion-start-epoch 2 \
    --second-stage-ib-warmup 10 \
    --condition-adapter \
    --condition-adapter-rank 8 \
    --condition-adapter-reg-lambda 0.00001 \
    --condition-denoise-gates \
    --condition-gate-reg-lambda 0.0001 \
    --DDIB-deterministic-eval \
    --ema-decay 0.99 \
    --warmup-epochs 5 \
    > "${LOG_DIR}/seed${seed}.log" 2>&1
done
