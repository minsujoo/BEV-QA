#!/usr/bin/env bash
# Launch BEV-QA checkpoint evaluations across 4 GPUs, one job per GPU at a time.
# Logs are written to /workspace/BEV-QA/nohup_eval_<run_id>.log

set -euo pipefail

cd /workspace/BEV-QA/BEVDriver/LAVIS

CFG="lavis/projects/bevqa/train.yaml"
SPLITS='["val_test"]'
BASE_LOG="/workspace/BEV-QA/nohup_eval"

# Group checkpoints per GPU (edit as needed)
GPU0=(lavis/out/bevqa/20251121051756/checkpoint_best.pth \
      lavis/out/bevqa/20251128181034/checkpoint_best.pth \
      lavis/out/bevqa/20251204045925/checkpoint_best.pth)
GPU1=(lavis/out/bevqa/20251122103352/checkpoint_best.pth \
      lavis/out/bevqa/20251129171746/checkpoint_best.pth \
      lavis/out/bevqa/20251204090827/checkpoint_best.pth)
GPU2=(lavis/out/bevqa/20251125022301/checkpoint_best.pth \
      lavis/out/bevqa/20251202043134/checkpoint_best.pth \
      lavis/out/bevqa/20251207050658/checkpoint_best.pth)
GPU3=(lavis/out/bevqa/20251127165452/checkpoint_best.pth \
      lavis/out/bevqa/20251203163432/checkpoint_best.pth)

run_seq() {
  local gpu="$1"; shift
  for ckpt in "$@"; do
    local run_id
    run_id=$(basename "$(dirname "$ckpt")")
    local log="${BASE_LOG}_${run_id}.log"
    echo "GPU ${gpu} -> ${ckpt} | log: ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
      python BEVDriver/LAVIS/bevqa_eval_best.py \
        --cfg-path "${CFG}" \
        --ckpt-path "${ckpt}" \
        --options run.world_size=1 run.distributed=false run.skip_generate=false \
                  run.valid_splits=[] run.test_splits="${SPLITS}" run.num_workers=0 \
        > "${log}" 2>&1
  done
}

run_seq 0 "${GPU0[@]}" &
run_seq 1 "${GPU1[@]}" &
run_seq 2 "${GPU2[@]}" &
run_seq 3 "${GPU3[@]}" &

wait
echo "All evaluations finished."
