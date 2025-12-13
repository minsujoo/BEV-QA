# BEV-QA: BEV-based Visual Question Answering for Autonomous Driving

BEV-QA adapts the BEVDriver encoder + Q-Former stack to a causal LLM head that generates answers for driving scenes (multi-view cameras + LiDAR). The project lives inside the bundled LAVIS fork; see `docs/BEV-QA.md` for architecture details and `AGENTS.md` for contributor rules.

## Repository layout (selected)
- `BEVDriver/LAVIS/lavis/models/drive_models/bevqa.py`: BEV-QA model/registry.
- `BEVDriver/LAVIS/lavis/projects/bevqa/train.yaml`: default train/eval config.
- `BEVDriver/LAVIS/bevqa_eval_best.py`: single-checkpoint evaluation (text generation).
- `run_all_bevqa_eval.sh`: batch evaluation across many checkpoints/GPU slots.
- `compute_bevqa_metrics.py`: metrics-only runner (pseudo-SPICE, BERTScore, ROUGE-L, chrF).
- `BEVDriver/timm/models/bevdriver_encoder.py`: BEV encoder.

## Setup
```bash
conda create -n bevdriver python=3.8
conda activate bevdriver
pip install -r BEVDriver/requirements.txt
cd BEVDriver/LAVIS && pip install -r requirements.txt && cd -
```
Notes: use the bundled `LAVIS`/`timm` in this repo (no upstream pip installs). CARLA is not required for offline BEV-QA.

## Data
- Bench2Drive_Base RGB/LiDAR and Chat-B2D QA. Set paths in `lavis/projects/bevqa/train.yaml` (`sensor_root`, `language_root`).
- The provided `val` is split 50/50 into `val_dev` (early stop) and `val_test` (reporting) with seed 42.

### Download datasets
You must download datasets manually (no auto-fetch in scripts):
- Chat-B2D QA: https://huggingface.co/datasets/poleyzdk/Chat-B2D
- Bench2Drive: https://huggingface.co/datasets/rethinklab/Bench2Drive

Example (huggingface-cli):
```bash
huggingface-cli download --repo-type dataset poleyzdk/Chat-B2D --local-dir /workspace/chat-B2D
huggingface-cli download --repo-type dataset rethinklab/Bench2Drive --local-dir /workspace/Bench2Drive_Base
```
Then point `sensor_root` and `language_root` in `lavis/projects/bevqa/train.yaml` to these local folders.

## Checkpoints (place manually)
- BEV encoder: download your pretrained BEVDriver encoder and set `model.encoder_model_ckpt` in `train.yaml` (example path: `weights/bevdriver/checkpoint_best.pth`).
- LLM: place the language model weights (e.g., LLaMA-7B) at `weights/llama-7b` and set `model.llm_model` accordingly.
- No checkpoints are auto-downloaded; copy them into the `weights/` folder or update the config paths to your locations.

### Download checkpoints
- LLaMA-7B (HuggingFace): https://huggingface.co/huggyllama/llama-7b (observe the model license; place under `weights/llama-7b`).
- BEVDriver main model checkpoint: https://syncandshare.lrz.de/dl/fiWRzThZRF4xY6DN2Ets7/Model%20Checkpoints/Main%20Model%20Llama-7b/checkpoint_best.pth (place at `weights/bevdriver/checkpoint_best.pth` or update `train.yaml`).

## Training
Edit `train.yaml` for paths (`encoder_model_ckpt`, `llm_model`, data roots). Example (4 GPUs, defaults: seed 42, world_size 4):
```bash
conda activate bevdriver
cd /workspace/BEV-QA/BEVDriver/LAVIS
torchrun --nproc_per_node=4 --master_port=29500 train.py --cfg-path lavis/projects/bevqa/train.yaml
```
Outputs land in `out/bevqa/<job_id>/` with TensorBoard/logs/checkpoints.

## Evaluation (text generation)
Single checkpoint → generate answers on `val_test`:
```bash
cd /workspace/BEV-QA/BEVDriver/LAVIS
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 \
python bevqa_eval_best.py \
  --cfg-path lavis/projects/bevqa/train.yaml \
  --ckpt-path lavis/out/bevqa/<run_id>/checkpoint_best.pth \
  --job-id <run_id> \
  --options run.world_size=1 run.distributed=false run.skip_generate=false \
            run.valid_splits=[] run.test_splits=["val_test"] run.num_workers=0
```
Results are saved to `out/bevqa/<run_id>/result/val_test_bevqa_epoch*.json` in the same run directory. If `--job-id` is omitted, a timestamped eval folder is created.

## Batch evaluation (multiple checkpoints)
`run_all_bevqa_eval.sh` runs one checkpoint per GPU at a time and logs to `/workspace/BEV-QA/nohup_eval_<run_id>.log`.
1) Edit the GPU arrays in the script to list your `checkpoint_best.pth` paths and desired `SPLITS` (e.g., `["val_test"]`).
2) Run from repo root: `nohup bash run_all_bevqa_eval.sh >/dev/null 2>&1 &`
Each checkpoint writes its outputs into its own run directory under `result/`.

## Metrics-only (pseudo-SPICE, BERTScore, ROUGE-L, chrF)
Use `compute_bevqa_metrics.py` on an existing prediction JSON (no generation):
```bash
cd /workspace/BEV-QA
python compute_bevqa_metrics.py \
  --input BEVDriver/LAVIS/lavis/out/bevqa/<run_id>/result/val_test_bevqa_epochX.json \
  --pseudo-spice --chrf --bert-model roberta-large --no-spice   # add/remove flags as needed
```
- Outputs: a sibling `<name>_metrics.json` with metrics + count.
- `--pseudo-spice` enables a lightweight token-overlap F1; set `--no-spice` to skip Java/pycocoevalcap; enable full SPICE only if Java + pycocoevalcap are installed.
- `--chrf` uses sacrebleu; BERTScore/ROUGE-L are on by default unless `--no-bertscore/--no-rouge`.

## Repro tips
- Seeds/default splits are defined in `train.yaml` (`seed: 42`, `val_split` section).
- Keep `TRANSFORMERS_OFFLINE=1` if you lack network; adjust `MASTER_PORT` to avoid conflicts when using `torchrun`.
- Use the `result/` subfolder in each run directory for all eval artifacts (JSON + metrics); avoid mixing across runs.

## License
Apache 2.0 (see `LICENSE`). Builds on BEVDriver, LAVIS, and related open-source work.
