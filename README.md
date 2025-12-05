# BEV‑QA: BEV‑based Visual Question Answering for Autonomous Driving

BEV‑QA adapts the BEVDriver architecture from waypoint prediction to autonomous‑driving VQA. We reuse the strong BEV encoder (multi‑view cameras + LiDAR) and Q‑Former alignment, and switch the output head to an auto‑regressive LLM that generates textual answers.

Key capabilities
- BEV encoder + Q‑Former alignment reused from BEVDriver.
- Causal LLM (OPT or LLaMA) generates VQA answers conditioned on BEV tokens and questions.
- Training objective is language modeling cross‑entropy on answer tokens.

Project status
- Implemented:
  - Model and registry wiring (`BEVDriver/LAVIS/lavis/models/drive_models/bevqa.py`)
  - Bench2Drive + Chat-B2D dataset loader + builder
  - Training config (`lavis/projects/bevqa/train.yaml`) with `train` + `val_dev`/`val_test` split and early stopping
  - Evaluation script for best checkpoints (`BEVDriver/LAVIS/bevqa_eval_best.py`)
  - Pseudo‑SPICE evaluation wrapper (`BEVDriver/tools/eval/spice_eval.py`)
- See `docs/BEV-QA.md` for detailed architecture and training/eval instructions.

Repository structure (selected)
- `BEVDriver/LAVIS/lavis/models/drive_models/bevqa.py`: BEV‑QA model.
- `BEVDriver/timm/models/bevdriver_encoder.py`: BEV encoder (reused).
- `docs/BEV-QA.md`: architecture and implementation guide.
- `AGENTS.md`: contributor guidelines.

Setup
```
conda create -n bevdriver python=3.8
conda activate bevdriver
pip install -r BEVDriver/requirements.txt
cd BEVDriver/LAVIS && pip install -r requirements.txt && cd -
# SPICE/metrics 평가용 추가 의존성
# - Java 런타임 (SPICE 내부 Stanford CoreNLP 호출): conda install -c conda-forge openjdk   또는   sudo apt-get install -y openjdk-11-jre-headless
# - 평가 지표: pip install bert_score rouge_score  (requirements에 포함되어 있으므로 설치만 확인)
```
Notes
- Use the bundled `LAVIS`/`timm` in this repo; do not `pip install` upstream replacements.
- CARLA is optional for BEV‑QA; not required for offline training/eval.

Quick start (inference)
```python
from lavis.common.registry import registry
cfg = {
  "encoder_model": "bevdriver_encoder",
  "encoder_model_ckpt": "/path/to/encoder.ckpt",
  "llm_model": "/path/to/llama-or-opt",
  "max_txt_len": 64,
  "num_query_token": 32,
}
Model = registry.get_model_class("bevqa")
model = Model.from_config(cfg).cuda()

samples = {
  "rgb": rgb, "rgb_left": rgb_l, "rgb_right": rgb_r, "rgb_center": rgb_c,
  "lidar": lidar, "measurements": meas,
  "vqa_question": ["Is the traffic light red?"],
}
answers = model.generate(samples, max_new_tokens=16)
```

Training
- Dataset: Bench2Drive_Base (RGB/LiDAR) + Chat‑B2D (multi‑turn driving QA). See `docs/BEV-QA.md` for the expected folder layout.
- Loss: token‑level cross‑entropy on answer text (teacher‑forcing).
- Execution (single GPU example):
  ```bash
  conda activate bevdriver
  cd BEVDriver/LAVIS
  python -m torch.distributed.run --nproc_per_node=1 --master_port=12345 \
    train.py --cfg-path lavis/projects/bevqa/train.yaml
  ```
  For multi-GPU training, increase `--nproc_per_node` (e.g., 4) and optionally set `run.world_size` in the config.

Validation / SPICE-style evaluation
- During training, the `bevqa_drive` task can run validation on the configured split(s) and store `{id, pred, ref}` JSON under `BEVDriver/LAVIS/lavis/out/bevqa/<job_id>/`.
- After training, you can evaluate a specific checkpoint (e.g., best epoch) with:
  ```bash
  conda activate bevdriver
  cd BEVDriver/LAVIS
  python bevqa_eval_best.py \
    --cfg-path lavis/projects/bevqa/train.yaml \
    --ckpt-path lavis/out/bevqa/<job_id>/checkpoint_best.pth \
    --options run.skip_generate=false run.valid_splits=[\"val_dev\"] run.test_splits=[\"val_test\"]
  ```
- This writes JSON files such as `val_dev_bevqa_epoch{epoch}.json` and `val_test_bevqa_epoch{epoch}.json` under `BEVDriver/LAVIS/lavis/out/bevqa/<eval_job_id>/`.
- Run the pseudo-SPICE wrapper on the held‑out split (e.g. `val_test`):
  ```bash
  python BEVDriver/tools/eval/spice_eval.py \
    --file BEVDriver/LAVIS/lavis/out/bevqa/<eval_job_id>/val_test_bevqa_epoch{epoch}.json
  ```

License and acknowledgements
- See `LICENSE`. This work builds on BEVDriver, LAVIS, and related open‑source projects.


## License
All code within this repository is under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
