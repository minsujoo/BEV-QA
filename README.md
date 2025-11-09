# BEV‑QA: BEV‑based Visual Question Answering for Autonomous Driving

BEV‑QA adapts the BEVDriver architecture from waypoint prediction to autonomous‑driving VQA. We reuse the strong BEV encoder (multi‑view cameras + LiDAR) and Q‑Former alignment, and switch the output head to an auto‑regressive LLM that generates textual answers.

Key capabilities
- BEV encoder + Q‑Former alignment reused from BEVDriver.
- Causal LLM (OPT or LLaMA) generates VQA answers conditioned on BEV tokens and questions.
- Training objective is language modeling cross‑entropy on answer tokens.

Project status
- Implemented: model and registry wiring (`lavis/models/drive_models/bevqa.py`), docs.
- Pending: SimLingo dataset loader, training config, SPICE evaluation wrapper. See `docs/BEV-QA.md`.

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

Training (coming soon)
- Dataset: SimLingo VQA (RGB/LiDAR/questions/answers). A dataset loader and `lavis/projects/bevqa/train.yaml` will be added.
- Loss: token‑level cross‑entropy on answer text (teacher‑forcing). See `docs/BEV-QA.md`.

Evaluation (coming soon)
- VQA text metrics with SPICE. A wrapper will be provided under `BEVDriver/tools/eval/spice_eval.py`.

License and acknowledgements
- See `LICENSE`. This work builds on BEVDriver, LAVIS, and related open‑source projects.


## License
All code within this repository is under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
