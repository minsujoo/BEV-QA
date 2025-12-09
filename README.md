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
- Dataset: Bench2Drive_Base (RGB/LiDAR) + Chat‑B2D (multi‑turn QA). `val`을 0.5/0.5로 나눠 `val_dev`(검증/early stop) + `val_test`(테스트 대용) 사용. `lavis/projects/bevqa/train.yaml`에서 sensor/language 루트를 `/workspace/Bench2Drive_Base`, `/workspace/chat-B2D/{train,val}`로 설정.
- 현재 기본 설정: `first_sentence_only=True`, `max_txt_len=192`, 디코딩 `max_new_tokens=64/min_new_tokens=8`, 프롬프트는 단문 요약.
- 4 GPU 예시:
  ```bash
  conda activate bevdriver
  cd /workspace/BEV-QA/BEVDriver/LAVIS
  python -m torch.distributed.run --nproc_per_node=4 --master_port=29500 \
    train.py --cfg-path lavis/projects/bevqa/train.yaml
  ```

평가 (텍스트 생성 포함)
- 단일 체크포인트:
  ```bash
  cd /workspace/BEV-QA/BEVDriver/LAVIS
  python bevqa_eval_best.py \
    --cfg-path lavis/projects/bevqa/train.yaml \
    --ckpt-path lavis/out/bevqa/<job_id>/checkpoint_best.pth \
    --options run.world_size=1 run.distributed=false run.skip_generate=false \
              run.valid_splits=[] run.test_splits=["val_test"] run.num_workers=0
  ```
  결과 JSON은 `lavis/out/bevqa/<eval_job_id>/val_test_bevqa_epoch*.json`에 저장.
- 여러 체크포인트 일괄 평가(1 GPU당 순차):
  ```bash
  cd /workspace/BEV-QA
  python BEVDriver/tools/eval/batch_bevqa_eval.py --gpus 0,1,2,3
  ```
  로그: `/workspace/BEV-QA/nohup_eval_<run_id>.log`.

지표 계산
- Pseudo-SPICE(토큰 F1), ROUGE-L, chrF, BERTScore 지원:
  ```bash
  python BEVDriver/tools/eval/spice_eval.py \
    --file BEVDriver/LAVIS/lavis/out/bevqa/<eval_job_id>/val_test_bevqa_epoch*.json \
    --metrics pseudo,rougeL,chrf,bertscore \
    --bert-model microsoft/deberta-base-mnli
  ```
  (필요 패키지: `pip install rouge-score sacrebleu bert-score`; 네트워크 불가 시 캐시에 있는 작은 모델로 BERTScore 실행 또는 제외)

시각화
- 이미지와 예측/정답 텍스트를 PNG로 렌더링:
  ```bash
  python BEVDriver/tools/visualize_bevqa.py \
    --result BEVDriver/LAVIS/lavis/out/bevqa/<eval_job_id>/val_test_bevqa_epoch*.json \
    --sensor-root /workspace/Bench2Drive_Base \
    --out-dir /workspace/BEV-QA/vis_bevqa \
    --num 10 --shuffle
  ```

License and acknowledgements
- See `LICENSE`. This work builds on BEVDriver, LAVIS, and related open‑source projects.


## License
All code within this repository is under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)
