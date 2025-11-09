# BEV-QA 아키텍처 및 구현 가이드

본 문서는 기존 BEVDriver를 자율주행 VQA(Visual Question Answering) 태스크로 전환하기 위한 설계·구현 지침입니다. 핵심 아이디어는 BEV 인코더와 Q‑Former 정렬 모듈을 재사용하고, 출력 목표를 웨이포인트 회귀에서 텍스트 생성으로 변경하는 것입니다.

## 목표 요약
- 기존: BEV 특징 → Waypoint 예측(L1/MSE)
- 변경: BEV 특징 + 질문 → VQA 답변 텍스트 생성(언어모델링 CE)
- 데이터: SimLingo RGB, LiDAR, VQA(drivelm)

## 디렉터리 및 신규 모듈
- 재사용
  - BEV 인코더: `BEVDriver/timm/models/bevdriver_encoder.py`
  - Q‑Former/Resampler: 기존 정렬 모듈(LAVIS 내) 재사용
- 제거(출력단)
  - Waypoint 헤드(예: GRU/MLP/WaypointAdapter) 및 관련 손실/로직
- 신규(제안 경로)
  - 모델: `BEVDriver/LAVIS/lavis/models/drive_models/bevqa.py`
  - 데이터셋: `BEVDriver/LAVIS/lavis/datasets/datasets/simlingo_vqa.py`
  - 설정: `BEVDriver/LAVIS/lavis/projects/bevqa/train.yaml`
  - 평가(SPICE 래퍼): `BEVDriver/tools/eval/spice_eval.py`

## 현재 구현 상태
- 아키텍처(완료)
  - 신규 모델 `bevqa` 등록 및 구현: `BEVDriver/LAVIS/lavis/models/drive_models/bevqa.py`
  - LAVIS 레지스트리 연결: `BEVDriver/LAVIS/lavis/models/__init__.py`에 `BEVQAModel` 추가
  - 동작: BEV 인코더 → Q‑Former → LLM로 연결, teacher‑forcing CE 학습 및 `generate()` 지원
  - 입력 키: `rgb`, `rgb_left`, `rgb_right`, `rgb_center`, `lidar`, `measurements`, `vqa_question`(list[str]), `vqa_answer`(list[str])
  - 출력: 학습 `{"loss": lm_loss}`, 평가 `List[str]` 답변
- 미구현(다음 단계)
  - SimLingo VQA 데이터셋 로더/빌더, collate
  - 프로젝트 설정 파일 `lavis/projects/bevqa/train.yaml`
  - SPICE 평가 래퍼 `tools/eval/spice_eval.py`

### 간단 사용 예(Python)
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
# train: model(samples) -> {"loss": ...}
# eval:  model.generate(samples, max_new_tokens=32)
```

## 모델 변경 사항(요구 동작)
1) 입력단(유지)
- BEV 인코더로 RGB+LiDAR 융합 특징 추출 → Q‑Former로 `bev_tokens` 생성

2) 출력단(제거)
- Waypoint 관련 헤드/로스/`forward` 분기 삭제

3) VQA(신규)
- 학습 입력: `{rgb_images, lidar_points, vqa_question, vqa_answer}`
- 토큰 구성: `bev_tokens` + `question_tokens` → LLM 디코더로 teacher‑forcing
- 손실: 언어모델링 CrossEntropyLoss(`ignore_index=pad_id`, 1‑토큰 시프트)
- 추론: `generate()`에서 `bev_tokens`+질문으로 auto‑regressive 답변 생성(beam/top‑p 선택)

예시 시그니처(개념):
```python
# train
loss, out = model(rgb, lidar, question_text, answer_text)
# eval
answers = model.generate(rgb, lidar, question_text, max_len=32, top_p=0.9)
```

## 데이터 로딩(SimLingo)
- 신규 Dataset: `simlingo_vqa.py`
- `__getitem__` 반환 값
  - `rgb_images`: 멀티뷰 텐서/리스트
  - `lidar_points`: 포인트클라우드 또는 BEV 변환 입력
  - `vqa_question`: 문자열
  - `vqa_answer`: 문자열(학습용 라벨)
- Collate
  - 이미지/라이다 배치 스택, 텍스트는 토크나이즈
  - `labels`: `[BOS] 정답 ... [EOS]`를 1‑토큰 시프트, pad는 `ignore_index`

## 학습 및 설정
- 손실: CE(언어모델링). 옵티마/스케줄러는 기존(AdamW, cosine 등) 재사용 가능
- 예시 실행
```bash
conda activate bevdriver
cd BEVDriver/LAVIS
python -m torch.distributed.run \
  --nproc_per_node=1 --master_port=12345 \
  train.py --cfg-path lavis/projects/bevqa/train.yaml
```
- `train.yaml` 핵심 항목(예)
  - `datasets: simlingo_vqa`
  - `model: bevqa`(인코더 경로, LLM 토크나이저, max_txt_len, freeze 백본 여부)
  - `optimization: batch_size, lr, epochs`

## 평가(SPICE)
1) 검증 단계 수집
- `validation_step`: `generated_answer`와 `vqa_answer`를 함께 저장
- `on_validation_epoch_end`: `{id, pred, ref}` JSON 생성

2) SPICE 실행(래퍼)
```bash
python BEVDriver/tools/eval/spice_eval.py \
  --pred outputs/val_preds.json --ref outputs/val_refs.json
```
- SPICE 외부 라이브러리/스크립트를 래퍼에서 호출하도록 구성(벤더링 권장)

## 구현 체크리스트
- [ ] Waypoint 헤드/로스 제거 및 `forward` 정리
- [ ] SimLingo Dataset/Collate/토크나이저 구현
- [ ] LLM `generate()` 경로 및 CE 손실 적용
- [ ] SPICE 평가 래퍼 연동 및 메트릭 로깅
- [ ] `train.yaml` 하이퍼파라미터/경로 정리

## 주의 사항
- 번들된 `LAVIS`/`timm` 사용(업스트림 pip 패키지로 교체 금지)
- 체크포인트/데이터/비밀키 커밋 금지, 경로는 설정 파일/인자화
- CARLA는 오프라인 VQA 학습에 필수 아님(시뮬 연동 시 별도 설정)
