# 공통 실습 파일

이 폴더는 프레임워크를 비교할 때 같은 데이터와 평가 규칙을 사용하기 위한 최소 실습 키트입니다.

## 구성

```text
labs/
├── data/
│   ├── train.jsonl
│   ├── eval.jsonl
│   └── demo_predictions.jsonl
├── configs/
│   ├── axolotl_gemma3_lora.yml
│   └── llamafactory_gemma3_lora.yaml
├── validate_dataset.py
├── evaluate_outputs.py
└── hf_sft.py
```

샘플의 반품·배송 정책은 학습용 가상 정책입니다. 실제 고객 대응에 사용하지 마세요.

## 1. CPU에서 바로 실행

저장소 루트:

```bash
python gemma/labs/validate_dataset.py
python gemma/labs/evaluate_outputs.py --demo
```

외부 패키지가 필요 없습니다.

## 2. 데이터 규격

```json
{"id":"train-001","messages":[{"role":"user","content":"질문"},{"role":"assistant","content":"모범 답변"}],"tags":["intent"]}
```

- UTF-8 JSONL
- ID 고유
- user/assistant 순서
- 마지막 turn은 assistant
- 실제 개인정보와 secret 금지

## 3. 자신의 데이터로 교체

1. 샘플 파일을 복사해 별도 프로젝트 경로에 둡니다.
2. `id`, `messages`, `tags` 구조를 유지합니다.
3. train/eval에 같은 질문이나 같은 원문 문서가 섞이지 않게 합니다.
4. 검증기를 실행합니다.
5. 토큰 길이는 선택한 모델 tokenizer로 별도 확인합니다.

## 4. Hugging Face 스모크 학습

```bash
pip install -U torch transformers datasets accelerate trl peft
python gemma/labs/hf_sft.py --help
```

GPU:

```bash
python gemma/labs/hf_sft.py \
  --output-dir outputs/hf-smoke \
  --max-steps 20
```

QLoRA:

```bash
pip install -U bitsandbytes
python gemma/labs/hf_sft.py \
  --output-dir outputs/hf-qlora-smoke \
  --max-steps 20 \
  --load-in-4bit
```

## 5. 설정 기반 도구

YAML의 `/ABSOLUTE/PATH/...`를 실제 절대 경로로 바꾸고 각 프레임워크 문서를 따릅니다. config는 학습 개념을 보여 주는 시작점입니다. 설치한 버전의 schema/help로 검증한 뒤 실행하세요.

## 6. 결과 디렉터리 권장 구조

```text
outputs/<run-id>/
├── adapter files
├── environment.txt
├── resolved-config.yaml
├── metrics.json
├── predictions.jsonl
└── notes.md
```

`outputs/`, 모델 가중치, 비밀키는 Git에 커밋하지 않습니다.

