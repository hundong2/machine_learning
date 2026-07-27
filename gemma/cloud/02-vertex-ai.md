# Google Cloud Vertex AI 관리형 튜닝

## 언제 선택하나

클러스터와 분산 런타임을 직접 관리하지 않고 데이터, 모델, 튜닝 모드, 평가에 집중하고 싶을 때 선택합니다. 정확한 지원 모델, 튜닝 모드, 리전은 자주 바뀌므로 작업 생성 직전에 공식 지원표를 확인합니다.

공식 출발점:

- [개방형 모델 지도/증류 미세 조정](https://cloud.google.com/vertex-ai/generative-ai/docs/models/open-model-tuning)
- [Vertex AI Model Garden](https://cloud.google.com/model-garden)

기준일 공식 문서는 Gemma 3과 Gemma 4 여러 모델을 지원하며 모델에 따라 FULL 또는 PEFT_ADAPTER 가능 여부가 다릅니다. 같은 Gemma 계열이라고 모두 같은 모드를 지원하지 않습니다.

## 0단계: 실행 전 계획

| 항목 | 값 |
|---|---|
| project |  |
| global/region |  |
| base model resource name |  |
| tuning mode | FULL / PEFT_ADAPTER |
| train/validation GCS URI |  |
| output GCS URI |  |
| epochs |  |
| 예상 비용 상한 |  |
| 배포 endpoint 필요 여부 |  |
| 삭제 예정 시각 |  |

처음에는 Console에서 현재 지원 옵션을 확인한 뒤, 재현을 위해 SDK 코드로 옮깁니다.

## 1단계: 지원표 확인

작업 생성 당일 공식 문서에서:

1. 정확한 model resource name
2. FULL/PEFT_ADAPTER 지원
3. 최대 sequence length
4. text/multimodal 형식
5. 지원 region/global
6. quota와 가격

을 기록합니다.

예시 resource name은 시점에 따라 다음처럼 보일 수 있습니다.

```text
google/gemma3@gemma-3-1b-it
google/gemma4@gemma-4-e2b-it
```

Hugging Face ID(`google/gemma-3-1b-it`)와 Vertex source model name을 혼동하지 마세요.

## 2단계: 인증과 SDK

```bash
gcloud auth application-default login
gcloud config set project PROJECT_ID
pip install -U google-cloud-aiplatform
```

```python
import vertexai
from vertexai.tuning import SourceModel, sft

PROJECT_ID = "your-project"
REGION = "global"
vertexai.init(project=PROJECT_ID, location=REGION)
```

로컬 사용자 인증보다 CI/운영은 전용 service account와 최소 IAM을 사용합니다.

## 3단계: 데이터 변환

Vertex 공식 문서는 JSONL의 prompt-completion 또는 turn-based chat 형식을 지원합니다.

Prompt completion:

```json
{"prompt":"반품 기간은 언제까지인가요?","completion":"상품 수령 후 7일 이내에 신청해 주세요."}
```

Chat:

```json
{"messages":[{"role":"user","content":"배송 조회는 어디서 하나요?"},{"role":"assistant","content":"주문 내역에서 배송 조회를 선택하세요."}]}
```

`labs/data/*.jsonl`은 chat 형식이므로 `id`, `tags` 같은 추가 필드를 제거한 제출용 파일을 별도로 만듭니다.

검증:

```bash
python gemma/labs/validate_dataset.py
wc -l vertex-train.jsonl vertex-validation.jsonl
```

Windows PowerShell에서는:

```powershell
(Get-Content vertex-train.jsonl).Count
```

## 4단계: GCS 업로드

```bash
gcloud storage cp vertex-train.jsonl \
  gs://YOUR_BUCKET/gemma/data/train.jsonl
gcloud storage cp vertex-validation.jsonl \
  gs://YOUR_BUCKET/gemma/data/validation.jsonl
```

권장 경로:

```text
gs://bucket/project/data/<dataset-hash>/train.jsonl
gs://bucket/project/runs/<run-id>/
```

파일을 같은 이름으로 덮어쓰지 말고 hash/version을 경로에 포함합니다.

## 5단계: 관리형 SFT Job

현재 공식 SDK 개념:

```python
sft_tuning_job = sft.train(
    source_model=SourceModel(
        base_model="google/gemma3@gemma-3-1b-it",
    ),
    tuning_mode="FULL",  # 모델별로 FULL 또는 PEFT_ADAPTER
    epochs=1,
    train_dataset="gs://YOUR_BUCKET/gemma/data/train.jsonl",
    validation_dataset="gs://YOUR_BUCKET/gemma/data/validation.jsonl",
    output_uri="gs://YOUR_BUCKET/gemma/runs/run-001",
)
```

Gemma 3 1B의 현재 지원 모드가 FULL이고 Gemma 4의 현재 지원 모드가 PEFT일 수 있습니다. 예시 주석을 믿지 말고 지원표에 맞춥니다.

첫 작업은 작은 데이터와 1 epoch로 end-to-end artifact 생성을 확인합니다.

## 6단계: 작업 관측

기록:

- tuning job resource name
- 시작/종료 시각과 region
- input dataset URI/hash
- source model
- tuning mode/epochs
- validation metric
- output artifact URI
- 실제 비용

작업 실패 시 error만 보고 재제출하지 말고 IAM, GCS region/permission, JSONL schema, model mode를 확인합니다.

## 7단계: 산출물

공식 문서는 output URI 아래에 최종과 중간 Safetensors checkpoint를 저장합니다.

```text
.../postprocess/node-0/checkpoints/final/
.../postprocess/node-0/checkpoints/checkpoint-N/
```

확인:

```bash
gcloud storage ls --recursive gs://YOUR_BUCKET/gemma/runs/run-001
```

artifact manifest에 파일명, 크기, checksum, base model, tokenizer source를 남깁니다.

## 8단계: 배포 전 오프라인 평가

endpoint를 만들면 계속 과금될 수 있으므로 먼저 artifact를 다운로드하거나 batch 경로로 고정 평가를 실행합니다.

```bash
gcloud storage cp --recursive \
  gs://YOUR_BUCKET/gemma/runs/run-001/postprocess/node-0/checkpoints/final \
  outputs/vertex-run-001
```

검증:

- tokenizer/chat template
- Safetensors shard와 index
- greedy baseline
- 성공/실패/경계/안전 세트
- base 모델 대비 회귀

## 9단계: endpoint 배포

공식 SDK는 Model Garden CustomModel을 GPU endpoint에 배포할 수 있습니다.

```python
from vertexai.preview import model_garden

model = model_garden.CustomModel(
    gcs_uri=(
        "gs://YOUR_BUCKET/gemma/runs/run-001/"
        "postprocess/node-0/checkpoints/final"
    ),
)
endpoint = model.deploy(
    machine_type="g2-standard-12",
    accelerator_type="NVIDIA_L4",
    accelerator_count=1,
)
```

이 호출부터 지속 비용이 발생할 수 있습니다.

배포 검사:

- private/public 접근
- 최소 replica와 autoscaling
- timeout, max tokens
- latency p50/p95
- concurrency
- safety/application guardrail
- audit log

## 10단계: 정리

사용 후 endpoint에서 모델을 undeploy하고 endpoint를 삭제합니다. Console과 SDK에서 실제 상태가 종료됐는지 확인합니다. GCS의 checkpoint는 보존 정책에 맞춰 관리하며 공유 bucket을 통째로 삭제하지 않습니다.

최종 비용 점검:

- endpoint replica
- tuning job
- GCS 저장/egress
- TensorBoard/로그
- container/image artifact

## GKE와 비교

| 항목 | Vertex AI | GKE |
|---|---|---|
| 시작 속도 | 빠름 | 플랫폼 준비 필요 |
| 인프라 제어 | 제한적 | 높음 |
| 지원 모델/모드 | 관리형 목록 | 컨테이너가 지원하면 가능 |
| 운영 부담 | 낮음 | 높음 |
| Kubernetes 역량 | 불필요 | 필요 |
| 커스텀 커널 | 제한 | 유연 |

## 완료 기준

- [ ] 현재 지원 model name과 tuning mode를 기록했다.
- [ ] 제출용 JSONL을 검증하고 versioned GCS에 업로드했다.
- [ ] 작은 1-epoch Job으로 artifact를 확인했다.
- [ ] endpoint 전에 오프라인 평가를 수행했다.
- [ ] endpoint와 잔여 리소스를 정리했다.

