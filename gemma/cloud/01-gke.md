# Google Cloud GKE에서 Gemma 튜닝

## 언제 선택하나

GKE는 컨테이너 이미지, Kubernetes Job, GPU 종류와 수량, 스토리지, 네트워크, 관측, 서빙을 직접 통제해야 할 때 적합합니다. 관리 부담이 필요 없다면 Vertex AI가 더 단순할 수 있습니다.

공식 출발점:

- [GKE에서 여러 GPU로 Gemma 파인튜닝](https://cloud.google.com/kubernetes-engine/docs/tutorials/finetune-gemma-gpu?hl=ko)
- [공식 GKE 샘플](https://github.com/GoogleCloudPlatform/kubernetes-engine-samples/tree/main/ai-ml/llm-finetuning-gemma)

> 비용 경고: 공식 예제의 학습 Job은 여러 L4 GPU를 요청합니다. manifest를 그대로 적용하기 전에 현재 가격, 할당량, GPU 개수, 예상 실행 시간을 확인하세요. 클러스터가 남아 있거나 서빙 Deployment가 계속 실행되면 학습이 끝난 뒤에도 비용이 발생할 수 있습니다.

## 0단계: 실행 전 승인표

아래가 모두 채워지기 전에는 리소스를 만들지 않습니다.

| 항목 | 값 |
|---|---|
| 프로젝트 ID |  |
| 리전/존 |  |
| GKE 모드 | Autopilot / Standard |
| GPU 종류·개수 |  |
| 현재 시간당 추정 비용 |  |
| 최대 실행 시간 |  |
| 예산 알림 |  |
| 담당자 |  |
| 삭제 예정 시각 |  |
| output/checkpoint 저장 위치 |  |

## 1단계: 사전 점검

```bash
gcloud auth list
gcloud config get-value project
gcloud services list --enabled
gcloud compute accelerator-types list --filter="name:nvidia-l4"
gcloud compute regions describe us-central1
```

필요한 IAM과 API는 공식 가이드의 현재 목록을 따릅니다. 개인 Owner를 장기 사용하지 말고 학습용 service account에 최소 권한을 부여합니다.

모델 약관과 Hugging Face token도 로컬 단계에서 먼저 검증합니다.

## 2단계: 샘플 revision 고정

```bash
git clone https://github.com/GoogleCloudPlatform/kubernetes-engine-samples
cd kubernetes-engine-samples/ai-ml/llm-finetuning-gemma
git rev-parse HEAD
```

공식 샘플은 Gemma 2B와 특정 library/container 버전을 사용할 수 있습니다. 다음을 최신 실습 목표와 비교합니다.

- 모델 ID
- Transformers/TRL/PEFT
- GPU 수
- dataset
- push destination
- vLLM image

Gemma 3로 바꿀 때 모델 ID만 바꾸지 말고 chat template, dtype, class, sequence length, vLLM 지원까지 확인합니다.

## 3단계: 컨테이너 로컬 검증

Dockerfile에서 base image tag와 Python 패키지를 확인합니다. 가능하면 digest로 고정합니다.

```bash
docker build -t gemma-finetune:smoke .
docker run --rm --gpus all gemma-finetune:smoke python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

그다음 GPU 없이 가능한 data validation 모드를 컨테이너에서 실행합니다.

```bash
docker run --rm gemma-finetune:smoke \
  python /workspace/validate_dataset.py
```

컨테이너가 곧 실험의 재현 단위입니다. `pip install -U`만 사용한 이미지보다 lock/revision이 있는 이미지를 선호합니다.

## 4단계: 클러스터 선택

### Autopilot

- 노드 관리를 줄이고 워크로드 요청에 따라 확장
- GPU workload 지원 리전/버전 확인
- 유휴 노드 운영 부담 감소

### Standard

- 노드 풀, taint, autoscaling, 드라이버를 더 세밀하게 제어
- 플랫폼 팀이 이미 GKE를 운영할 때 적합
- 유휴 GPU 노드의 비용 통제가 중요

공식 예제의 생성 명령을 실행하기 전에 release channel과 현재 지원 버전을 조회합니다. 예제의 placeholder를 그대로 붙여 넣지 마세요.

## 5단계: 비밀 관리

공식 예제는 Kubernetes Secret을 만듭니다.

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_api_token="$HF_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
```

안전 점검:

- shell history에 토큰을 직접 입력하지 않았는가?
- manifest나 Git에 base64 secret을 저장하지 않았는가?
- read-only token으로 충분한가?
- 모델을 Hub에 push해야 할 때만 write 권한을 별도 사용했는가?
- 장기 운영은 Secret Manager/Workload Identity 연계를 검토했는가?

## 6단계: Artifact Registry와 Cloud Build

```bash
gcloud artifacts repositories create gemma \
  --repository-format=docker \
  --location=us \
  --description="Gemma training images"

gcloud builds submit .
```

이미지는 `latest`가 아니라 불변 tag를 사용합니다.

```text
us-docker.pkg.dev/PROJECT_ID/gemma/finetune:git-<short-sha>
```

빌드 provenance:

- source Git SHA
- image digest
- dependency lock
- build time
- vulnerability scan 결과

## 7단계: Job manifest 감사

공식 manifest를 적용하기 전에:

```bash
kubectl apply --dry-run=server -f finetune.yaml
kubectl diff -f finetune.yaml
```

반드시 볼 항목:

- `resources.requests/limits`의 GPU 개수
- accelerator node selector
- model ID
- `max_steps`, dataset limit, sequence length
- secret reference
- `/dev/shm`
- checkpoint/output volume
- restart/backoff policy
- service account

첫 Job은 다음처럼 축소합니다.

```text
GPU: 1
model: Gemma 3 1B IT
max_steps: 5
dataset: 20 samples
sequence: 256
push: false
```

공식 8-GPU manifest를 바로 첫 실행으로 사용하지 않습니다.

## 8단계: Job 실행과 관측

```bash
kubectl apply -f finetune-smoke.yaml
kubectl get jobs,pods -w
kubectl logs job/finetune-smoke -f
kubectl describe pod POD_NAME
```

수집할 값:

- pending 시간과 이유
- image pull 시간
- 모델 download 시간
- compile/warmup 시간
- step time, tokens/s
- GPU utilization/VRAM
- checkpoint upload 시간
- 총 wall time

`Pending`이 오래 지속되면 코드보다 quota, accelerator availability, node selector, taint를 먼저 봅니다.

## 9단계: 멀티 GPU 확장

단일 GPU가 통과한 뒤 2개, 그다음 목표 개수로 확장합니다.

각 단계에서:

```text
scaling efficiency
= single-GPU step time
  / (N-GPU step time × N)
```

데이터 병렬 GPU 수가 늘면 effective batch도 늘 수 있으므로 accumulation을 조정합니다. NCCL 로그, 네트워크, shared memory, checkpoint shard도 확인합니다.

## 10단계: 서빙

공식 예제는 vLLM OpenAI-compatible server를 Deployment/Service로 배포합니다. 먼저 로컬/Job에서 병합 모델이 vLLM과 호환되는지 확인합니다.

```bash
kubectl apply -f serve-gemma.yaml
kubectl wait --for=condition=Available --timeout=700s \
  deployment/vllm-gemma-deployment
kubectl port-forward service/llm-service 8000:8000
```

검증:

- readiness/liveness probe
- cold start
- 같은 chat template
- concurrency별 latency/throughput
- OOM 시 복구
- unauthorized 외부 노출 방지

## 11단계: 반드시 정리

작업 직후:

```bash
kubectl delete job finetune-smoke
kubectl delete deployment vllm-gemma-deployment
kubectl delete service llm-service
```

클러스터가 더 필요 없으면 정확한 프로젝트·리전·이름을 다시 확인한 뒤:

```bash
gcloud container clusters delete CLUSTER_NAME \
  --location=CONTROL_PLANE_LOCATION
```

추가 확인:

```bash
gcloud container clusters list
gcloud compute instances list
gcloud artifacts docker images list LOCATION-docker.pkg.dev/PROJECT/REPOSITORY
```

Cloud Storage, Artifact Registry, static IP, load balancer도 비용 대상인지 확인합니다. 공유 프로젝트의 리소스는 소유자를 확인하지 않고 삭제하지 않습니다.

## 완료 기준

- [ ] 비용·할당량·삭제 시각을 승인했다.
- [ ] container를 로컬에서 검증하고 digest를 기록했다.
- [ ] server dry-run과 manifest 감사를 했다.
- [ ] 1-GPU 5-step Job부터 실행했다.
- [ ] checkpoint와 고정 평가를 확인했다.
- [ ] 모든 과금 리소스의 정리 여부를 확인했다.

