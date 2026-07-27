# 07. Axolotl

## 언제 선택하나

Axolotl은 데이터 전처리, SFT, LoRA/QLoRA, 평가, 멀티 GPU를 YAML 하나로 관리하는 프레임워크입니다. 단일 GPU 실험을 팀 표준 설정과 클라우드로 확장할 때 강점이 있습니다.

공식 출발점:

- [Axolotl 문서](https://docs.axolotl.ai/)
- [Quickstart](https://docs.axolotl.ai/docs/getting-started.html)
- [SFT agent docs](https://docs.axolotl.ai/docs/agents/sft.html)
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)

Google Gemma 페이지의 링크는 Gemma 2 보관 notebook입니다. 설치와 설정은 현재 Axolotl 문서를 사용합니다.

## 1단계: 환경

Axolotl은 Linux, 최신 Python/PyTorch, NVIDIA 또는 지원 AMD GPU를 전제로 합니다. 현재 공식 요구사항을 확인한 뒤 uv 환경 또는 Docker를 선택합니다.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --no-build-isolation axolotl
axolotl --help
```

Docker는 의존성 충돌이 적지만 다음을 명시해야 합니다.

- GPU runtime
- `--ipc=host` 또는 shared memory
- 데이터/출력 volume
- HF token 전달 방식
- image tag 또는 digest

## 2단계: 예제와 schema 가져오기

```bash
axolotl fetch examples
axolotl config-schema --field adapter
axolotl agent-docs sft
```

인터넷 블로그보다 설치된 버전과 함께 제공되는 schema가 더 정확합니다.

## 3단계: YAML 준비

이 저장소의 [`labs/configs/axolotl_gemma3_lora.yml`](../labs/configs/axolotl_gemma3_lora.yml)을 시작점으로 사용합니다.

핵심 개념:

```yaml
base_model: google/gemma-3-1b-it
adapter: lora
load_in_4bit: false
sequence_len: 256
micro_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0002
num_epochs: 1
```

데이터 형식 키는 현재 schema에서 확인합니다. 공통 `messages` JSONL을 chat template 데이터로 지정하고, Gemma tokenizer의 template을 사용하도록 설정합니다.

## 4단계: preprocess

```bash
axolotl preprocess path/to/axolotl_gemma3_lora.yml
```

확인:

- 샘플 수
- 길이 percentiles
- 잘린 샘플 수
- rendered turns
- label mask
- cache 경로와 데이터 revision

캐시는 설정이나 데이터가 바뀌면 무효화해야 합니다. 다른 run의 오래된 prepared dataset을 재사용하지 않도록 경로에 데이터 hash를 포함합니다.

## 5단계: 스모크 학습

YAML에 작은 `max_steps`를 지정한 뒤:

```bash
axolotl train path/to/axolotl_gemma3_lora.yml
```

성공 기준:

- finite loss
- 예상 장치 수
- adapter checkpoint
- resolved config
- 재로드 가능한 tokenizer

## 6단계: QLoRA

```yaml
adapter: qlora
load_in_4bit: true
```

실제 키 조합은 현재 config schema를 따릅니다. LoRA와 QLoRA run의 output directory를 분리하고, base 모델이 어떤 양자화 방식으로 로드됐는지 로그를 저장합니다.

## 7단계: sample packing

짧은 샘플은 padding 낭비가 큽니다. Axolotl의 packing/multipacking을 켜기 전에:

1. 평균 토큰 길이와 padding 비율 측정
2. packing off 기준 실험
3. packing on에서 샘플 경계와 loss mask 확인
4. tokens/s와 eval 비교

packing은 속도 기능이지 품질 향상을 보장하는 기능이 아닙니다.

## 8단계: 멀티 GPU

단일 GPU YAML이 검증된 뒤 공식 multi-GPU 방식으로 실행합니다.

- DDP: 모델이 각 GPU에 들어가고 데이터 병렬
- FSDP: 파라미터·그래디언트·optimizer state 샤딩
- DeepSpeed ZeRO: stage에 따라 상태와 파라미터 샤딩

비교할 항목:

| 항목 | DDP | FSDP | ZeRO |
|---|---|---|---|
| GPU당 VRAM |  |  |  |
| 처리량 |  |  |  |
| checkpoint 크기/형식 |  |  |  |
| 재개 시간 |  |  |  |
| 병합 난이도 |  |  |  |

## 9단계: 설정 검증을 CI로

팀 프로젝트에서는 다음을 자동화합니다.

1. YAML schema validation
2. 경로와 secret placeholder 검사
3. 데이터 샘플 10개 preprocess
4. tiny/dummy 모델로 1-step
5. output artifact 검사

GPU가 없는 CI에서는 schema와 데이터만, 주기적 GPU CI에서는 1-step을 실행합니다.

## 디버깅

- config key 오류: 설치 버전 `config-schema`
- dataset format 오류: rendered sample과 role mapping
- Flash Attention 오류: GPU 아키텍처·Torch·CUDA wheel
- OOM: sequence, micro batch, QLoRA, checkpointing
- 멀티 GPU hang: NCCL, shared memory, network interface, timeout
- 재개 불가: checkpoint 종류와 optimizer state 저장 여부
- 출력 이상: Gemma chat template과 tokenizer artifact

## 전문가 확장

- 동일 YAML을 Docker와 bare-metal에서 비교
- FSDP/DeepSpeed checkpoint consolidation 검증
- 데이터 cache lineage와 hash 자동 기록
- W&B/MLflow에 resolved config와 git SHA 저장
- Kubernetes Job/Spot 중단 후 checkpoint 재개
- Axolotl config를 HF `SFTConfig`로 변환하는 내부 도구 작성

## 완료 기준

- [ ] 설치 버전 schema로 YAML을 검증했다.
- [ ] preprocess 출력과 loss mask를 확인했다.
- [ ] LoRA 또는 QLoRA 스모크 학습을 완료했다.
- [ ] packing on/off를 공정 비교했다.
- [ ] checkpoint 재개 또는 멀티 GPU 중 하나를 검증했다.

