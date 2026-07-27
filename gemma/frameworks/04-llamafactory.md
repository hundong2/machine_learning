# 04. LLaMA-Factory

## 언제 선택하나

LLaMA-Factory는 많은 모델과 SFT/DPO/PPO 계열 학습을 YAML 또는 WebUI로 실행합니다. 코드를 적게 쓰고 실험 설정을 명시적으로 보관하려는 경우 좋습니다. Gemma, Gemma 2, Gemma 3을 지원하지만 모델별 template 값을 정확히 맞춰야 합니다.

공식 출발점:

- [LLaMA-Factory 문서](https://llamafactory.readthedocs.io/en/latest/)
- [SFT](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)
- [데이터 준비](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html)
- [LoRA 병합](https://llamafactory.readthedocs.io/en/latest/getting_started/merge_lora.html)

Google 튜닝 페이지의 링크는 Gemma 1용 보관 notebook입니다. 이 장은 최신 CLI 흐름을 사용합니다.

## 1단계: 환경

최신 release의 Python 요구사항을 먼저 확인합니다. 공식 프로젝트는 최근 uv 기반 설치를 권장합니다.

```bash
git clone https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
llamafactory-cli version
```

소스 revision을 기록합니다.

```bash
git rev-parse HEAD
```

## 2단계: 공통 데이터 변환

LLaMA-Factory는 Alpaca 또는 ShareGPT 계열 형식을 지원합니다. 공통 `messages`를 ShareGPT 형식으로 변환하거나, 현재 문서의 messages 매핑을 `dataset_info.json`에 선언합니다.

예시 개념:

```json
{
  "gemma_customer_support": {
    "file_name": "/absolute/path/to/train.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages"},
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

설치 revision에 포함된 `data/README.md`와 `data/dataset_info.json` 예시를 기준으로 키 이름을 확인하세요.

## 3단계: YAML 이해

이 저장소의 [`labs/configs/llamafactory_gemma3_lora.yaml`](../labs/configs/llamafactory_gemma3_lora.yaml)을 복사해 다음 항목을 실제 경로에 맞춥니다.

```yaml
model_name_or_path: google/gemma-3-1b-it
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
dataset: gemma_customer_support
template: gemma3
cutoff_len: 256
max_samples: 100
output_dir: saves/gemma3-1b/lora/smoke
```

핵심:

- `stage`: `sft`
- `finetuning_type`: `lora`, `full`, `freeze`
- `template`: 모델 대화 형식. Gemma 3은 현재 지원표의 값을 사용
- `cutoff_len`: 토큰 길이
- `gradient_accumulation_steps`: effective batch 조절
- `quantization_bit: 4`: QLoRA 경로

## 4단계: 데이터 전처리만 검증

전체 학습 전에 CLI help와 preprocess 예제를 사용해 변환 결과를 캐시합니다.

```bash
llamafactory-cli help
```

전처리 결과에서 임의 샘플을 decode해 user/model turn과 EOS를 확인합니다. WebUI 미리보기만 믿지 말고 tokenized label mask까지 확인합니다.

## 5단계: 스모크 학습

```bash
CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli train path/to/llamafactory_gemma3_lora.yaml
```

YAML에 `max_steps: 20`을 추가해 먼저 확인합니다.

성공 기준:

- loss가 유한값
- output dir에 adapter와 trainer state 생성
- loss plot 또는 로그 생성
- 예상 GPU 한 장만 사용

## 6단계: CLI override로 한 변수만 변경

```bash
llamafactory-cli train path/to/config.yaml \
  learning_rate=1e-4 \
  logging_steps=1 \
  output_dir=saves/gemma3-1b/lora/lr-1e-4
```

원본 YAML을 보존하고 run별 output directory를 분리합니다. 실험 관리에서는 최종 resolved config를 반드시 저장하세요.

## 7단계: 채팅과 평가

추론용 YAML에서 base model과 adapter 경로, template을 동일하게 지정합니다.

```bash
llamafactory-cli chat path/to/inference_config.yaml
```

자동 평가:

```bash
llamafactory-cli eval path/to/eval_config.yaml
```

프레임워크 자체 평가와 이 자료의 고정 업무 평가를 둘 다 실행합니다.

## 8단계: LoRA 병합

```bash
llamafactory-cli export path/to/merge_config.yaml
```

병합 설정의 핵심:

```yaml
model_name_or_path: google/gemma-3-1b-it
adapter_name_or_path: saves/gemma3-1b/lora/smoke
template: gemma3
export_dir: saves/gemma3-1b/merged
export_device: cpu
export_legacy_format: false
```

양자화된 base를 사용해 LoRA를 병합하지 않습니다. 비양자화 원본을 로드해 병합한 뒤 필요하면 별도 양자화합니다.

## 9단계: WebUI

WebUI는 설정을 탐색할 때 유용하지만, 전문가 수준의 재현성을 위해 다음을 지킵니다.

1. 실행 전 생성된 설정을 내보냅니다.
2. 모델·데이터·출력 경로를 기록합니다.
3. UI 기본값이 버전 업데이트로 바뀔 수 있음을 가정합니다.
4. 성공한 run을 CLI YAML로 다시 실행합니다.

## 10단계: 멀티 GPU

기본적으로 보이는 모든 장치를 사용할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train config.yaml
```

멀티 노드나 강제 torchrun:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train config.yaml
```

DeepSpeed/FSDP는 공식 분산 문서의 현재 설정을 사용합니다. GPU 수가 늘면 effective batch가 변하는지 확인합니다.

## 디버깅

- dataset 없음: `dataset_info.json` 위치와 dataset 이름
- 역할 오류: ShareGPT tag 매핑
- 이상한 출력: `template` 불일치
- OOM: cutoff, batch, quantization, checkpointing
- 병합 실패: base/adapter mismatch 또는 quantized base
- 재개 실패: `save_only_model`로 optimizer state를 저장하지 않았는지
- 버전 업데이트 후 파라미터 오류: release와 config schema revision 확인

## 완료 기준

- [ ] 데이터 등록과 tokenized preview를 확인했다.
- [ ] YAML로 20-step LoRA를 실행했다.
- [ ] CLI override 결과를 별도 run으로 저장했다.
- [ ] adapter chat과 병합 모델 chat을 비교했다.
- [ ] 실행 revision과 resolved config를 보관했다.

