# 03. 모델 형식·병합·배포·운영

튜닝이 끝났다는 것은 학습 loop가 종료됐다는 뜻일 뿐입니다. 실제 완료 조건은 새 프로세스와 목표 런타임에서 정확한 모델·tokenizer·template을 로드해 회귀 평가를 통과하는 것입니다.

## 1. 산출물 종류

### LoRA adapter

- 작은 파일
- base model과 함께 로드
- 여러 업무별 adapter 교체 가능
- base revision 불일치에 취약

### 병합 Safetensors

- base + adapter가 하나의 일반 모델
- Transformers/vLLM 등에서 단순 로드
- 저장 공간과 배포 전송량 증가
- 병합 후 되돌리기보다 원본 adapter를 함께 보존

### Keras preset/checkpoint

- KerasHub/Keras 생태계에서 자연스러움
- 다른 런타임으로 갈 때 변환 지원 확인

### JAX/Orbax 계열 checkpoint

- optimizer/data state까지 연구 재개에 유용
- 바로 서빙하지 못하고 변환이 필요할 수 있음

### GGUF

- llama.cpp/Ollama 계열 로컬 추론에 적합
- 다양한 quantization
- training checkpoint가 아니라 배포 형식

## 2. artifact manifest

각 run에 다음 파일을 만듭니다.

```yaml
run_id: gemma-support-20260727-001
base_model: google/gemma-3-1b-it
base_revision: "<commit hash>"
framework: transformers-trl-peft
framework_versions: "environment.txt"
dataset_hash: "<sha256>"
chat_template_hash: "<sha256>"
tuning: lora
lora_r: 8
lora_alpha: 16
target_modules: [q_proj, k_proj, v_proj, o_proj]
output_format: peft_adapter
evaluation_report: "eval-report.json"
license_reviewed: true
```

## 3. 병합 검증

1. 비양자화 base를 정확한 revision으로 로드
2. adapter 로드
3. `merge_and_unload`
4. Safetensors 저장
5. 새 프로세스에서 로드
6. 고정 프롬프트 logits/greedy 출력 비교

허용오차는 dtype과 runtime을 고려해 정합니다. 문자열 하나가 같았다고 병합 전체가 정확하다고 결론 내리지 않습니다.

## 4. 채팅 템플릿 계약

배포 bundle에는 다음을 포함합니다.

- tokenizer files
- `tokenizer_config.json`
- chat template
- special token IDs
- generation config
- 최대 context/output 제한

학습과 배포 템플릿의 hash를 비교합니다. 가장 흔한 배포 품질 저하는 모델 weight보다 template/BOS/EOS 불일치입니다.

## 5. 정량 성능

최소 측정:

- time to first token
- inter-token latency
- tokens/s
- p50/p95/p99 latency
- 동시 요청별 처리량
- peak VRAM/RAM
- cold start
- prompt/output token별 비용

같은 모델이라도 quantization, batch scheduler, context length에 따라 달라집니다.

## 6. 품질 회귀 게이트

배포 전 CI:

```text
artifact integrity
→ tokenizer/template contract
→ 5개 smoke prompts
→ 고정 eval 전체
→ safety/challenge
→ latency budget
→ 승인
```

실패하면 자동으로 이전 artifact를 유지합니다.

## 7. 안전과 운영

- 모델이 정책을 “학습했다”고 application authorization을 제거하지 않기
- 개인정보와 비밀을 prompt/log에서 최소화
- 출력 후처리와 도구 권한 검증
- 거절/안전 동작을 별도 회귀
- 입력·출력 길이와 rate limit
- model/dataset/license 문서화
- drift와 사용자 피드백 모니터링

## 8. canary와 rollback

1. shadow traffic로 출력만 비교
2. 내부 사용자 canary
3. 낮은 비율의 실제 traffic
4. 품질·latency·오류율 감시
5. 자동/수동 rollback 기준

rollback 단위는 “코드”만이 아니라 model artifact + tokenizer + generation config + prompt template 전체입니다.

## 9. 전문가 과제

- adapter hot-swap과 merged model의 latency 비교
- BF16, INT8, 4-bit GGUF 품질/속도 Pareto frontier
- vLLM과 llama.cpp의 chat template 일치 테스트
- 모델 artifact SBOM과 checksum
- 재학습 trigger와 data lineage
- 비용/품질 SLO dashboard

## 완료 기준

- [ ] artifact manifest와 checksum을 만들었다.
- [ ] adapter와 merge 결과를 새 프로세스에서 검증했다.
- [ ] tokenizer/template/generation config를 bundle로 관리한다.
- [ ] 품질·성능 회귀 게이트가 있다.
- [ ] canary와 rollback 절차가 문서화되어 있다.

