# OpenAI Agents를 Ollama에서 쓰는 방법 (LiteLLM 프록시)

LiteLLM이 OpenAI 호환 REST API(/v1)를 열고 내부에서 Ollama로 위임합니다. Agents SDK는 이 OpenAI 호환 엔드포인트에 붙습니다.

- 예시 설정 파일: `ua/chap04/examples/litellm_ollama_config.yaml`
- 프록시 주소(예시): `http://127.0.0.1:4000/v1`
- 모델 이름: 설정의 `model_name` (예: `llama3`)

빠른 절차
1) Ollama에서 모델 준비: `ollama pull llama3`
2) LiteLLM 설치: `uv pip install litellm[proxy]`
3) 프록시 실행: `litellm --config ua/chap04/examples/litellm_ollama_config.yaml`
4) 환경변수 설정: `OPENAI_BASE_URL=http://127.0.0.1:4000/v1`, `OPENAI_API_KEY=test-key`
5) Agents 코드에서 `model="llama3"`로 호출

문제 해결
- 401/403: OPENAI_API_KEY가 config의 `master_key`와 일치하는지 확인
- 404 모델 없음: Ollama에 모델이 pull되어 있고 `model_name`↔`litellm_params.model` 매핑이 맞는지 확인
- 스트리밍/툴콜: LiteLLM가 OpenAI 호환 기능을 제공(제한은 공식 문서 참고)

유용한 링크
- LiteLLM Proxy 문서: https://docs.litellm.ai/docs/proxy
- Ollama: https://ollama.com


