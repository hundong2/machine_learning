# Researcher 에이전트 코드 설명

이 문서는 `researcher/main.py`의 예산 인식 에이전트(BATS 아이디어 반영) 구현을 간단히 설명합니다.

## 구성 개요
- `AgentState`: 에이전트 상태를 정의하는 `TypedDict`.
  - 질문, 대화 기록, 계획, 예산 상태(`search_total/used`, `browse_total/used`), 궤적, 답변 후보, 도구 결과, 통합 비용, 시도 횟수 등.
- 툴 더미:
  - `search_tool_mock(query)`: 검색 결과 문자열 반환.
  - `browse_tool_mock(url, goal)`: 브라우징 결과 문자열 반환.
- LLM 예시:
  - `ChatGoogleGenerativeAI(model="gemini-pro")`로 예시 객체만 초기화(실제 호출은 시뮬레이션).

## 그래프 노드
- `think_and_plan_node(state)`
  - 예산 상태(남은 검색/탐색 횟수)에 기반해 전략(HIGH/MEDIUM/LOW/CRITICAL) 텍스트 생성.
  - 다음 행동(`next_action_type`)을 결정하고 필요 시 `proposed_tool_call` 설정.
  - 궤적과 대화 기록에 사고 내용을 누적.
- `call_tool_node(state)`
  - 제안된 도구 호출을 실행하고 결과를 `tool_results`에 추가.
  - 예산 사용량(`search_used/browse_used`)과 `unified_cost` 업데이트.
- `verify_node(state)`
  - 현재 답변/궤적/예산을 평가하여 `verification_decision`을 설정(SUCCESS/CONTINUE/PIVOT).
  - 시도 횟수 증가 및 요약 기반 컨텍스트 축약.
- `generate_answer_node(state)`
  - 검증 성공 시 기존 답변을 확정, 아니면 규칙에 따라 `None` 또는 가상 답변 생성.
  - 최종 답변을 `final_verified_answers` 및 `chat_history`에 누적.

## 라우팅 함수
- `route_from_think_plan(state)`
  - tool → `call_tool_node`, answer → `generate_answer_node`, stop → `END`.
- `route_after_verification(state)`
  - SUCCESS → `generate_answer_node`, CONTINUE/PIVOT → `think_plan_node` (단, 최대 시도 초과 시 `generate_answer_node`).

## 그래프 빌드
- `StateGraph(AgentState)` 생성 후 다음을 등록:
  - 노드: `think_plan_node`, `call_tool_node`, `verify_node`, `generate_answer_node`.
  - 간선: `add_edge`, `add_conditional_edges`로 라우팅 구성.
  - 시작점: `set_entry_point("think_plan_node")`.
  - `compile()`로 앱 생성.

## 실행 힌트
실행 하네스 예시(주석 참고)를 바탕으로 `app.invoke(initial_state)`로 단위 테스트 가능합니다. 실제 LLM을 연결하려면 `.env`의 `GOOGLE_API_KEY`를 로드하고, `llm.invoke` 응답 파싱(<answer> 태그) 로직을 추가하세요.

## 설계 포인트
- 예산 기반 전략 분류는 남은/총 비율로 단순·명확화했습니다.
- 모든 노드가 상태를 **불변식으로 업데이트**하며, 히스토리/궤적을 누적해 컨텍스트 손실을 줄입니다.
- 검증 노드는 결정에 따라 시도 횟수와 컨텍스트 요약을 관리해 비용을 억제합니다.

## 사용 라이브러리 상세 안내 (지침서)

- `typing`
  - `TypedDict`: 딕셔너리 형태의 구조적 타입 정의. 키/값 타입을 명시해 상태 스키마를 안정적으로 관리.
  - `List`, `Dict`, `Literal`: 리스트/딕셔너리/리터럴(허용 값 집합) 타입 힌트.

- `langgraph`
  - `StateGraph`: 상태 기반 워크플로우를 구성하는 그래프 빌더.
    - `add_node(name, fn)`: 노드(함수)를 그래프에 등록.
    - `add_edge(src, dst)`: 단일 경로 간선 추가.
    - `add_conditional_edges(src, router_fn, mapping)`: 분기 라우팅. `router_fn(state)` 반환값을 `mapping` 키로 매핑해 다음 노드 결정.
    - `set_entry_point(name)`: 시작 노드 지정.
    - `compile()`: 실행 가능한 앱으로 컴파일. `app.invoke(state)` 호출.

- `langchain_core.messages`
  - `BaseMessage`, `HumanMessage`: LLM과의 대화 메시지 모델. `HumanMessage(content=...)`로 사용자/내부 메시지 기록.

- `langchain_core.pydantic_v1`
  - `BaseModel`, `Field`: 인자 스키마 검증용(툴 인자 등). `Field(description=...)`로 문서화.

- `langchain_google_genai`
  - `ChatGoogleGenerativeAI`: Google Gemini 채팅 모델 래퍼.
    - 주요 인자: `model`(예: `gemini-pro`, `gemini-3.0-pro`), `api_key`, `temperature`.
    - `invoke(messages)`: LangChain 메시지 배열을 입력해 응답 객체 반환. `response.content`에 텍스트.

- `python-dotenv`
  - `load_dotenv()`: `.env` 파일의 환경 변수를 현재 프로세스에 로드.
    - 본 코드에서 `GOOGLE_API_KEY`를 로드해 LLM에 전달.

### 자주 하는 실수와 팁
- 그래프 API 메서드명 혼동
  - `addnode`가 아니라 `add_node`, `setentrypoint`가 아니라 `set_entry_point`.
- 라우팅 매핑 키 일치
  - `route_from_think_plan`이 반환하는 문자열과 `add_conditional_edges`의 매핑 키가 정확히 동일해야 함.
- LLM 응답 파싱
  - 태그 기반(`<answer>`) 파싱은 간단하지만, 모델이 태그를 생략할 수 있어 예외 처리와 기본값(`None`)을 준비.
- 예산 계산 일관성
  - `remaining = total - used` 공식을 함수/노드 전반에서 동일하게 사용하면 상태 누락에도 안전.

### 빠른 시작 명령어
```bash
python researcher/main.py
```
실행 후 출력되는 딕셔너리에서 `final_verified_answers`, `verification_decision`, `tool_results` 등을 확인하세요.
