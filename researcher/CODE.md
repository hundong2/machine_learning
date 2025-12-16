# 예산 인식 Researcher 에이전트 가이드
LangGraph로 예산을 고려해 검색/브라우징을 조절하며 답변을 생성하는 에이전트를 초보자도 따라 만들 수 있게 단계별로 설명합니다.

## 예시 파일
[researcher/main.py (예시 실행 파일)](https://github.com/hundong2/machine_learning/blob/main/researcher/main.py)

## 답변
아래 가이드는 개념 → 작은 예제 → 확장 순으로 진행하며, 실행 가능한 코드 조각과 함께 LangGraph/LLM 연동의 핵심을 설명합니다.

### 1) 핵심 개념 빠르게 잡기
- 상태(State): 에이전트의 진행 상황을 담은 딕셔너리. 예산, 도구 결과, 답변 후보 등.
- 노드(Node): 상태를 입력받아 새로운 상태를 반환하는 함수. 사고/계획, 도구 호출, 검증, 최종 답변 등.
- 간선(Edge): 다음에 실행할 노드를 결정하는 연결.
- 라우터(Router): 상태를 보고 분기 키를 반환해 간선 매핑으로 다음 노드를 선택.

간단 도식: 사고(plan) → 도구(tool) → 검증(verify) → 답변(answer)

### 2) 최소 실행 예제로 구조 이해하기
아래 코드는 상태 스키마와 노드 2개만으로 “사고 → 답변” 흐름을 보여줍니다.

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    question: str
    chat_history: List[str]
    plan: str
    final_verified_answers: List[str]
    next_action_type: str

def think_and_plan_node(state: AgentState) -> AgentState:
    plan = f"질문 분석: {state['question']} → 간단 답변 시도"
    state['plan'] = plan
    state['chat_history'].append(plan)
    state['next_action_type'] = 'answer'
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    answer = f"요약 답변: {state['question']}에 대한 핵심 포인트"
    state['final_verified_answers'].append(answer)
    state['chat_history'].append(answer)
    state['next_action_type'] = 'stop'
    return state

def route_from_think_plan(state: AgentState) -> str:
    return {
        'answer': 'generate_answer_node',
        'stop': END,
    }.get(state['next_action_type'], 'generate_answer_node')

graph = StateGraph(AgentState)
graph.add_node('think_plan_node', think_and_plan_node)
graph.add_node('generate_answer_node', generate_answer_node)
graph.add_conditional_edges('think_plan_node', route_from_think_plan, {
    'generate_answer_node': 'generate_answer_node',
    END: END,
})
graph.set_entry_point('think_plan_node')
app = graph.compile()

initial = {
    'question': 'Gradient Descent란?',
    'chat_history': [],
    'plan': '',
    'final_verified_answers': [],
    'next_action_type': '',
}

result = app.invoke(initial)
print(result['final_verified_answers'][-1])
```

실행 결과는 질문에 대한 간단 요약입니다. 여기서 도구 호출/검증을 단계적으로 추가해 갑니다.

### 3) 예산을 고려한 도구 호출 추가
검색/브라우징 예산을 상태에 넣고, “도구를 쓸지 바로 답변할지” 결정합니다.

```python
from typing import Literal

class AgentState(TypedDict):
    question: str
    chat_history: List[str]
    plan: str
    final_verified_answers: List[str]
    next_action_type: Literal['tool','answer','stop']
    # 예산
    search_total: int
    search_used: int

def search_tool_mock(query: str) -> str:
    return f"[검색 결과] {query} 관련 상위 요약"

def think_and_plan_node(state: AgentState) -> AgentState:
    remaining = state['search_total'] - state['search_used']
    if remaining > 0:
        state['plan'] = '한 번 검색 후 답변 생성'
        state['next_action_type'] = 'tool'
    else:
        state['plan'] = '예산 없음: 직접 답변'
        state['next_action_type'] = 'answer'
    state['chat_history'].append(state['plan'])
    return state

def call_tool_node(state: AgentState) -> AgentState:
    result = search_tool_mock(state['question'])
    state['chat_history'].append(result)
    state['search_used'] += 1
    state['next_action_type'] = 'answer'
    return state

def route_from_think_plan(state: AgentState) -> str:
    if state['next_action_type'] == 'tool':
        return 'call_tool_node'
    if state['next_action_type'] == 'answer':
        return 'generate_answer_node'
    return END
```

이제 남은 예산에 따라 검색을 1회 수행한 뒤 답변으로 이동합니다.

### 4) 검증 노드로 무한 루프 방지하기
LangGraph의 `GraphRecursionError`는 보통 라우팅이 “계속 같은 노드로 되돌아가는” 경우 발생합니다. 검증 노드를 넣어 종료 신호를 명확히 만듭니다.

```python
def verify_node(state: AgentState) -> AgentState:
    # 매우 단순한 검증: 히스토리에 검색 결과가 있으면 성공
    success = any('[검색 결과]' in h for h in state['chat_history'])
    state['chat_history'].append(f"검증결과: {'SUCCESS' if success else 'CONTINUE'}")
    state['next_action_type'] = 'stop' if success else 'answer'
    return state

# 간선 설계 예시
# call_tool_node → verify_node → (SUCCESS: stop / CONTINUE: answer)
```

검증을 통해 “정지(stop)”로 라우팅되면 그래프가 종료되어 재귀 한도를 넘지 않습니다.

### 5) LLM 연동과 <answer> 파싱 예제
Gemini와 연결할 때는 `.env`의 `GOOGLE_API_KEY`를 로드하고, 모델이 `<answer>...</answer>`를 포함하도록 프롬프트를 구성합니다.

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.0-pro", api_key=os.getenv('GOOGLE_API_KEY'))

def parse_answer(text: str) -> str:
    start, end = text.find('<answer>'), text.find('</answer>')
    if start != -1 and end != -1 and end > start:
        return text[start+9:end].strip()
    return 'None'

def generate_answer_node(state: AgentState) -> AgentState:
    prompt = f"질문: {state['question']}\n<answer>핵심만 요약해줘</answer>"
    r = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(r, 'content', '') or str(r)
    answer = parse_answer(content)
    state['final_verified_answers'].append(answer)
    state['chat_history'].append(f"LLM응답: {answer}")
    state['next_action_type'] = 'stop'
    return state
```

LLM이 태그를 누락해도 안전하게 기본값 `'None'`으로 처리합니다.

### 6) 전체 실행 예시와 명령어
아래 명령으로 실행하고 결과 키를 확인하세요.

```bash
python researcher/main.py
```

확인할 필드: `final_verified_answers`, `chat_history`, `search_used` 등.

### 7) 라이브러리와 올바른 임포트
- `typing`: `TypedDict`, `Literal`, `List` 등으로 상태 스키마를 안정적으로 정의.
- `langgraph`: `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point`, `compile`.
- `langchain_core.messages`: `HumanMessage`로 프롬프트를 전달.
- `pydantic`: 도구 인자 스키마가 필요하다면 `from pydantic import BaseModel, Field`를 사용하세요. (이전 `langchain_core.pydantic_v1` 대신 권장)
- `python-dotenv`: `.env`의 `GOOGLE_API_KEY`를 로드.
- `langchain_google_genai`: `ChatGoogleGenerativeAI`로 Gemini 호출.

### 8) 자주 하는 실수와 해결법
- 매핑 키 불일치: 라우터가 반환하는 문자열과 `add_conditional_edges` 매핑 키가 다르면 잘못된 노드로 흐르거나 루프가 생깁니다.
- 종료 신호 누락: 어떤 분기에서도 `stop`에 도달하지 않으면 재귀 한도 초과 에러.
- 검증 후 라우팅: 검증이 항상 같은 노드로 되돌리면 루프가 생깁니다. 성공 시 `stop`, 실패 시 다른 노드로 이동.
- 예산 갱신 누락: `used`를 늘리지 않으면 라우팅 로직이 계속 도구 호출을 선택할 수 있습니다.

### 9) LangGraph 재귀 한도(Recursion) 오류 대처 체크리스트
- `config={"recursion_limit": 50}`를 늘릴 수 있지만, 근본 해결은 라우팅입니다.
- 각 분기에 대해 다음을 점검하세요:
  - 최소 한 경로는 `END`로 이어지나?
  - 동일 노드로 무한 회귀하는 경로가 없나?
  - 검증 노드에서 성공 시 `stop`으로 확실히 이동하나?
  - 실패 시 다른 노드로 이동하며 상태가 변해 다음 분기 결과가 바뀌나?

## 추가 자료
- LangGraph Errors: https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT
- LangGraph Routing: https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges
- Google AI Python SDK: https://python.langchain.com/docs/integrations/chat/google_generative_ai
- Pydantic 사용 가이드: https://docs.pydantic.dev/latest/

# 예산 인식 Researcher 에이전트 가이드
LangGraph로 예산을 고려해 검색/브라우징을 조절하며 답변을 생성하는 에이전트를 초보자도 따라 만들 수 있게 단계별로 설명합니다.
## 예시 파일
[researcher/main.py (예시 실행 파일)](https://github.com/hundong2/machine_learning/blob/main/researcher/main.py)
## 답변
아래 가이드는 개념 → 작은 예제 → 확장 순으로 진행하며, 실행 가능한 코드 조각과 함께 LangGraph/LLM 연동의 핵심을 설명합니다.
### 추가 자료
- [LangGraph Errors](https://docs.langchain.com/oss/python/langgraph/errors/GRAPH_RECURSION_LIMIT)
- [LangGraph Routing](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [Google AI Python SDK](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
- [Pydantic 사용 가이드](https://docs.pydantic.dev/latest/)