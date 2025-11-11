# MessageState의 구조는 무엇인가?
LangGraph에서 대화 이력을 담는 전용 상태 타입으로, "messages" 리스트 하나를 갖는 TypedDict입니다. 리스트에는 LangChain 메시지(HumanMessage, AIMessage, ToolMessage 등)가 시간순으로 쌓입니다.

## 예시 파일
[vscode에서 열기: ua/chap06/tool_calling.py](vscode://file/Users/donghun2/workspace/machine_learning/ua/chap06/tool_calling.py)

## 답변
- 구조
  - MessageState는 사실상 {"messages": list[BaseMessage]} 형태의 상태 컨테이너입니다.
  - messages에는 HumanMessage, AIMessage, SystemMessage, ToolMessage 등 LangChain 메시지 객체가 순서대로 저장됩니다.
  - 상태 병합 규칙은 “append(add_messages)”입니다. 즉, 노드가 {"messages": [새메시지]}를 반환하면 기존 리스트 끝에 이어 붙습니다.
- 사용 패턴
  - 마지막 메시지 접근: last = state["messages"][-1]
  - 도구 호출 판단: isinstance(last, ToolMessage) 또는 last.tool_calls 유무 검사(AIMessage가 도구 호출을 제안할 때 tool_calls가 채워짐)
  - 상태 업데이트: 노드 반환값으로 {"messages": [response]}를 돌려주면 LangGraph가 자동 병합

간단 예시
````python
from langgraph.graph import MessageState
from langchain_core.messages import HumanMessage, AIMessage

# 초기 상태
state: MessageState = {"messages": [HumanMessage(content="서울 날씨 알려줘")]}

# 모델 응답을 상태에 추가(노드 반환 예)
update = {"messages": [AIMessage(content="현재 맑고 23도입니다.")]}
# LangGraph가 병합 시: state["messages"] += update["messages"]

# 마지막 메시지 확인
last = state["messages"][-1]  # AIMessage(...)
````

핵심 요약
- 키는 항상 "messages" 하나이며, 값은 메시지 객체 리스트.
- 병합은 “추가(append)”이므로 노드가 반환한 메시지가 대화 이력에 누적됩니다.

### 추가 자료
- LangGraph 상태 병합(메시지 추가 규칙): https://langchain-ai.github.io/langgraph/concepts/low_level/#merging-state
- LangChain 메시지 타입 개요: https://python.langchain.com/docs/concepts/messages