# langgraph

- `Node`와 `Edge`로 구성된 자료구조, 객체 간의 관계를 표현하는 데 매우 효과적.  
- `DAG(Directed Acyclic Graph)` : 순환이 없는 그래프, 많은 워크플로우 시스템이 DAG만을 이용.
  - LangGraph는 순환 그래프도 사용가능. 

- `상태`, `노드`, `에지`

## 상태
  - 상태는 그래프 실행 과정에서 지속적으로 유지되는 데이터로 각 노드가 실행 될 때 마다 읽고 쓸 수 있음. 
  - 전체 워크플로의 컨텍스트를 관리하는 중앙 저장소 역할.  
  - 상태는 `TypedDict`나 `Pydantic` 모델로 정의되어 타입 안전성을 보장.  

## 노드 

- 그래프의 기본 실행 단위 
- 각 노드는 특정 작업을 수행하는 함수나 에이전트를 나타냄

### 특징

- 현재 상태를 입력으로 받음
- 특정 작업을 수행(LLM호출, 데이터 처리, 외부 API호출 등)
- 업데이트 된 상태를 반환 

## 에지 

- 노드간의 연결을 정의, 실행 흐름을 제어

### 일반 에지

- 항상 같은 경로로 진행

### 조건부 에지

- 상태에 따라 다른 노드로 분기 
- 조건부에서는 동적 라우팅을 가능하게 하여, 런타임에 실행 경로를 결정할 수 있다. 

[example langgraph](./hello_langgraph.py). 

## 조건부 라우팅 적용: 감정 분석 챗봇

- example, 시작노드 -> 감정 파악 -> 부정 답변 | 중립 답변 | 긍정 답변 -> 종료 노드
- [Conditional LangGraph example](./conditional_routing.py)  

## check pointer를 활용한 상태 관리

- check pointer system은 영속성과 오류 복구를 위한 핵심 기능 
- 상태의 영속성 이란? 
  - 그래프 실행 중 각 노드의 상태를 저장 한다는 의미 
  - 여러 대화나 세션의 상태를 독립적으로 관리할 수 있어서 동시에 여러 워크플로를 처리할 수 있습니다.  
- 사용법
  - 체크 포인터 설정
  - 그래프에 체크 포인터 연결
  - 스레드 ID로 상태 관리 

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

checkpointer=SqliteSaver.from_conn_string(":memory:") #check pointer setting
app = StateGraph(state_schema).compile(checkpointer=checkpointer) #graph에 체크 포인터 연결
config = { "configure":{"thread_id": "thread-1"}}
result = app.invoke(input_data, config=config)

```

- 기본적으로 제공하는 check pointer
  - `BaseCheckpointSaver` : 추상 기본 클래스
  - `InMemorySaver` : 메모리 기반 구현
  - `SQLiteSaver`, `PostgresSaver` : 영구 저장소 구현 
