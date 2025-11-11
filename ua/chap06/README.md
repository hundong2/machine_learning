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

- `loop_workflow`
  - [example code](./loop_workflow.py)
- output

```sh
game start!
입력: 5
[1번째 시도] 추측: 5
[check_guess] 5 ( 시도: 1회)
[route_game] 현재 상태: playing, 시도 횟수: 1
입력: 3
[2번째 시도] 추측: 3
[check_guess] 3 ( 시도: 2회)
[route_game] 현재 상태: playing, 시도 횟수: 2
입력: 10
[3번째 시도] 추측: 10
[check_guess] 10 ( 시도: 3회)
[route_game] 현재 상태: playing, 시도 횟수: 3
입력: 55
[4번째 시도] 추측: 55
[check_guess] 55 ( 시도: 4회)
[route_game] 현재 상태: playing, 시도 횟수: 4
입력: 4
[5번째 시도] 추측: 4
[check_guess] 4 ( 시도: 5회)
시도 횟수 초과
[route_game] 현재 상태: lost, 시도 횟수: 5
최종 결과: 아쉽네요! 최대 시도 횟수를 초과했습니다. 정답은 28였습니다.
게임 상태: lost
총 시도: 5회
```

### Parallel workflow

- [Example](./parallel_execution.py). 
- output

```sh
===대시보드 생성 워크플로우 실행===
대시보드 생성 시작 - 위치 : Seoul
뉴스 데이터 수집 중...
주식 시장 데이터 수집 중...
날씨 데이터 수집 중...
뉴스 요약: 2개의 기사 수집 완료.
날씨: 맑음, 온도: 22°C, 습도: 65%
주식 시장: KOSPI 2500.75, NASDAQ 13000.5
대시보드 보고서 생성 중...
===대시보드 보고서===

    날씨: 맑음, 온도: 22°C, 습도: 65%
    뉴스: 2개의 기사 수집 완료.
    주식 시장: KOSPI 2500.75, NASDAQ 13000.5
    병렬 처리 시간: 1.59초
```

# ToolNode를 사용한 도구 사용기능 만들기 

- ![06.png](./06.png)
- [example code](./tool_calling.py)  