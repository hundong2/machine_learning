# chapter 4

## Agent 

- reasoning : 현재 상황을 이해하고 추론
- planning : 요청을 해결할 계획을 세우는 것 

### Augmented LLM ( 증강 된 LLM ) 

- Agent Runtime
  - Orchestration 
    - instruction 
    - memory
    - reasoning/ planning 
  - model
  - tools


## google ADK

- [ADK](https://google.github.io/adk-docs/tools/built-in-tools). 

### LLMAgent

- 대규모 언어 모델을 핵심 엔진으로 활용
- 자연어 이해와 추론 : 복잡한 언어적 맥락을 파악하고 논리적 추론을 수행
- 동적 의사 결정 : 상황에 따라 적절한 도구를 선택하고 실행 방향을 결정
- 유연성 : 예측하기 어려운 상황에서도 창의적이고 적응적인 해결책 제시
- 언어 중심 작업 : 텍스트 생성, 번역, 요약 등 언어 관련 작업에 특화 

### Workflow agent

- `SequencialAgent` : 에이전트들을 순차적으로 실행
- `ParallelAgent` : 여러 에이전트를 동시에 병렬 실행
- `LoopAgent` : 특정 조건이 만족될 때까지 반복 실행 

### Custom Agent 

- `Base Agent`를 직접 확장하여 만들어 가장 높은 수준의 맞춤화를 제공 

- `고유한 운영 로직 구현` : 특별한 비즈니스 요구사항에 맞춘 독특한 동작 방식
- `전문화 된 통합` : 특정 시스템이나 서비스와의 깊은 연동
- `유연한 결정론 수준` : 필요에 따라 예측 가능하거나 적응적인 동작 선택 

## 모델 중립성과 풍부한 도구

- `LiteLLM`을 사용하면 GPT나 클로드 같은 다양한 LLM을 함께 사용할 수 있다. 
- `Langchain`이나 `CrewAI`와 쉽게 통합 가능 
- `대화 이력을 관리하는 기능도 세션과 메모리라는 개념으로 편하게 지원`
  - `SessionService`, `MemoryService`

## google ADK는 다음의 파일 구조를 가지고 있어야 한다.

- my_agent_project/
  - my_agent/
    - `agent.py`
    - `.env`
- requirements.txt

- [agent example](./agent_example/agent.py). 

- `adk web`

```sh
pwd
/ch04
adk web
```

```sh
adk run agent_example
```

```sh
adk api_server
```

### 여러 도구를 사용하는 에이전트 : 날씨와 야구 랭킹 에이전트 

- `geopy`: https://github.com/geopy/geopy
- `uv pip install geopy`
- [Multi tool agent](./multi-tool-agent/agent.py). 

### 구조화 된 출력을 지원하는 에이전트

[example code](./structured-output-agent/agent.py). 

- output 

```sh
[user]: 개발 서적 3권만 추천해주세요
[book_recommender]: {
  "recommendations": [
    {
      "title": "클린 코드(Clean Code)",
      "author": "로버트 C. 마틴",
      "genre": "프로그래밍",
      "reason": "소프트웨어 장인 정신을 기르고, 가독성 높고 유지보수하기 쉬운 코드를 작성하는 원칙과 실천 방법을 배울 수 있는 개발자 필독서입니다.",
      "rating": 4.8
    },
    {
      "title": "실용주의 프로그래머(The Pragmatic Programmer)",
      "author": "앤드류 헌트, 데이비드 토머스",
      "genre": "프로그래밍",
      "reason": "특정 기술이 아닌, 프로그래머로서 가져야 할 태도와 습관, 효율적인 개발 철학을 다루어 개발자의 전문성을 키우는 데 큰 도움을 줍니다.",
      "rating": 4.7
    },
    {
      "title": "데이터 중심 애플리케이션 설계",
      "author": "마틴 클레프만",
      "genre": "시스템 설계",
      "reason": "현대적인 데이터 시스템의 근간이 되는 기술과 원리를 깊이 있게 탐구합니다. 분산 시스템과 데이터베이스를 다루는 백엔드 개발자에게 특히 추천합니다.",
      "rating": 4.9
    }
  ],
  "total_count": 3
}
```

### 구글 ADK의 제약 사항

- 단일 에이전트에서는 도구(tools)와 구조화된 출력(output_schema)를 동시에 사용할 수 없음. 
- 즉, 외부 API를 호출하는 도구를 사용하면서 동시에 결과를 특정 JSON형식으로 강제할 수 없다. 
- `멀티 에이전트`로 해결 가능. 

## 멀티 에이전트 

- 첫번쨰 에이전트, 도구를 활용하여 외부 데이터를 수집. 
- 두번째 에이전트, 첫번째 에이전트가 수집한 데이터를 받아 output_schema를 사용하여 원하는 JSON구조로 반환 
- 오케스트레이터 에이전트, 전체 프로세스를 관리, 각 에이전트 간의 데이터 전달과 실행 순서를 조정 

- [Example Multi agent](./multi-agent-for-bestseller-book/agent.py). 

## 워크플로우 에이전트 만들기 : 날씨 정보, 오늘 뉴스, 주식 멀티 에이전트 

- SequencialAgent, ParallelAgent
