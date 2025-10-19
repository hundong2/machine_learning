# 시스템 프롬프트(system prompt)란?
LLM에게 “역할·목표·제약”을 최우선 순위로 규정해 전체 응답 방식을 결정하는 초기 지시문.

## 예시 파일
https://platform.openai.com/docs/guides/text-generation/chat-completions-api

## 답변
- 정의: 대화 시작 시 모델에 주는 최상위 지시. 모델의 역할, 말투, 금지/허용 사항, 도구 사용 규칙 등을 규정하며 사용자 메시지보다 우선한다.
- 목적: 일관된 페르소나와 안전 가드레일을 유지하고, 과업 범위와 출력 형식을 안정화한다.
- 특징
  - 우선순위 높음: system > developer/instruction > user > assistant.
  - 지속성: 세션 동안 전반적 행동을 관장.
  - 구체성 중요: 모호하면 모델이 임의 해석.

간단 예시
````python
# OpenAI Chat API
messages = [
  {"role": "system", "content": "당신은 간결하고 정확한 데이터 분석 조교입니다. 숫자는 소수점 둘째 자리로."},
  {"role": "user", "content": "MAE와 RMSE 차이 설명해줘"}
]

# LangChain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
  ("system", "당신은 친절한 수학 튜터입니다. 증명은 단계별로."),
  ("human", "{question}")
])

# openai-agents 라이브러리(예시): instruction 필드가 사실상 system prompt 역할
hello_agent = Agent(
    name="HelloAgent",
    description="Greet the user.",
    model="gpt-4o-mini",
    instruction="항상 한글로, 한 문장 인사만.",
    tools=[]
)
````

베스트 프랙티스
- 역할·목표·제약·출력형식을 명시적으로 분리
- 간결하되 예시(few-shot)로 의도 고정
- 금지사항과 경계 조건(토큰 한도, 길이, 포맷)을 구체화

### 추가 자료
- https://platform.openai.com/docs/guides/prompt-engineering
- https://python.langchain.com/docs/concepts#prompts