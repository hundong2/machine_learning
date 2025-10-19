"""
이 파일은 '루프 에이전트' 예제입니다.
여러 하위 에이전트를 순환(반복) 실행하면서 조건을 만족하면 멈추는 구조를 보여줍니다.

핵심 개념(문법 중심 간단 설명):
- import: 다른 모듈에서 클래스/함수(여기서는 에이전트 관련)를 가져옵니다.
- 키워드 인자: name=..., model=... 처럼 '이름=값'으로 전달하면 순서와 무관하게 의미가 명확해집니다.
- 문자열 연결: 파이썬은 줄 끝에 역슬래시(\)를 두고 다음 줄에 문자열을 두면 이어 붙여 해석됩니다.
- 클래스 상속: class X(BaseAgent): 처럼 BaseAgent의 기능을 물려받아 확장합니다.
- 비동기 제너레이터: async def ... -> AsyncGenerator[T, None] 와 같이 정의하고, 내부에서 yield로 값을 여러 번 내보냅니다.
- 상태 읽기: ctx.session.state.get("키", 기본값)으로 이전 단계(또는 같은 루프 내)의 출력을 읽습니다.
"""

from google.adk.agents import LoopAgent, LlmAgent, BaseAgent 
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# 랜덤 메시지 생성 에이전트
# - LlmAgent: LLM을 호출하는 기본 에이전트 유형입니다.
# - output_key: 이 에이전트의 출력이 저장될 상태 키 이름입니다(후속 에이전트가 참조).
random_generator = LlmAgent(
    name="RandomGenerator",
    model="gemini-2.5-flash",
    description="랜덤한 메시지를 생성하는 에이전트 입니다. 스팸 메시지와 정상적인 메시지를 60:40 비율로 생성합니다.",
    output_key="random_message",
    # instruction: 모델에 줄 지시문(문자열). 아래는 두 문자열 리터럴을 줄바꿈하여 이어 붙인 형태입니다.
    #  - 역슬래시(\) + 다음 줄의 문자열 → 하나로 합쳐짐
    instruction="스팸 메시지와 정상적인 메시지를 60:40 확률로 생성합니다. 스팸 메시지는 '[웹발신]'을 앞에 넣고, 정상적인 메시지는 " \
    "따로 표시 하지 않아도 됩니다. 반드시 하나의 메시지만 출력해야 합니다.",
)

# 메시지가 스팸인지 검사하는 에이전트
# - random_generator의 출력(random_message)을 내부적으로 참고하도록 설정되어 있을 수 있습니다(프레임워크 규칙에 따름).
spam_checker=LlmAgent(
    name="SpamChecker",
    model="gemini-2.5-flash",
    description="메시지가 스팸인지 확인하세요. 스팸이면 'fail', 아니면 'pass'를 반환하세요.",
    output_key="spam_status"
)

# 사용자 정의 에이전트: BaseAgent를 상속해 맞춤 로직을 작성합니다.
# - _run_async_impl: 비동기 제너레이터 메서드로, 필요 시 여러 이벤트를 yield로 내보냅니다.
# - ctx: InvocationContext(실행 컨텍스트)로, 세션 상태/입력/메타데이터 등의 접근을 제공합니다.
class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(
            self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 세션 상태에서 spam_status 값을 읽습니다. 없으면 기본값 "fail"을 사용합니다.
        status = ctx.session.state.get("spam_status", "fail")
        # 비교 연산(status == "pass")의 결과는 불리언(True/False)입니다.
        # True면 멈춤(escalate), False면 계속(loop)
        should_stop = status == "pass"  # pass면 멈추기
        # EventActions(escalate=...)를 담은 Event를 yield하여 루프 에이전트에 신호를 보냅니다.
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))

# 루프 에이전트: 하위 에이전트들을 반복 실행합니다.
# - max_iterations: 루프의 최대 반복 횟수(안전장치)
# - sub_agents: 실행 순서대로 나열된 하위 에이전트 리스트
#   1) random_generator → 메시지 생성
#   2) spam_checker → 스팸 여부 판단(결과를 spam_status에 저장)
#   3) CheckStatusAndEscalate → spam_status가 "pass"면 escalate=True로 루프 종료
root_agent = LoopAgent(
    name="SpamCheckLoop",
    max_iterations=10,
    sub_agents=[random_generator, spam_checker, CheckStatusAndEscalate(name="StopChecker")]
)