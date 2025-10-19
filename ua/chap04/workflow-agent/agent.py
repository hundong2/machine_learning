"""
이 파일은 여러 에이전트를 조합하여 작업을 수행하는 예제입니다.

핵심 개념 요약(초보자용):
- Agent: LLM(모델)에 "역할/지시문"을 주고, 입력을 받아 출력을 만드는 구성 요소입니다.
- output_key: 해당 Agent의 출력에 붙일 이름(키). 다음 단계에서 이 키로 값을 참조합니다.
- ParallelAgent: 여러 Agent를 동시에 실행하고, 각자의 output_key 결과를 모아서 딕셔너리로 반환합니다.
- SequentialAgent: 나열된 Agent들을 순서대로 실행하며, 앞 단계의 결과 딕셔너리를 다음 단계 입력(템플릿 치환)에 자동 전달합니다.
- 템플릿 placeholder: instruction 문자열 안의 {weather_info} 같은 부분은 실행 시 같은 이름의 키를 가진 값으로 치환됩니다.

문법 팁:
- 키워드 인자(name=..., model=...)는 순서에 구애받지 않고 의미를 명확히 전달합니다.
- 리스트([...])는 여러 값을 순서 있게 담는 자료형이며, sub_agents=[a,b,c]는 세 개의 에이전트를 담는 예입니다.
"""

from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

# 날씨 정보를 수집하는 단일 에이전트 정의
# - name: 에이전트 식별용 이름(문자열)
# - model: 사용할 언어 모델 이름
# - output_key: 이 에이전트의 결과를 담아둘 키 이름(후속 단계에서 {weather_info}로 참조)
# - instruction: 모델에게 줄 지시문(자연어 명령)
# - tools: 에이전트가 사용할 수 있는 도구 목록(여기서는 구글 검색 도구)
weather_fetcher = Agent(
    name="weather",
    model="gemini-2.5-flash",
    output_key="weather_info",
    instruction="오늘의 날씨 정보를 제공하세요.",
    tools=[google_search]
)

# 뉴스 요약을 수집하는 에이전트
news_fetcher= Agent(
    name="news",
    model="gemini-2.5-flash",
    output_key="news_info",
    instruction="오늘의 주요 뉴스를 요약하세요.",
    tools=[google_search]
)

# 주식 시장 동향을 수집하는 에이전트
stock_fetcher= Agent(
    name="stocks",
    model="gemini-2.5-flash",
    output_key="stock_info",
    instruction="주요 주식 시장 동향을 제공하세요.",
    tools=[google_search]
)

# 세 개의 수집 에이전트를 병렬로 실행하는 오케스트레이터
# - sub_agents: 동시에 실행할 에이전트들을 리스트로 전달
#   실행 결과는 {"weather_info": ..., "news_info": ..., "stock_info": ...} 형태의 딕셔너리로 합쳐집니다.
parallel_fetcher=ParallelAgent(
    name="multi_info_fetcher",
    sub_agents=[weather_fetcher, news_fetcher, stock_fetcher],
    description="여러 정보를 동시에 수집"
)

# 종합 브리핑을 작성하는 에이전트
# instruction 내부의 {weather_info}/{news_info}/{stock_info}는 바로 위 병렬 실행의 결과 딕셔너리에서
# 동일한 키 이름을 찾아 치환됩니다. (SequentialAgent가 중간 결과를 다음 단계로 자동 전달)
summarizer = Agent(
    name="daily_briefing",
    model="gemini-2.5-pro",
    instruction="""
    수집된 정보를 종합하여 일일 브리핑을 작성하세요:
    - 날씨 : {weather_info}
    - 뉴스 : {news_info}
    - 주식 시장 동향 : {stock_info}
    간결하고 읽기 쉬운 형식으로 정리하세요. 
    """
)

# 순차 파이프라인: 먼저 병렬 수집(parallel_fetcher)이 실행되고, 그 결과가 summarizer로 전달됩니다.
# - sub_agents의 순서가 실행 순서를 의미합니다.
daily_briefing_pipeline = SequentialAgent(
    name="daily_briefing_pipeline",
    sub_agents=[parallel_fetcher, summarizer],
    description="정보를 병렬로 수집한 후 종합 브리핑 생성"
)

# 최종 진입점 에이전트(실행 시 이 에이전트를 호출하면 전체 파이프라인이 동작)
root_agent = daily_briefing_pipeline