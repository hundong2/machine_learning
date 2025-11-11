import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from geopy.geocoders import Nominatim
import math
import os

from dotenv import load_dotenv
load_dotenv()

def calculator(expression: str) -> str:
    """math expression 계산기"""
    try:
        expression = expression.replace("sqrt", "math.sqrt")
        expression = expression.replace("sin", "math.sin")
        expression = expression.replace("cos", "math.cos")

        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"계산 결과: {result}"
    except Exception as ex:
        return f"계산 오류: {str(ex)}"
    
def get_weather(city_name: str) -> str:
    """도시 이름으로 날씨 정보 가져오기"""
    if city_name:
        latitude, longitude = get_coordinates(city_name)
    else:
        raise ValueError("도시 이름이 제공되지 않았습니다.")
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
    response = httpx.get(url)
    response.raise_for_status()
    return response.json()

def get_coordinates(city_name: str) -> str:
    """도시 이름으로 위도/경도 가져오기"""
    
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(city_name)
    if location:
        print(f"{city_name}의 좌표는 위도 {location.latitude}, 경도 {location.longitude}입니다.")
        return location.latitude, location.longitude
    else:
        print(f"{city_name}의 좌표를 찾을 수 없습니다.")
        raise ValueError("좌표를 찾을 수 없습니다.")
    
def currency_convertor(amount: float, from_currency: str, to_currency: str) -> str:
    """통화 간 환율을 계산한다."""
    print(f"{amount} {from_currency} to {to_currency} 환율 계산 중...")
    rates = {("USD", "KRW"): 1300.0, ("KRW", "USD"): 0.00077}
    rate_key = (from_currency.upper(), to_currency.upper())
    if rate_key in rates:
        rate = rates[rate_key]
        converted = amount * rate
        return f"{amount} {from_currency} = {converted:.2f} {to_currency}"

def should_continue(state: MessagesState):
    print("\n---분기 결정 ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print(f"결정: 도구 호출 필요 ({len(last_message.tool_calls)}개)")
        return "tools"
    else:
        print("결정: 도구 호출 불필요, 종료")
        return END

#call_model 노드를 생성하는 함수 
def create_call_model_function(model_with_tools):
    """model_with_tools를 호출하는 함수 생성"""
    def call_model(state: MessagesState):
        """LLM을 호출하여 응답을 생성하는 노드 함수"""
        last_message = state["messages"][-1]
        #도구 실행 결과를 받았는지, 아니면 사용자 질문을 받았는지에 따라 분기 
        if isinstance(last_message, ToolMessage):
            print("\n모델 호출 (도구 결과 기반)")
            print(f"도구 결과: {last_message.content[:300]}...")
        else:
            print("\n모델 호출 (사용자 입력 기반)")
            print(f"사용자 입력: {last_message.content}...")
        #모델을 호출하여 다음 행동을 결정하게 함 
        response = model_with_tools.invoke(state["messages"])

        if response.tool_calls:
            print(f"모델의 판단: 도구 호출 -> {response.tool_calls}")
        else:
            print(f"모델의 판단: 응답 생성 -> {response.content}...")
        return {"messages": [response]}
    return call_model

def create_graph(model_with_tools, tool_node):
    """LLM workflow 그래프 생성"""
    workflow = StateGraph(MessagesState)
    call_model = create_call_model_function(model_with_tools)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "call_model")
    workflow.add_conditional_edges("call_model", should_continue, ["tools", END])
    workflow.add_edge("tools", "call_model")
    return workflow.compile()

def llm_tool_call(query: str):
    """하나의 질문에 대해 전체 LLM 워크플로우를 설명하고 로그를 출력합니다."""
    tools = [ calculator, get_weather, currency_convertor ]
    tool_node = ToolNode(tools)
    model = init_chat_model(
        model="gemini-2.0-flash",
        model_provider="google_genai", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    model_with_tools = model.bind_tools(tools)

    print(f"질문 : {query}")
    print("-" * 50)

    #LLM 기반 워크플로 생성
    app = create_graph(model_with_tools, tool_node)
    #그래프 시각화
    mermaid_png = app.get_graph().draw_mermaid_png()
    with open("./06.png", "wb") as f:
        f.write(mermaid_png)
    app.invoke({"messages": [HumanMessage(content=query)]})

    print("-" * 50)
    print("처리 완료")
    print("=" * 50 + "\n")

def main():
    print("===LangGraph ToolNode 예제 (LLM 기반)===")

    test_queries = [
        "2+3 * 4의 값을 계산해줘",
        "서울 날씨 어 때?",
        "100달러를 원화로 바꿔줘",
        "sqrt(16)을 계산해줘",
        "도쿄 날씨가 궁금해",
        "1000원을 달러로 환전해줘"
    ]

    print("\nLLM기반 도구 호출 시작")
    for query in test_queries:
        try:
            llm_tool_call(query)
        except Exception as e:
            print(f"오류 발생: {e}")
            print("-" * 50)
if __name__ == "__main__":
    main()