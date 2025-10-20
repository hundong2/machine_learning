from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
import random
import os
from dotenv import load_dotenv
#class state definition, 워크플로 전체에서 공유되는 데이터 구조
class EmotionBotState(BaseModel):
    user_message: str = Field(default="", description="사용자 입력 메시지")
    emotion: str = Field(default="", description="분석된 감정")
    response: str = Field(default="", description="최종 응답 메세지")


load_dotenv()

# model_provider를 'google'로 변경
# 사용할 Gemini 모델 이름을 'model'에 지정 (예: 'gemini-1.5-flash')
# api_key 대신 google_api_key 사용
llm = init_chat_model(
    model="gemini-2.5-pro",
    model_provider="google_genai", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

def analyze_emotion(state: EmotionBotState) -> Dict[str, Any]:
    message=state.user_message
    print(f"LLM 감정 분석 중: ''{message}''")
    messages = [
        SystemMessage(
            content="당신은 감정 분석 전문가 입니다. 사용자의 메시지를 분석하여 'positive', 'negative', 'neutral' 중 하나로 감정을 분류해주세요. " \
            "답변은 반드시 하나의 단어만 출력하세요."
        ),
        HumanMessage(content=f"메시지: '{message}'의 감정을 분석해주세요.")
    ]
    response = llm(messages)
    emotion = response.content.strip().lower()

    #유효성 검사
    if emotion not in ['positive', 'negative', 'neutral']:
        emotion = 'neutral'  #기본값 설정
    print(f"분석된 감정: {emotion}")
    return {"emotion": emotion}

def generate_positive_response(state: EmotionBotState) -> Dict[str, Any]:
    responses = [ "정말 좋은 소식이네요!", "기분이 좋으시군요!", "멋지네요!"]
    return {"response": random.choice(responses)}

def generate_negative_response(state: EmotionBotState) -> Dict[str, Any]:
    responses = [
        "힘든 시간을 보내고 계시군요. 괜찮으실 거예요.",
        "무슨 일이 있었나요? 이야기해 주세요.",
        "당신의 기분이 나아지길 바랍니다."
    ]
    return {"response": random.choice(responses)}

def generate_neutral_response(state: EmotionBotState) -> Dict[str, Any]:
    responses = [
        "그렇군요. 더 이야기해 주세요.",
        "알겠습니다. 다른 이야기도 해볼까요?",
        "흥미롭네요. 계속 말씀해 주세요."
    ]
    return {"response": random.choice(responses)}

def route_by_emotion(
        state: EmotionBotState,
) -> Literal["positive_response", "negative_response", "neutral_response"]:
    emotion = state.emotion
    print(f"감정에 따른 라우팅: {emotion}")
    if emotion == "positive":
        return "positive_response"
    elif emotion == "negative":
        return "negative_response"
    else:
        return "neutral_response"
    
def create_emotion_bot_graph():
    workflow = StateGraph(EmotionBotState)
    workflow.add_node("analyze_emotion", analyze_emotion)
    workflow.add_node("positive_response", generate_positive_response)
    workflow.add_node("negative_response", generate_negative_response)
    workflow.add_node("neutral_response", generate_neutral_response)
    workflow.add_edge(START, "analyze_emotion")
    workflow.add_conditional_edges(
        "analyze_emotion",
        route_by_emotion,
        {
            "positive_response": "positive_response",
            "negative_response": "negative_response",
            "neutral_response": "neutral_response",
        }
    )
    workflow.add_edge("positive_response", END)
    workflow.add_edge("negative_response", END)
    workflow.add_edge("neutral_response", END)

    return workflow.compile()

def main():
    print("감정 분석 챗봇 예제 시작")
    app = create_emotion_bot_graph()

    test_cases = [
        "오늘 정말 행복해!",
        "요즘 너무 힘들어.",
        "그냥 그런 하루였어."
    ]
    for i, message in enumerate(test_cases, 1):
        print(f"test {i}: '{message}'")
        state = EmotionBotState(user_message=message)
        result = app.invoke(state)
        print(f"최종 응답: {result['response']}\n")
        #mermaid_png = app.get_graph().draw_mermaid_png()
        #with open(f"./{i}conditional_routing.png", "wb") as f:
        #    f.write(mermaid_png)
if __name__ == "__main__":
    main()