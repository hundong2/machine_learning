from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os 
import re

load_dotenv()

# model_provider를 'google'로 변경
# 사용할 Gemini 모델 이름을 'model'에 지정 (예: 'gemini-1.5-flash')
# api_key 대신 google_api_key 사용
llm = init_chat_model(
    model="gemini-2.5-pro",
    model_provider="google_genai", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

import json

class MemoryBotState(BaseModel):
    # description 오탈자 수정 (dscription -> description)
    user_message: str = Field(default="", description="사용자 입력 메시지")
    user_name: str = Field(default="", description="사용자 이름")
    user_preference: Dict[str, Any] = Field(
        default_factory=dict, description="사용자 선호도"
    )
    response: str = Field(default="", description="최종 응답")

def process_message(state: MemoryBotState) -> Dict[str, Any]:
    message = state.user_message
    user_name = state.user_name
    preference = state.user_preference.copy()
    system_prompt = f"""
    당신은 사용자의 정보를 기억하는 메모리 봇입니다. 
    현재 기억하는 정보:
    - 사용자 이름: {user_name if user_name else "모름"}
    - 좋아하는 것: {preference.get("likes", [])}
    - 싫어하는 것: {preference.get("dislikes", [])}
    사용자 메시지를 분석하여 다음 JSON 형식으로 응답하세요. 
    {{
        "response": "사용자에게 줄 응답 메세지",
        "new_name": "새로운 사용자 이름 (변경사항 없으면 빈 문자열)",
        "new_likes": ["새로 추가된 좋아하는 것들"],
        "new_dislikes": ["새로 추가된 싫어하는 것들"]
    }}
    """
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
    response = llm.invoke(messages)
    content = getattr(response, "content", "")
    # 모델이 JSON만 반환하지 않는 경우를 대비한 안전한 파싱 처리
    try:
        result = json.loads(content)
    except Exception:
        # 본문에서 JSON 블록만 추출 시도
        m = re.search(r"\{[\s\S]*\}", str(content))
        if m:
            try:
                result = json.loads(m.group(0))
            except Exception:
                result = {"response": str(content), "new_name": "", "new_likes": [], "new_dislikes": []}
        else:
            # 최후 수단: 모델 응답 전체를 텍스트로 사용
            result = {"response": str(content), "new_name": "", "new_likes": [], "new_dislikes": []}
    if result.get("new_name"):
        user_name=result["new_name"]
    if result.get("new_likes"):
        preference.setdefault("likes", []).extend(result["new_likes"])
    if result.get("new_dislikes"):
        preference.setdefault("dislikes", []).extend(result["new_dislikes"])
    bot_response = result.get("response", "죄송해요, 이해하지 못했어요.")

    return {
        "response": bot_response,
        "user_name": user_name,
        "user_preference": preference
    }

def create_memory_bot_graph():
    checkpointer = InMemorySaver()#랭그래프의 체크포인트 시스템 중 가장 기본적인 구현체, 메모리에 상태를 저장하는 방식
    # 휘발성, 빠른 속도, 개발/테스트용 Production에서는 SQLiteSaver, PostgresSaver 등 사용 
    workflow = StateGraph(MemoryBotState)
    workflow.add_node("process_message", process_message)
    workflow.add_edge(START, "process_message")
    workflow.add_edge("process_message", END)
    return workflow.compile(checkpointer=checkpointer)

def main():
    print("===Inmemory bot test===")
    app = create_memory_bot_graph()
    thread_id = "gyul_123"
    conversations = [
        "안녕하세요! 저는 길동이라고 해요.",
        "저는 축구를 정말 좋아해요.",
        "저는 매운 음식을 싫어해요.",
        "오늘 날씨가 정말 좋네요!",
        "저는 여행 가는 것을 좋아해요.",
        "내이름은 뭐였지?",
        "내가 좋아하는것과 싫어하는 것은?"
    ]
    for i, message in enumerate(conversations, 1):
        print(f"[{i}] user: {message}")
        config = { "configurable": {"thread_id": thread_id} }
        result = app.invoke({"user_message": message}, config)
        print(f"[{i}] bot : {result['response']}")
        print(
            f"memory name = {result.get('user_name')},"
            f"favorite={result.get('user_preference', {})}\n"
        )
if __name__ == "__main__":
    main()