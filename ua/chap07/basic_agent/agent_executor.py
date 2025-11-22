from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message
from a2a.utils import new_agent_text_message
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
load_dotenv()

class HelloAgent:
    """랭체인과 OpenAIfmf tkdydgks rkseksgks Hello Wrold Agent"""
    def __init__(self):
        self.chat = init_chat_model(
        model="gemini-2.0-flash",
        model_provider="google_genai", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 
                    사용자에게 인사하는 역할을 합니다.
                    """,
                ),
                (
                    "user", "{user_input}"
                )
            ]
        )
    async def invoke(self, user_message: str) -> str:
        """유저 메시지를 처리하고 응답을 생성합니다."""
        chain = self.prompt | self.chat
        # 프롬프트 템플릿의 변수 {user_input}과 일치하도록 수정
        # - "message" → "user_input" 으로 변경하여 KeyError 해결
        response = await chain.ainvoke({"user_input": user_message})
        return response.content
class HelloAgentExecutor(AgentExecutor):
    """간단한 Hello World 에이전트의 Executor 구현"""
    def __init__(self):
        self.agent = HelloAgent()

    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue
    ):
        """요청을 처리하고 응답을 생성합니다."""
        message=context.message
        for part in message.parts:
            if part.root.text:
                user_message = part.root.text
        result = await self.agent.invoke(user_message)
        await event_queue.enqueue_event(new_agent_text_message(result))
    async def cancel(
            self,
            context: RequestContext,
            event_queue: EventQueue
    ):
        """요청을 취소"""
        error_msg = "취소 기능은 지원되지 않습니다. Hello World 에이전트는 즉시 응답합니다."
        error_message = Message(
            role="agent",
            parts=[{"type": "text", "text": error_msg}],
            messageId="cancel_error"
        )
        event_queue.enqueue_event(error_message)

