from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import BaseMessage
from langchain.graph.message import add_messages

class NewsState(BaseModel):
    """뉴스처리 상태를 관리하는 데이터 모델"""
    #pydantic 이 모르는 타입을 허용
    model_config = ConfigDict(arbitrary_types_allowed=True)
    #대화의 히스토리 저장을 위한 필드
    messages: Annotated[list[BaseMessage], add_messages] = []
    raw_news: list[dict[str, Any]] = []# RSS 피드에서 수집한 뉴스 데이터 저장
    summarized_news: list[dict[str, Any]] = []# 요약된 뉴스 데이터 저장
    