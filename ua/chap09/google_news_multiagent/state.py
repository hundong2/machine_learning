from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict
from langchain_core.messages import BaseMessage
from langchain.graph.message import add_messages

class NewsState(BaseModel):
    """뉴스 처리 상태를 관리하는 데이터 모델입니다.

    Attributes:
        messages (list[BaseMessage]): 대화 기록을 저장하는 리스트입니다. `add_messages`로 어노테이트되어 Langchain 그래프에서 메시지를 추가하는 데 사용됩니다.
        raw_news (list[dict[str, Any]]): RSS 피드에서 가져온 원시 뉴스 데이터를 저장하는 리스트입니다. 각 항목은 뉴스 기사의 속성을 담은 사전입니다.
        summarized_news (list[dict[str, Any]]): 요약된 뉴스 데이터를 저장하는 리스트입니다. 각 항목은 요약된 뉴스 기사의 속성을 담은 사전입니다.
    """
    # pydantic이 알 수 없는 유형을 허용하도록 구성합니다.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # 대화 기록 저장을 위한 필드입니다.  Langchain 그래프에 메시지를 추가하는 데 사용됩니다.
    messages: Annotated[list[BaseMessage], add_messages] = []
    # RSS 피드에서 수집한 원시 뉴스 데이터를 저장합니다. 각 항목은 뉴스 기사의 속성을 담은 사전입니다.
    raw_news: list[dict[str, Any]] = []
    # 요약된 뉴스 데이터를 저장합니다. 각 항목은 요약된 뉴스 기사의 속성을 담은 사전입니다.
    summarized_news: list[dict[str, Any]] = []
    categorized_news: dict[str, list[dict[str,Any]]] = {}
    final_report: str = "" #리포트 문자열로 저장
    error_log: list[str] =[]#error log 기록
