from google.adk.agents import Agent
from pydantic import BaseModel, Field

class BookRecommendation(BaseModel):
    title: str = Field(description="Title 제목")
    author: str = Field(description="Author 저자")
    genre: str = Field(description="Genre 장르")
    reason: str = Field(description="Reason for recommendation 추천 이유")
    rating: float = Field(description="Rating 평점 (0.0 to 5.0)")

class BookList(BaseModel):
    recommendations: list[BookRecommendation]
    total_count: int

root_agent = Agent(
    name="book_recommender",
    model="gemini-2.5-pro",
    description="책을 추천하고 구조화된 형식으로 반환",
    instruction="""
    사용자의 관심사에 맞는 책을 추천하세요. 
    반드시 지정된 JSON 형식으로 응답해야 합니다.
    """,
    output_schema=BookList
)