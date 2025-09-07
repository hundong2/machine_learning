from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

llm = init_chat_model(
    model="model-identifier",
    model_provider="openai", 
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)

google = init_chat_model(
    model="gemini-2.5-pro",
    model_provider="google_genai", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

class MovieReview(BaseModel):
    title: str = Field(..., description="The title of the movie")
    review: str = Field(..., description="The review text")
    rating: int = Field(..., ge=1, le=5, description="The rating given to the movie (1-5)")

structured_llm = llm.with_structured_output(MovieReview)
result: MovieReview = structured_llm.invoke("인셉션 영화에 대해 리뷰를 작성해주세요.")

print(result.title)
print(result.review)
print(result.rating)
"""
인셉션(2010) – 꿈과 현실이 뒤엮이는 시간의 정밀한 조각가
We’ll ……… ...… ? .. ??…… 수정 일정…….…‑… 공화, 다미라……………“지금은… 신과… ……... 아로 오?…‐다니?
1
"""
print("================")
google_llm = google.with_structured_output(MovieReview)
result: MovieReview = google_llm.invoke("인셉션 영화에 대해 리뷰를 작성해주세요.")

print(result.title)
print(result.review)
print(result.rating)
"""
인셉션
인생 최고의 영화!
5
"""
