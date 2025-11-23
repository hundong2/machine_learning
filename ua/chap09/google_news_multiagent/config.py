#project 설정 
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    #OpenAI 
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.5-pro"
    GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_MODEL_PROVIDER = "google_genai"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.9 #nucleus sampling
    FREQUENCY_PENALTY = 0.0
    PRESENCE_PENALTY = 0.0

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    RSS_URL: str = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    MAX_NEWS_COUNT: int = 10

    NEWS_CATEGORIES: list[str] = [
        "정치", "경제", "사회", "문화", "세계", "IT/과학"
    ]

    NEWS_PER_CATEGORY: int = 30 

    OUTPUT_DIR: str = f"{ROOT_DIR}/outputs"

    @classmethod
    def validate(cls) -> bool:
        if not cls.GEMINI_API_KEY:
            print("OpenAI API key가 설정되지 않았습니다.")
            print("환경 변수 'GOOGLE_API_KEY'를 설정해주세요")
            return False
        return True