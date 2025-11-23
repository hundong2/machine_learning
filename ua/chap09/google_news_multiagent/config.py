```python
# project 설정
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    애플리케이션의 설정을 관리하는 클래스입니다.
    환경 변수에서 값을 읽어오고, 기본 설정을 제공하며, 설정 값의 유효성을 검사합니다.

    Attributes:
        GEMINI_API_KEY (str): Google Gemini API 키입니다. 환경 변수 'GOOGLE_API_KEY'에서 로드됩니다.
        GEMINI_MODEL (str): 사용할 Gemini 모델의 이름입니다. 기본값은 "gemini-2.5-pro"입니다.
        GEMINI_API_BASE_URL (str): Gemini API의 기본 URL입니다. 기본값은 "https://generativelanguage.googleapis.com/v1beta"입니다.
        GEMINI_MODEL_PROVIDER (str): 모델 제공자 이름입니다. 기본값은 "google_genai"입니다.
        MAX_TOKENS (int): API 호출 시 사용할 최대 토큰 수입니다. 기본값은 2048입니다.
        TEMPERATURE (float): 생성된 텍스트의 무작위성을 제어하는 값입니다. 0과 1 사이의 값이며, 값이 높을수록 더 무작위적인 텍스트가 생성됩니다. 기본값은 0.7입니다.
        TOP_P (float): nucleus sampling을 위한 확률 임계값입니다.  모델이 고려할 확률 분포의 상위 p%의 토큰만 고려합니다. 기본값은 0.9입니다.
        FREQUENCY_PENALTY (float): 모델이 이미 사용한 단어를 반복하지 않도록 하는 페널티입니다. 기본값은 0.0입니다.
        PRESENCE_PENALTY (float): 모델이 새로운 단어를 사용하도록 장려하는 페널티입니다. 기본값은 0.0입니다.
        ROOT_DIR (str): 현재 파일이 있는 디렉토리의 절대 경로입니다.
        RSS_URL (str): Google News RSS 피드 URL입니다. 기본값은 "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"입니다.
        MAX_NEWS_COUNT (int): RSS 피드에서 가져올 최대 뉴스 기사 수입니다. 기본값은 10입니다.
        NEWS_CATEGORIES (list[str]): 분석할 뉴스 카테고리 목록입니다.
        NEWS_PER_CATEGORY (int): 각 뉴스 카테고리별로 가져올 뉴스 기사 수입니다.
        OUTPUT_DIR (str): 출력을 저장할 디렉토리입니다. ROOT_DIR을 기준으로 생성됩니다.

    Methods:
        validate() -> bool: API 키가 설정되었는지 확인합니다.
    """
    # OpenAI API 키를 환경 변수에서 로드합니다.
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    # 사용할 Gemini 모델의 이름입니다.
    GEMINI_MODEL = "gemini-2.5-pro"
    # Gemini API의 기본 URL입니다.
    GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    # 모델 제공자 이름
    GEMINI_MODEL_PROVIDER = "google_genai"
    # API 호출 시 사용할 최대 토큰 수입니다.
    MAX_TOKENS = 2048
    # 생성된 텍스트의 무작위성을 제어하는 값입니다.
    TEMPERATURE = 0.7
    # nucleus sampling을 위한 확률 임계값입니다.
    TOP_P = 0.9  # nucleus sampling
    # 모델이 이미 사용한 단어를 반복하지 않도록 하는 페널티입니다.
    FREQUENCY_PENALTY = 0.0
    # 모델이 새로운 단어를 사용하도록 장려하는 페널티입니다.
    PRESENCE_PENALTY = 0.0

    # 현재 파일이 있는 디렉토리의 절대 경로를 구합니다.
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Google News RSS 피드 URL입니다.
    RSS_URL: str = "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"
    # RSS 피드에서 가져올 최대 뉴스 기사 수입니다.
    MAX_NEWS_COUNT: int = 10

    # 분석할 뉴스 카테고리 목록입니다.
    NEWS_CATEGORIES: list[str] = [
        "정치", "경제", "사회", "문화", "세계", "IT/과학"
    ]

    # 각 뉴스 카테고리별로 가져올 뉴스 기사 수입니다.
    NEWS_PER_CATEGORY: int = 30

    # 출력을 저장할 디렉토리입니다.
    OUTPUT_DIR: str = f"{ROOT_DIR}/outputs"

    @classmethod
    def validate(cls) -> bool:
        """
        애플리케이션 설정을 검증합니다.

        Returns:
            bool: 설정이 유효하면 True, 그렇지 않으면 False를 반환합니다.
        """
        # OpenAI API 키가 설정되지 않았는지 확인합니다.
        if not cls.GEMINI_API_KEY:
            print("OpenAI API key가 설정되지 않았습니다.")
            print("환경 변수 'GOOGLE_API_KEY'를 설정해주세요")
            return False
        return True
```