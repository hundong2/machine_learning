from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

# model_provider를 'google'로 변경
# 사용할 Gemini 모델 이름을 'model'에 지정 (예: 'gemini-1.5-flash')
# api_key 대신 google_api_key 사용
model = init_chat_model(
    model="gemini-2.5-pro",
    model_provider="google_genai", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

def get_model():
    return model

if __name__=="__main__":
    response = model.invoke("Hello, world!")
    print(response)