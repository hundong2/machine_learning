from langchain.chat_models import init_chat_model
import os 
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="model-identifier",
    model_provider="openai", 
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)

def get_model():
    return model

if __name__=="__main__":
    response = model.invoke("Hello, world!")
    print(response)