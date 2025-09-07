from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()
prompt = ChatPromptTemplate.from_template(
    "주어지는 문구에 대하여 50자 이내의 짧은 시를 작성해 주세요: {word}"
)

chat_model = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)
parser = StrOutputParser()

chain = prompt | chat_model | parser 
result = chain.invoke({"word": "sky"})
print(result)

"""
푸른 하늘 위로 새는 노래, 별빛이 흐르는 바람.
"""