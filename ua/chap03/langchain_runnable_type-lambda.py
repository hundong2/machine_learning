from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()
prompt = ChatPromptTemplate.from_template(
    "주어진 {word}와 유사한 단어 3가지를 나열해주세요. 단어만 나열합니다."
)

model = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)

parser = StrOutputParser()

# Parallel processing chain 구성
# 하나의 입력을 받아 여러 작업을 동시에 수행할 수 있다.
chain = RunnableParallel(
    synonyms = prompt | model | parser, # 유사어 생성 작업
    word_count = RunnableLambda(lambda x: len(x['word'])), # 단어 길이 계산 작업
    uppercase = RunnableLambda(lambda x: x['word'].upper()) # 대문자 변환 작업
)

result = chain.invoke({"word": "peaceful"})
print(result)
"""
{'synonyms': 'tranquil, serene, harmonious', 'word_count': 8, 'uppercase': 'PEACEFUL'}
"""