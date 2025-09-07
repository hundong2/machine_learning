from langchain_core.runnables import RunnablePassthrough, RunnableParallel
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

#병렬 처리 체인 구성 
#하나의 입력을 받아 여러 작업을 동시에 수해할 수 있다. 
chain = RunnableParallel(
    {
        "original": RunnablePassthrough(), #original data, 받은 입력을 그대로 통과 시킴 
        "processed": prompt | model | parser # processed data
    }
)

result = chain.invoke({"word": "사과"})
print(result)
"""
{'original': {'word': '사과'}, 'processed': '과일, 과수, 과정'}
"""