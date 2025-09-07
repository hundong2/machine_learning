from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

model = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)
parser = StrOutputParser()

def is_english(x: dict) -> bool:
    return all(ord(char) < 128 for char in x['word'])

english_prompt = ChatPromptTemplate.from_template(
    "Give me 3 synonyms for {word}. only list the words."
)

korean_prompt = ChatPromptTemplate.from_template(
    "주어진 {word}와 유사한 단어 3가지를 나열해주세요. 단어만 나열합니다."
)

langchain_aware_chain = RunnableBranch(
    (is_english, english_prompt | model | parser), # 영어일 경우
    (korean_prompt | model | parser) # 한국어일 경우, 조건이 거짓일 때 실행 될 prompt 
)

english_word = {"word": "happy"}
english_result = langchain_aware_chain.invoke(english_word)
print(f'Synonyms for {english_word["word"]}: {english_result}')

korean_word = {"word": "행복"}
korean_result = langchain_aware_chain.invoke(korean_word)
print(f'Synonyms for {korean_word["word"]}: {korean_result}')

"""
Synonyms for happy: joyful, content, delighted
Synonyms for 행복: 즐거, 기�, 유
"""
