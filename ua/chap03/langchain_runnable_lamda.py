from langchain_core.runnables import RunnableLambda

def add_exclamation(text: str) -> str:
    """Adds an exclamation mark to the end of the text."""
    return text + "!"

# RunnableLambda로 감싸서 Runnable로 만들기
exclamation_runnable = RunnableLambda(add_exclamation) #일반 파이썬 함수를 랭체인의 Runnable 인터페이스로 변환. 

result = exclamation_runnable.invoke("안녕하세요")
print(result)

results = exclamation_runnable.batch(["안녕", "반가워", "좋은아침"]) #여러 입력을 한번에 처리할수 있음. 
print(results)

"""
안녕하세요!
['안녕!', '반가워!', '좋은아침!']
"""

