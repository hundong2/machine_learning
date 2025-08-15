from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1단계: 데이터 저장소 만들기 (State)
class MyState(TypedDict):
    message: str

# 2단계: 작업 함수 만들기 (Node)
def say_hello(state):
    return {"message": "Hello, LangGraph!"}

# 3단계: 그래프 만들기
graph = StateGraph(MyState)
graph.add_node("hello", say_hello)
graph.add_edge(START, "hello")
graph.add_edge("hello", END)

# 4단계: 실행하기
app = graph.compile()
result = app.invoke({"message": ""})
print(result)
