from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

#워크플로우 단계 정의, enum 또는 문자열 사용 가능 
class WorkflowStep:
    GREETING = "GREETING"
    PROCESSING = "PROCESSING"

#graph status 모델 정의
class GraphState(BaseModel):
    name: str = Field(default="", description="사용자 이름")
    greeting: str = Field(default="", description="생성된 인사말")
    processed_message: str = Field(default="", description="처리된 최종 메시지")

# first node function
def generate_greeting(state: GraphState) -> Dict[str, Any]:
    name = state.name or "아무개"
    greeting = f"안녕하세요, {name}님! 만나서 반갑습니다."
    print(f"[generate_greeting] 인사말 생성: {greeting}")
    return {"greeting": greeting}

# second node : 인사말을 처리하고 최종 메세지 생성

def process_message(state: GraphState) -> Dict[str, Any]:
    greeting = state.greeting
    processed_message = f"{greeting} LangGraph에 오신 것을 환영합니다."
    print(f"[process_message] 최종 메세지: {processed_message}")
    return {"processed_message": processed_message}

# graph 생성
def create_hello_graph():
    workflow = StateGraph(GraphState)

    #node 추가
    workflow.add_node(WorkflowStep.GREETING, generate_greeting)
    workflow.add_node(WorkflowStep.PROCESSING, process_message)
    
    #start point setting
    workflow.add_edge(START, WorkflowStep.GREETING)

    #edge 추가 
    workflow.add_edge(WorkflowStep.GREETING, WorkflowStep.PROCESSING)
    workflow.add_edge(WorkflowStep.PROCESSING, END)

    app = workflow.compile() #CompiledStateGraph, Runnable interface를 구현, invoke() 메서드 제공

    return app

def main():
    print("LangGraph Hello World 예제 시작")
    app = create_hello_graph()

    initial_state=GraphState(name="홍길동", greeting="", processed_message="")
    print("초기 상태:", initial_state.model_dump())
    print("start graph execution...")

    final_state = app.invoke(initial_state)

    print("그래프 실행 종료")
    print("final state:", final_state)
    print(f"\n결과 메시지: {final_state['processed_message']}")
    app.get_graph().draw_ascii()
    result=app.get_graph().draw_mermaid_png()
    with open("./hello_langgraph_.png","wb") as f:
        f.write(result)

if __name__ == "__main__":
    main()