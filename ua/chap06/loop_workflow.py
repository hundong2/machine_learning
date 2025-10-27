from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import random

class GuessGameState(BaseModel):
    target_number: int = Field(default=0, description="맞춰야할 숫자")
    user_guess: int = Field(default=0, description="사용자 추측")
    attempts: int = Field(default=0, description="시도 횟수")
    max_attempts: int = Field(default=5, description="최대 시도 횟수")
    game_status: str = Field(default="playing", description="게임 상태")
    response: str = Field(default="", description="응답 메시지")

def game_setup(state: GuessGameState) -> Dict[str, Any]:
    target = random.randint(1, 50)
    print("game start!")

    return {
        "target_number": target,
        "game_status": "playing",
        "response": f"1~50 사이의 숫자를 맞춰 보세요! ( 최대 {state.max_attempts}회)",
        "attempts": 0
    }

def user_guess(state: GuessGameState) -> Dict[str, Any]:
    guess = input("입력: ")
    print(f"[{state.attempts + 1 }번째 시도] 추측: {guess}")
    return { "user_guess": int(guess), "attempts": state.attempts + 1}

def check_guess(state: GuessGameState) -> Dict[str, Any]:
    target = state.target_number
    guess = state.user_guess
    attempts = state.attempts

    print(f"[check_guess] {guess} ( 시도: {attempts}회)")

    if guess == target:
        print("정답!")
        return {
            "game_status": "won",
            "response": f"축하합니다! {attempts}회 만에 정답 {target}을 맞추셨습니다!"
        }
    elif attempts >= state.max_attempts:
        print("시도 횟수 초과")
        return {
            "game_status": "lost",
            "response": f"아쉽네요! 최대 시도 횟수를 초과했습니다. 정답은 {target}였습니다."
        }
    else:
        hint = "더 큰수 " if guess < target else "더 작은 수"
        remaining = state.max_attempts - attempts
        return {
            "game_status": "playing",
            "response" : f"힌트: {hint} 남은 시도 횟수: {remaining}회"
        }
    
def route_game(state: GuessGameState) -> Literal["continue", "end"]:
    print(f"[route_game] 현재 상태: {state.game_status}, 시도 횟수: {state.attempts}")
    if state.game_status == "playing":
        return "continue"
    else:
        return "end"
    
def create_guess_game_graph():
    workflow = StateGraph(GuessGameState)
    workflow.add_node("setup", game_setup)
    workflow.add_node("guess", user_guess)
    workflow.add_node("check", check_guess)
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "guess")
    workflow.add_edge("guess", "check")
    workflow.add_conditional_edges(
        "check",
        route_game,
        {
            "continue": "guess",
            "end": END 
        },
    )
    return workflow.compile()

def main():
    app = create_guess_game_graph()
    initial_state = GuessGameState(max_attempts=5)
    result = app.invoke(initial_state)
    print(f"최종 결과: {result['response']}")
    print(f"게임 상태: {result['game_status']}")
    print(f"총 시도: {result['attempts']}회")

if __name__ == "__main__":
    main()