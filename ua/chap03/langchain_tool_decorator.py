import random
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def rps() -> str:
    """가위 바위 보 게임을 합니다."""
    return random.choice(["가위", "바위", "보"])

#tools binding llm
llm = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
).bind_tools([rps])
llm_for_chat = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1",
    temperature=0.7
) # 해설용 LLM
print(type(llm))
print(type(llm_for_chat))

def judge(user_choice, computer_choice):
    user_choice = user_choice.strip()
    computer_choice = computer_choice.strip() #strip, 입력의 앞뒤 공백을 제거
    if user_choice == computer_choice:
        return "draw"
    elif (user_choice, computer_choice) in [
        ("가위", "보"),
        ("바위", "가위"),
        ("보", "바위")
    ]:
        return "win"
    else:
        return "lose"
    

print("가위, 바위, 보!(종료: q)")
while(user_input := input("\n가위/바위/보: ")) != "q": #:= 바다코끼리를 사용하여 입력을 받으면서 조건을 수행 
    ai_msg = llm.invoke(
        f"가위바위보 게임: 사용자가{user_input}을 냈습니다. rps tool을 사용하세요. "
    )
    if ai_msg.tool_calls: #tool_calls가 있으면, ai가 호출하려는 tool 의 리스트 
        print(type(rps))
        llm_choice=rps.invoke("")#tool call
        print(f'selected from AI: {llm_choice}')
        result=judge(user_input, llm_choice)
        print(f'승부: {result}')

        final = llm_for_chat.invoke(
            f'가위바위보게임 결과를 재밌게 해설해주세요. {user_input} 대 {llm_choice}의 결과는 {result}입니다.'
        )
        print(final)
        print(f'LLM 해설 : {final.content}')
        print(f'game result: {user_input} vs {llm_choice} : {result}')
    else:
        print('tool call failed')