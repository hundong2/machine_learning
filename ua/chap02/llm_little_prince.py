from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

LITTLE_PRINCE_PERSONA="""
당신은 생택쥐페리의 '어린왕자'입니다. 다음 특성을 따라주세요:
1. 순수한 관점으로 세상을 바라봅니다.
2. "어째서?"라는 질문을 자주하며 호기심이 많습니다.
3. 철학적 통찰을 단순하게 표현합니다.
4. "어른들은 참이상해요"라는 표현을 씁니다.
5. B-612 소행성에서 왔으며 장미와의 관계를 언급합니다.
6. 여우의 "길들임"과 "책임"에 대한 교훈을 중요시 합니다.
7. "중요한 것은 눈에 보이지 않아"라는 문장을 사용합니다.
8. 공손하고 친절한 말투를 사용합니다.
9. 비유와 은유로 복잡한 개념을 설명합니다.
항상 간결하게 답변하세요. 길어야 두세 문장으로 응답하고, 어린 왕자의 순수함과 지혜를 담아내세요. 
복잡한 주제도 본질적으로 단순화하여 설명하세요. 
"""
client = OpenAI(
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1" #using qwen model
)

def chatbot_response(user_message: str, previous_response_id: str = None):
    result = client.responses.create(model='model-identifier',
                                     reasoning={"effort": "low"},
                                     instructions=LITTLE_PRINCE_PERSONA,
                                     input=user_message,
                                     previous_response_id=previous_response_id
                                     )
    return result

if __name__=='__main__':
    while True:
        try:
            previous_response_id = None
            user_message=input('Message: ')
            if user_message.lower() == "exit":
                print("exit!")
                break
            response = chatbot_response(user_message, previous_response_id=previous_response_id)
            previous_response_id = response.id
            print(f'chatbot: {response.output_text}')
            '''   
            I'm Qwen, the large language model developed by Tongyi Lab! 😊 How can I assist you today?
'''
        except KeyboardInterrupt as e:
            print("exit!")
            break