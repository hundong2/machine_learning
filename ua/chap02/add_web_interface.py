from openai import OpenAI
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1" #using qwen model
)

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
messages = []
previous_response_id = None
def chatbot_response(user_message: str, previous_response_id: str = None):
    result = client.responses.create(model='model-identifier',
                                     reasoning={"effort": "low"},
                                     instructions=LITTLE_PRINCE_PERSONA,
                                     input=user_message,
                                     previous_response_id=previous_response_id
                                     )
    return result

@app.get("/", response_class=HTMLResponse)
async def read_root():
    chat_history=""
    for msg in messages:
        if msg['role'] == 'user':
            chat_history += f'<p><b>you:</b>{msg["content"]}</p>'
        else:
            chat_history += f'<p><b>chatbot:</b>{msg["content"]}</p>'
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1> chatbot</h1>
        <div>
            {chat_history}
        </div>
        <form action="/chat" method="post">
            <input type="text" name="message" placeholder="Type your message here..." required>
            <button type="submit">Send</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
    
@app.post("/chat", response_class=HTMLResponse)
# async def chat(message: str = Form(...)):
# Form(...) 설명:
# FastAPI에서 사용하는 기능으로, 웹 페이지의 HTML <form> 태그로부터 데이터를 받아올 때 사용됩니다.
# 사용자가 웹페이지의 입력창에 메시지를 입력하고 'Send' 버튼을 누르면, 이 데이터는 HTTP POST 요청의 'Form 데이터' 형태로 서버에 전송됩니다.
# 여기서 message: str = Form(...)는 클라이언트가 보낸 Form 데이터 중에서 'name' 속성이 'message'인 값을 찾아서
# 'message'라는 파라미터에 담아달라고 FastAPI에게 알려주는 역할을 합니다.
# '...' (Ellipsis 라고 부릅니다)는 이 값이 반드시 필요하다는(required) 의미입니다. 
# 만약 'message' 값이 비어있으면 FastAPI는 자동으로 오류를 발생시킵니다.
async def chat(message: str = Form(...)):
    # global 키워드 설명:
    # 파이썬에서 함수 안에서 변수를 수정하면, 기본적으로 그 함수 안에서만 사용되는 '지역 변수'가 새로 만들어집니다.
    # 하지만 'global' 키워드를 사용하면 "이 함수 안에서 수정하려는 변수는 함수 밖에 있는 전역 변수(global variable)입니다"라고 알려주는 역할을 합니다.
    # 즉, 이 키워드를 통해 함수가 실행될 때마다 'previous_response_id'와 'messages' 변수의 값이 초기화되지 않고,
    # 이전의 값을 계속 유지하고 업데이트할 수 있게 됩니다.
    # 여기서는 채팅 기록(messages)과 이전 대화 ID(previous_response_id)를 계속해서 저장하고 관리하기 위해 사용합니다.
    global previous_response_id, messages
    messages.append({'role': 'user', 'content': message})
    result = chatbot_response(message, previous_response_id=previous_response_id)
    previous_response_id = result.id

    messages.append({'role': 'little_prince', 'content': result.output_text})
    return await read_root()

if __name__=="__main__":
    uvicorn.run(
        "add_web_interface:app", host="127.0.0.1", port=8000, reload=True
    )