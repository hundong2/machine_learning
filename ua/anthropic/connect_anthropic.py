import anthropic
import os 
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
model = 'claude-3-haiku-20240307'
conversation=[]

conversation.append({"role":"user", "content": "너는 누구야?"})

response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=conversation
)

assistant_message = response.content[0]['text']
print(assistant_message)
conversation.append({"role":"assistant", "content": assistant_message})
conversation.append({"role":"user", "content": "2020년 월드 시리즈 우승팀이 어디야?"})
response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=conversation
)
print(response.content[0].text)