from openai import OpenAI
from dotenv import load_dotenv
import os 
import rich

load_dotenv()

openai_api_key = os.getenv("GOOGLE_API_KEY")
client = OpenAI(
    api_key=openai_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

def stream_chat_completion(prompt, model_name="gemini-2.5-pro"):
    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": "You are a helpful assistant."},
            { "role": "user", "content": prompt}
        ],
        stream=True
    )
    for chunk in stream:
        content=chunk.choices[0].delta.get.content
        if content is not None:
            print(content, end='')

def stream_reponse(prompt, model="gemini-2.5-pro"):
    with client.reponses.stream(model=model, input=prompt) as stream:
        for event in stream:
            if "output_text" in event.type:
                rich.print(event)

if __name__=='__main__':
    user_prompt = input("Enter your prompt: ")
    stream_chat_completion(user_prompt)
    stream_reponse(user_prompt)

