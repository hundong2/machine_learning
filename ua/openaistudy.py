from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

openai_api_key = os.getenv("GOOGLE_API_KEY")
client = OpenAI(
    api_key=openai_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )

def get_openai_completion(prompt, model_name="gemini-2.5-pro"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": "You are a helpful assistant."},
            { "role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__=='__main__':
    user_prompt = input("Enter your prompt: ")
    response = get_openai_completion(user_prompt)
    print("ChatGPT response:", response)

