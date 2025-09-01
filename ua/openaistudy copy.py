from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

openai_api_key = os.getenv("LMS_API_KEY")
client = OpenAI(
    api_key=openai_api_key,
    base_url="http://192.168.45.167:50505/v1"
                )

def get_openai_completion(prompt, model_name="model-identifier"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": "You are a helpful assistant."},
            { "role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__=='__main__':
    user_prompt = 'what is the capital of France?'
    response = get_openai_completion(user_prompt)
    print("ChatGPT response:", response)

