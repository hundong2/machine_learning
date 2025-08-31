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
    response = client.responses.create(
        model=model_name,
        tools=[{"type": "web_search_preview"}],
        input=prompt
    )
    return response.output_text

if __name__=='__main__':
    user_prompt = """
    https://platform.openai.com/docs/api-reference/responses/create
    를 읽어서 리스폰스 API에 대해서 설명해줘. 
    """
    response = get_openai_completion(user_prompt)
    print("ChatGPT response:", response)

