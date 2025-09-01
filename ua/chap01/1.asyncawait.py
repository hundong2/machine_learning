import asyncio
import os
from ua.googleapi.basic import get_gemini_client
from openai import AsyncOpenAI


async def get_local_llm(prompt, model_name="model-identifier"):
    openai_api_key = os.getenv("LMS_API_KEY")
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url="http://192.168.45.167:50505/v1"
                    )
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            { "role": "system", "content": "You are a helpful assistant."},
            { "role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


async def call_async_gemini(prompt: str, model: str = "gemini-2.5-pro") -> str:
    client = get_gemini_client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt
    )
    print(f'{response.text}')
    return response.text

async def main():
    print("Starting async calls...")
    response_first = call_async_gemini("Hello, how can I help you today?")
    print(f"Response: {response_first}")
    response_second = get_local_llm("What's the weather like today?")
    print(f"Response: {response_second}")
    response_first_msg, response_second_msg = await asyncio.gather(response_first, response_second)
    print(f"Response 1: {response_first_msg}")
    print(f"Response 2: {response_second_msg}")

if __name__ == "__main__":
    asyncio.run(main())