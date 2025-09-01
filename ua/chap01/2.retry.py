import asyncio
import os
import logging
import random
from dotenv import load_dotenv
from ua.googleapi.basic import get_gemini_client
from openai import AsyncOpenAI

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simulate_random_failure():
    if random.random() < 0.5:
        logger.warning("Simulated transient error occurred.")
        raise ConnectionError('connection error for testing')
    await asyncio.sleep(random.uniform(0.1, 0.5))


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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(), #if emptry then all exception handling
    before_sleep=lambda retry_state: logger.warning(f'API call fail: {retry_state.outcome.exception()}, {retry_state.attempt_number} retrying...')
)
async def call_async_gemini(prompt: str, model: str = "gemini-2.5-pro") -> str:
    client = get_gemini_client()
    await simulate_random_failure() # simulate transient failure
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
    try:
        response_first_msg, response_second_msg = await asyncio.gather(response_first, response_second, return_exceptions=False)
        print(f"Response 1: {response_first_msg}")
        print(f"Response 2: {response_second_msg}")     
    except Exception as ex:
        logger.error(f"Error occurred while gathering responses: {ex}")


if __name__ == "__main__":
    asyncio.run(main())