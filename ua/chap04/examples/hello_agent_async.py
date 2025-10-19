import os 
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
from agents import Agent, function_tool, WebSearchTool, FileSearchTool, set_default_openai_key, Runner 
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

set_default_openai_key(openai_api_key)


hello_agent = Agent(
    name = "HelloAgent",
    instructions = "you are say hello agent, your job is say hello to me"
)

result= Runner.run_sync(hello_agent, "영어로 인사해주세요.")
print(result.final_output)