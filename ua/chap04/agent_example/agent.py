from google.adk.agents import Agent
import os
from dotenv import load_dotenv
load_dotenv()


def greet_user() -> str: #agent tools로 활용 
    return "Hello, User! Welcome to the Agent Example."

root_agent = Agent(
    name="greet_agent",
    description="An agent that greets the user.",
    model="gemini-2.5-flash",
    instruction="Greet the user warmly and provide a friendly message.",#system prompt 
    tools=[greet_user]
)