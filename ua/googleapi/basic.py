from google import genai
from google.genai import types
from dotenv import load_dotenv
import os 
load_dotenv()

def basic_api_test():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="Hello, how can I help you today?"
    )
    print(f'{response.text}')

def thinking_setting():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response=client.models.generate_content(
        model="gemini-2.5-flash", #gemini-2.5 pro is default thinking model so thinking_budget did't set.
        contents="Explain how AI works in a few words",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) #Disables thinking
        )
    )
    print(response.text)

def main():
    #basic_api_test()
    thinking_setting()

if __name__=='__main__':
    main()
