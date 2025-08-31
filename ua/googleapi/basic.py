from google import genai
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

def main():
    basic_api_test()

if __name__=='__main__':
    main()
