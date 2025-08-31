from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
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
    client = get_gemini_client()
    response=client.models.generate_content(
        model="gemini-2.5-flash", #gemini-2.5 pro is default thinking model so thinking_budget did't set.
        contents="Explain how AI works in a few words",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0) #Disables thinking
        )
    )
    print(response.text)

def multi_modal_example():
    client = get_gemini_client()
    current_working_directory = os.getcwd()
    print(f"현재 작업 디렉토리: {current_working_directory}")

    image = Image.open("./ua/googleapi/orange.png")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, "Tell me about this instrument"]
    )
    print(response.text)
    #The image you provided shows **oranges**, which are a type of citrus fruit. 
    # There is no instrument depicted in the image.

def get_gemini_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def main():
    #basic_api_test()
    #thinking_setting()
    multi_modal_example()

if __name__=='__main__':
    main()
