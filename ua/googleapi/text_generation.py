from basic import get_gemini_client
from google import genai
from google.genai import types

def example_code_1():
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, how can I help you today?"
    )
    print(f'{response.text}')
def example_code_2():
    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are a cat. Your name is Neko."),
        contents="Hello there"
    )
    print(f'example code 2:{response.text}')

def example_code_3():
    client = get_gemini_client()
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        temperature=0.1 # model의 무작위성을 설정 [ 0 ~ 2 의 값을 가짐]
    )
    )
    print(f'example code 3:{response.text}')
def main():
    print("text generation!")
#    example_code_1()
#    example_code_2()
    example_code_3()

if __name__=="__main__":
    main()