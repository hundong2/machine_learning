from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

client = OpenAI(
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)

def chatbot_response(user_message: str):
    result = client.responses.create(model='model-identifier', 
                                     input=user_message)
    return result

if __name__=='__main__':
    while True:
        try:
            user_message=input('Message: ')
            if user_message.lower() == "exit":
                print("exit!")
                break
            response = chatbot_response(user_message)
            print(f"Response: {response}")

            result = chatbot_response(user_message)
            print(f'chatbot: {result.output_text}')
            '''   
            I'm Qwen, the large language model developed by Tongyi Lab! 😊 How can I assist you today?
'''
        except KeyboardInterrupt as e:
            print("exit!")
            break