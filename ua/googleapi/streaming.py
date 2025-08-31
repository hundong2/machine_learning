from basic import get_gemini_client
from dotenv import load_dotenv

load_dotenv()

def streaming_ex():
    client = get_gemini_client()
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=['Expalain how AI works']
    )
    for chunk in response:
        print(f'{chunk.text}')
def main():
    streaming_ex()

if __name__=="__main__":
    main()