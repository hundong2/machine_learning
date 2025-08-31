import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일에서 환경 변수 로드
load_dotenv()

# Gemini API 키 설정
# GOOGLE_API_KEY 환경 변수에서 API 키를 가져옵니다.
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

genai.configure(api_key=api_key)

def get_gemini_completion(prompt, model_name="gemini-2.5-pro"):
    """
    Gemini 모델을 사용하여 프롬프트에 대한 응답을 생성합니다.

    Args:
        prompt (str): 모델에 전달할 프롬프트.
        model_name (str): 사용할 Gemini 모델의 이름.

    Returns:
        str: 모델이 생성한 텍스트 응답.
    """
    try:
        # 사용할 모델 선택
        model = genai.GenerativeModel(model_name)
        
        # 텍스트 생성 요청
        response = model.generate_content(prompt)
        
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # 사용자로부터 프롬프트 입력받기
    user_prompt = input("Gemini에게 무엇이든 물어보세요: ")
    
    # Gemini로부터 응답 받기
    completion = get_gemini_completion(user_prompt)
    
    # 결과 출력
    print("\n[Gemini의 답변]")
    print(completion)
    