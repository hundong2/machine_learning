import os
import sys
import subprocess
from google import genai
from google.genai import types

def get_gemini_client():
    """Gemini API 클라이언트를 검색하고 구성합니다.

    이 함수는 환경 변수에서 API 키를 검색하고,
    Gemini 클라이언트를 구성하고 반환합니다. API 키를 찾을 수 없으면
    오류 메시지를 인쇄하고 프로그램을 종료합니다.

    반환:
        genai.Client: 구성된 Gemini API 클라이언트.

    발생:
        SystemExit: GOOGLE_API_KEY 환경 변수가 설정되지 않은 경우.
    """
    api_key = os.getenv("GOOGLE_API_KEY") # 환경 변수에서 API 키를 가져옵니다.
    if not api_key:
        print("Error: GOOGLE_API_KEY is missing.")
        sys.exit(1) # API 키가 없으면 오류 메시지를 출력하고 종료합니다.
    return genai.Client(api_key=api_key) # Gemini API 클라이언트를 설정하고 반환합니다.

def get_changed_files():
    """Git을 사용하여 마지막 두 커밋 간에 변경된 파일 목록을 검색합니다.

    이 함수는 `git diff` 명령을 사용하여 수정된 파일을 식별합니다.
    `HEAD~1`(이전 커밋)과 `HEAD`(현재 커밋) 사이.
    파일 시스템에 현재 존재하는 파일만 포함하도록 목록을 필터링합니다.

    반환:
        list[str]: 변경된 파일 경로 목록 또는 오류가 발생하거나 변경 사항이 없는 경우 빈 목록입니다.
    """
    try:
        # 최근 커밋과 그 이전 커밋 사이의 변경된 파일 목록을 가져오는 Git 명령
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        # subprocess를 사용하여 Git 명령을 실행하고 결과를 캡처합니다.  `check=True`는 오류 발생 시 예외를 발생시킵니다.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n') # 결과를 줄바꿈으로 분할하여 파일 목록을 생성합니다.
        return [f for f in files if f and os.path.exists(f)] # 비어 있지 않고 존재하는 파일만 필터링합니다.
    except subprocess.CalledProcessError:
        return [] # Git 명령 실행 중 오류가 발생하면 빈 목록을 반환합니다.

def review_code(client, file_path, content):
    """Gemini API를 사용하여 Python 코드를 검토하고, 독스트링과 인라인 주석을 추가합니다.

    이 함수는 파일 경로와 내용을 포함하는 프롬프트를 Gemini에 보냅니다.
    API는 자세한 독스트링과 인라인 주석을 추가하도록 지시합니다. 논리
    및 코드의 동작은 변경되지 않은 상태로 유지되어야 합니다. Gemini의 응답
    API는 마크다운 코드 블록을 제거하여 정리됩니다.

    Args:
        client (genai.Client): 구성된 Gemini API 클라이언트.
        file_path (str): 검토 중인 Python 파일의 경로.
        content (str): Python 파일의 내용.

    반환:
        str: 마크다운 코드 블록이 제거된 독스트링과 주석이 추가된 검토된 코드.
    """
    prompt = f"""
    당신은 전문 기술 작가이자 코드 리뷰어입니다.
    다음 Python 코드를 분석해주세요.
    
    작업:
    1. 함수와 클래스에 상세한 독스트링(Google 스타일)을 추가하세요.
    2. 복잡한 로직이나 기술적 세부사항을 설명하는 인라인 주석을 추가하세요.
    3. 변수명, 함수 로직, 동작을 변경하지 마세요.
    4. 수정된 전체 코드만 반환하세요. 마크다운 코드 블록(```)은 사용하지 마세요.
    
    파일: {file_path}
    코드:
    {content}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ) # Gemini API를 호출하여 코드 리뷰를 수행합니다.
    # 마크다운 코드 블록 제거 (혹시 포함될 경우).  생성된 텍스트에서 ""과 ""를 제거합니다.
    cleaned_text = response.text.replace("", "").replace("", "")
    return cleaned_text

def expand_markdown(client, file_path, content):
    """Gemini API를 사용하여 Markdown 파일의 내용을 확장합니다.

    이 함수는 파일 경로와 내용을 포함하는 프롬프트를 Gemini API에 전송하여,
    기술적 깊이를 가지고 개념을 확장하고 누락된 기술 용어에 대한 설명을 추가하도록 지시합니다.
    원본 내용은 유지되며, 필요에 따라 개선 사항을 추가하거나 삽입합니다.
    Gemini API의 응답은 마크다운 코드 블록을 제거하여 정리됩니다.

    Args:
        client (genai.Client): 설정된 Gemini API 클라이언트.
        file_path (str): 확장할 Markdown 파일의 경로.
        content (str): Markdown 파일의 내용.

    반환:
        str: 마크다운 코드 블록이 제거된 확장된 Markdown 내용.
    """
    prompt = f"""
    당신은 전문 기술 작가입니다.
    다음 Markdown 내용을 검토해주세요.
    
    작업:
    1. 내용이 간단하다면, 기술적 깊이를 가지고 개념을 확장하세요.
    2. 기술 용어에 대한 설명이 누락된 경우 추가하세요.
    3. 원본 내용은 유지하되, 필요한 개선사항을 추가하거나 삽입하세요.
    4. 수정된 전체 마크다운 내용만 반환하세요. 마크다운 코드 블록(```)은 사용하지 마세요.

    파일: {file_path}
    내용:
    {content}
    """
    
    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt
    ) # Gemini API를 호출하여 Markdown 콘텐츠를 확장합니다.
    cleaned_text = response.text.replace("markdown", "").replace("", "") # 생성된 텍스트에서 "markdown"과 ""를 제거합니다.
    return cleaned_text

def main():
    """변경된 파일을 처리하고 검토된 내용으로 업데이트하는 기본 함수입니다."""
    client = get_gemini_client() # Gemini API 클라이언트를 가져옵니다.
    changed_files = get_changed_files() # 변경된 파일 목록을 가져옵니다.
    
    if not changed_files:
        print("No changed files found.")
        return # 변경된 파일이 없으면 메시지를 출력하고 종료합니다.

    for file_path in changed_files: # 변경된 각 파일에 대해 반복합니다.
        print(f"Processing: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f: # 파일을 읽기 모드로 엽니다.
            content = f.read() # 파일 내용을 읽습니다.

        new_content = None
        if file_path in ['.github/scripts/code_review.py']:
            continue # 이 스크립트와 특정 파일은 건너뜁니다.
        if file_path.endswith('.py'):
            new_content = review_code(client, file_path, content) # Python 파일인 경우 코드 리뷰를 수행합니다.
        elif file_path.endswith('.md'):
            new_content = expand_markdown(client, file_path, content) # Markdown 파일인 경우 콘텐츠 확장을 수행합니다.
            
        if new_content and new_content != content: # 새 콘텐츠가 있고 이전 콘텐츠와 다른 경우
            with open(file_path, 'w', encoding='utf-8') as f: # 파일을 쓰기 모드로 엽니다.
                f.write(new_content) # 새 콘텐츠를 파일에 씁니다.
            print(f"Updated: {file_path}") # 업데이트된 파일 경로를 출력합니다.

if __name__ == "__main__":
    main() # 스크립트가 직접 실행되면 main 함수를 호출합니다.