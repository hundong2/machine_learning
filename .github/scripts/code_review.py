import os
import sys
import subprocess
from google import genai
from google.genai import types

def get_gemini_client():
    """Retrieves and configures the Gemini API client.

    This function retrieves the API key from the environment variables,
    configures the Gemini client, and returns it.  If the API key is not found,
    it prints an error message and exits the program.

    Returns:
        genai.Client: Configured Gemini API client.

    Raises:
        SystemExit: If the GOOGLE_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY") # 환경 변수에서 API 키를 가져옵니다.
    if not api_key:
        print("Error: GOOGLE_API_KEY is missing.")
        sys.exit(1) # API 키가 없으면 오류 메시지를 출력하고 종료합니다.
    return genai.Client(api_key=api_key) # Gemini API 클라이언트를 설정하고 반환합니다.

def get_changed_files():
    """Retrieves a list of files changed between the last two commits using Git.

    This function uses the `git diff` command to identify files that have been
    modified between the `HEAD~1` (previous commit) and `HEAD` (current commit).
    It filters the list to include only those files that currently exist in the
    file system.

    Returns:
        list[str]: A list of file paths that have been changed, or an empty list if an error occurs or no changes are found.
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
    """Reviews Python code using the Gemini API, adding docstrings and inline comments.

    This function sends a prompt containing the file path and content to the Gemini
    API, instructing it to add detailed docstrings and inline comments.  The logic
    and behavior of the code must remain unchanged.  The response from the Gemini
    API is then cleaned by removing any markdown code blocks.

    Args:
        client (genai.Client): The configured Gemini API client.
        file_path (str): The path to the Python file being reviewed.
        content (str): The content of the Python file.

    Returns:
        str: The reviewed code with added docstrings and comments, with markdown code blocks removed.
    """
    prompt = f"""
    You are an expert Technical Writer and Code Reviewer.
    Analyze the following Python code.
    
    Task:
    1. Add detailed Docstrings (Google Style) to functions and classes.
    2. Add inline comments explaining complex logic or technical details.
    3. DO NOT change any variable names, function logic, or behavior.
    4. ONLY return the full modified code. Do not use markdown code blocks ().
    
    File: {file_path}
    Code:
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
    """Expands the content of a Markdown file using the Gemini API.

    This function sends a prompt containing the file path and content to the Gemini
    API, instructing it to expand on the concepts with technical depth and add
    explanations for missing technical terms.  The original content should be
    preserved, with improvements appended or inserted as necessary.  The response
    from the Gemini API is then cleaned by removing any markdown code blocks.

    Args:
        client (genai.Client): The configured Gemini API client.
        file_path (str): The path to the Markdown file being expanded.
        content (str): The content of the Markdown file.

    Returns:
        str: The expanded Markdown content, with markdown code blocks removed.
    """
    prompt = f"""
    You are an expert Technical Writer.
    Review the following Markdown content.
    
    Task:
    1. If the content is brief, expand on the concepts with technical depth.
    2. If there are missing explanations for technical terms, add them.
    3. Keep the original content, but append or insert necessary improvements.
    4. ONLY return the full modified markdown content. Do not use markdown code blocks ().

    File: {file_path}
    Content:
    {content}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    ) # Gemini API를 호출하여 Markdown 콘텐츠를 확장합니다.
    cleaned_text = response.text.replace("markdown", "").replace("", "") # 생성된 텍스트에서 "markdown"과 ""를 제거합니다.
    return cleaned_text

def main():
    """Main function to process changed files and update them with reviewed content."""
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
    