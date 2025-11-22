import os
import sys
import subprocess
from google import genai
from google.genai import types

# 워크스페이스의 ua/googleapi/basic.py 스타일을 참고하여 클라이언트 설정
def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY is missing.")
        sys.exit(1)
    return genai.Client(api_key=api_key)

def get_changed_files():
    """Git을 통해 변경된 파일 목록을 가져옵니다."""
    try:
        # 최근 커밋과 그 이전 커밋 사이의 변경된 파일 목록
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        files = result.stdout.strip().split('\n')
        return [f for f in files if f and os.path.exists(f)]
    except subprocess.CalledProcessError:
        return []

def review_code(client, file_path, content):
    """Python 코드에 주석과 설명을 추가합니다 (로직 변경 금지)."""
    prompt = f"""
    You are an expert Technical Writer and Code Reviewer.
    Analyze the following Python code.
    
    Task:
    1. Add detailed Docstrings (Google Style) to functions and classes.
    2. Add inline comments explaining complex logic or technical details.
    3. DO NOT change any variable names, function logic, or behavior.
    4. ONLY return the full modified code. Do not use markdown code blocks (```).
    
    File: {file_path}
    Code:
    {content}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    # 마크다운 코드 블록 제거 (혹시 포함될 경우)
    cleaned_text = response.text.replace("```python", "").replace("```", "")
    return cleaned_text

def expand_markdown(client, file_path, content):
    """Markdown 파일의 내용을 보강합니다."""
    prompt = f"""
    You are an expert Technical Writer.
    Review the following Markdown content.
    
    Task:
    1. If the content is brief, expand on the concepts with technical depth.
    2. If there are missing explanations for technical terms, add them.
    3. Keep the original content, but append or insert necessary improvements.
    4. ONLY return the full modified markdown content. Do not use markdown code blocks (```).

    File: {file_path}
    Content:
    {content}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    cleaned_text = response.text.replace("```markdown", "").replace("```", "")
    return cleaned_text

def main():
    client = get_gemini_client()
    changed_files = get_changed_files()
    
    if not changed_files:
        print("No changed files found.")
        return

    for file_path in changed_files:
        print(f"Processing: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = None
        
        if file_path.endswith('.py'):
            new_content = review_code(client, file_path, content)
        elif file_path.endswith('.md'):
            new_content = expand_markdown(client, file_path, content)
            
        if new_content and new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Updated: {file_path}")

if __name__ == "__main__":
    main()