```python
import re
from datetime import datetime, timedelta

def clean_html(html_text:str) -> str:
    """HTML 태그를 제거합니다.

    HTML 텍스트에서 모든 HTML 태그를 제거하고 정리된 텍스트를 반환합니다.

    Args:
        html_text: 정리할 HTML 텍스트입니다.

    Returns:
        HTML 태그가 제거된 정리된 텍스트입니다. 빈 문자열이 입력되면 빈 문자열을 반환합니다.

    Raises:
        None
    """
    if not html_text:
        return ""
    clean_text = re.sub("<.*?>", "", html_text) # HTML 태그 제거 (<로 시작해서 >로 끝나는 모든 문자열)
    # 연속된 공백(스페이스, 탭, 줄바꿈)을 하나의 공백으로 정리
    clean_text = re.sub("\s+", " ", clean_text).strip() # \s+는 하나 이상의 공백 문자를 의미
    return clean_text

def truncate_text(text: str, max_length: int = 500) -> str:
    """텍스트를 지정된 최대 길이로 자릅니다.

    텍스트가 최대 길이보다 길면 텍스트를 자르고 "..."을 추가합니다.

    Args:
        text: 자를 텍스트입니다.
        max_length: 텍스트의 최대 길이입니다 (기본값: 500).

    Returns:
        최대 길이로 잘린 텍스트입니다. 텍스트가 비어 있거나 최대 길이보다 짧으면 원래 텍스트를 반환합니다.

    Raises:
        None
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..." # 텍스트를 자르고 "..."을 추가

def format_date(date_string: str) -> str:
    """날짜 문자열의 형식을 정리합니다.

    날짜 문자열에서 "GMT" 부분을 제거하고 정리된 날짜 문자열을 반환합니다.

    Args:
        date_string: 형식을 정리할 날짜 문자열입니다.

    Returns:
        정리된 날짜 문자열입니다. 입력이 비어 있으면 "날짜 정보 없음"을 반환합니다.
        처리 중 오류가 발생하면 원래 날짜 문자열을 반환합니다.

    Raises:
        None
    """
    if not date_string:
        return "날짜 정보 없음"
    
    try:
        if "GMT" in date_string:
            date_string = date_string.split("GMT")[0].strip() # "GMT"를 기준으로 문자열을 분할하고 첫 번째 부분만 사용
        return date_string
    except Exception as ex:
        return date_string

def convert_gmt_to_kst(gmt_time_str: str) -> str:
    """GMT 시간을 KST 시간으로 변환합니다.

    GMT 형식의 시간 문자열을 KST 형식의 시간 문자열로 변환합니다.

    Args:
        gmt_time_str: 변환할 GMT 시간 문자열입니다 (예: "Sat, 20 Jan 2024 10:00:00 GMT").

    Returns:
        KST 형식의 시간 문자열입니다 (예: "2024-01-20 19:00:00").

    Raises:
        ValueError: GMT 시간 문자열의 형식이 올바르지 않은 경우 발생합니다.
    """
    KST_OFFSET_HOURS = 9 # KST는 GMT보다 9시간 빠름
    gmt_time = datetime.strptime(gmt_time_str, "%a, %d %b %Y %H:%M:%S GMT") # GMT 시간 문자열을 datetime 객체로 변환
    # datetime.strptime(): 문자열을 datetime 객체로 변환하는 함수
    # %a: 요일 (Sun, Mon, ...)
    # %d: 일 (01, 02, ...)
    # %b: 월 (Jan, Feb, ...)
    # %Y: 년 (2024)
    # %H: 시 (00, 01, ...)
    # %M: 분 (00, 01, ...)
    # %S: 초 (00, 01, ...)
    kst_time = gmt_time + timedelta(hours=KST_OFFSET_HOURS) # GMT 시간에 9시간을 더하여 KST 시간 계산
    # timedelta(): 날짜와 시간 간의 차이를 나타내는 객체
    return kst_time.strftime("%Y-%m-%d %H:%M:%S") # KST 시간을 지정된 형식의 문자열로 변환
    # strftime(): datetime 객체를 문자열로 변환하는 함수
    # %Y: 년 (2024)
    # %m: 월 (01, 02, ...)
    # %d: 일 (01, 02, ...)
    # %H: 시 (00, 01, ...)
    # %M: 분 (00, 01, ...)
    # %S: 초 (00, 01, ...)
```