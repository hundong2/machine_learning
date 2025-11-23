import re

def clean_html(html_text:str) -> str:
    """HTML tag remove"""
    if not html_text:
        return ""
    clean_text = re.sub("<.*?>", "", html_text)
    #연속된 공백(스페이스, 탭, 줄바꿈)을 하나의 공백으로 정리
    clean_text = re.sub("\s+", " ", clean_text).strip()
    return clean_text

def truncate_text(text: str, max_length: int = 500) -> str:
    """text truncate"""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_date(date_string: str) -> str:
    """date format cleaning"""
    if not date_string:
        return "날짜 정보 없음"
    
    try:
        if "GMT" in date_string:
            date_string = date_string.split("GMT")[0].strip()
        return date_string
    except Exception as ex:
        return date_string

def convert_gmt_to_kst(gmt_time_str: str) -> str:
    """GMT convert to KST"""
    KST_OFFSET_HOURS = 9
    gmt_time = datetime.strptime(gmt_time_str, "%a, %d %b %Y %H:%M:%S GMT")
    kst_time = gmt_time + timedelta(hours=KST_OFFSET_HOURS)
    return kst_time.strftime("%Y-%m-%d %H:%M:%S")

