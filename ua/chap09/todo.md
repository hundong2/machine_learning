# 멀티에이전트 뉴스 수집 및 요약 시스템 

1. `__start__`. 
2. `collect`. 
3. `summarize`. 
4. `organize`.
5. `report`. 
6. `__end__`. 

## project install for dependency

`pip install feedparser trafilatura py3langid beautifulsoup4` 

- `feedparser` : RSS나 Atom 같은 뉴스 피드 데이터를 읽어와 파이썬에서 다루기 쉽게 변환해 주는 도구입니다.  
- `trafilatura` : 웹페이지의 복잡한 HTML에서 광고나 메뉴를 제외하고, 순수한 본문 텍스트만 깔끔하게 추출하는 스크래핑 도구입니다.  
- `py3langid` : 추출된 텍스트가 한국어인지 영어인지 언어를 자동으로 판별해 주는 라이브러리입니다.  
- `beautifulsoup4` : HTML 코드를 분석하여 특정 태그나 데이터를 찾고 추출할 때 사용하는 가장 대중적인 파싱 라이브러리입니다.  
