#RSS 수집 에이전트 
import json
import re
import asyncio
from typing import Optional

import httpx
import feedparser
import trafilatura
from bs4 import BeautifulSoup
from utils import convert_gmt_to_kst
from state import NewsState

#구글 뉴스 API 관련 상수 정의 
GOOGLE_NEWS_BASE_URL = "https://news.google.com"
GOOGLE_NEWS_API_URL = f"{GOOGLE_NEWS_BASE_URL}/_/DotsSplashUi/data/batchexecute"
KOREA_PARAMS ="&hl=ko&gl=KR&ceid=KR:ko"

class RSSCollectorAgent:
    """RSS 피드를 수집하는 에이전트"""
    def __init__(self):
        self.name = "RSS Collector"
        self.rss_url = f"{GOOGLE_NEWS_BASE_URL}/rss?{KOREA_PARAMS[1:]}"
        self.feed = None
    def load_feed(self) -> None:
        """RSS 피드를 로드합니다."""
        self.feed = feedparser.parse(self.rss_url)
        # feedparser는 RSS/Atom 피드를 파싱하는 파이썬 라이브러리 
        #XML 형식의 RSS 피드를 파싱하여 파이선 객체로 변환해줍니다.  feedparser 가 없다면 httpx를 사용하여 XML을 가져오고 파싱하는 과정이 필요
    @staticmethod
    def extract_chosun_content(html_content):
        """Josun Ilbo 뉴스 콘텐츠 추출"""
        pattern = r"Fusion\.globalContent\s*=\s*({.*?});"
        match = re.search(pattern, html_content, re.DOTALL)

        if match:
            try:
                content_data = json.loads(match.group(1))
                texts = []
                if "content_elements" in content_data:
                    for element in content_data["content_elements"]:
                        if element.get("type") == "text" and "content" in element:
                            texts.append(element.get("content"))
                return "\n".join(texts)
            except json.JSONDecodeError:
                return ""
    async def extract_article_url(self, google_news_url: str) -> Optional[str]:
        """Stack Overflow 뉴스 기사 URL 추출
        https://stackoverflow.com/questions/7938897/how-to-scrape-google-rssfeed-links/79388987#79388987
        google news는 JavaScript를 사용하여 페이지를 리다이렉션시키므로 내부 API를 직접호출하여 우회"""
        async with httpx.AsyncClient() as client:
            try:
                #c-wiz component data extract from google news page 
                response = await client.get(google_news_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                #data extract from c-wiz tag
                data_element = soup.select_one("c-wiz[data-p]") #c-wiz 구글이 사용하는 웹 컴포넌트 
                if not data_element:
                    return None
                raw_data = data_element.get("data-p")
                json_data = json.loads(raw_data.replace("%.@.", '["garturlreq",'))
                payload = {
                    "f.req": json.dumps(
                        [
                            [
                                [
                                    "Fbv4je",
                                    json.dumps(json_data[:-6] + json_data[-2:]),
                                    "null",
                                    "generic"
                                ]
                            ]
                        ]
                    )
                }
                headers = {
                    "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }

                # Google News internal API 호출
                api_response = await client.post(
                    GOOGLE_NEWS_API_URL,
                    headers=headers,
                    data=payload
                )
                cleaned_response = api_response.text.replace(")]}'","")
                response_data = json.loads(cleaned_response)
                article_url = json.loads(response_data[0][2])[1]
                return article_url
            except Exception:
                return None
    async def parse_entry(self, entry) -> dict[str, Optional[str]]:
        """RSS 피드 항목을 파싱합니다."""
        google_news_url = entry.link + KOREA_PARAMS
        original_url = await self.extract_article_url(google_news_url)
        content = ""

        if original_url:
            #trafilatura를 사용하여 웹페이지 다운로드 
            downloaded = trafilatura.fetch_url(original_url)
            if downloaded:
                if "chosun.com" in original_url:
                    content = self.extract_chosun_content(downloaded)
                else:
                    #trafilatura로 일반 기사 추출(한국어 최적화) 
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,#댓글 제외
                        include_images=False,#이미지 제외
                        include_links=False,#링크 제외
                        target_language="ko"#한국어
                    )
        return {
            "title": entry.title,
            "published_kst": convert_gmt_to_kst(entry.published),
            "source": entry.source.get("title", "Unknown"),
            "google_news_url": google_news_url,
            "original_url": original_url,
            "content": content
        }
    async def collect_rss(self, state: NewsState) -> NewsState:
        """RSS 피드를 수집하고 상태를 업데이트합니다."""
        try:
            if not self.feed:
                self.load_feed()

                tasks = [self.parse_entry(entry) for entry in self.feed.entries]
                raw_news = await asyncio.gather(*tasks) #tasks리스트들을 풀어서 asyncio.gather(task1, task2, ...)형태로 전달
                #asyncio.gather는 여러 비동기 작업을 병렬로 실행하고 결과를 모읍니다.
                state.raw_news = raw_news
                print(f"총 {len(raw_news)}개의 뉴스를 수집했습니다.")
        except Exception as e:
            print(f"RSS 수집 중 오류 발생: {e}")
            state.error_log.append(f"RSS 수집 오류: {e}")
        return state
        


