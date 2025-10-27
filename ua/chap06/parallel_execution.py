from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
import time
import random

class DashboardState(BaseModel):
    user_location: str = "Seoul"
    weather_data: Dict[str, Any] = {}
    news_data: Dict[str, Any] = {} 
    stock_data: Dict[str, Any] = {}
    dashboard_report: str = ""
    start_time: float =0.0

def coordinator(state: DashboardState) -> Dict[str, Any]:
    print(f"대시보드 생성 시작 - 위치 : {state.user_location}")
    return {"start_time": time.time()}

def weather_checker(state: DashboardState) -> Dict[str, Any]:
    print("날씨 데이터 수집 중...")
    time.sleep(random.uniform(1.0, 2.0))  # 시뮬레이션 지연
    weather_info={
        "location": state.user_location,
        "condition": "맑음",
        "temperature": 22,
        "humidity": 65
    }
    print(f"날씨: {weather_info['condition']}, 온도: {weather_info['temperature']}°C, 습도: {weather_info['humidity']}%")
    return {"weather_data": weather_info}

def new_fetcher(state: DashboardState) -> Dict[str, Any]:
    print("뉴스 데이터 수집 중...")
    time.sleep(random.uniform(1.0, 2.0))  # 시뮬레이션 지연
    news_info={
        "articles": [
            {"title": "경제 성장률 상승", "summary": "최근 경제 성장률이 예상보다 높게 나타났습니다."},
            {"title": "기술 혁신 가속화", "summary": "신기술 도입으로 산업 전반에 혁신이 일어나고 있습니다."}
        ],
        "count": 2
    }
        
    
    print(f"뉴스 요약: {news_info['count']}개의 기사 수집 완료.")
    return {"news_data": news_info}

def stock_analyzer(state: DashboardState) -> Dict[str, Any]:
    print("주식 시장 데이터 수집 중...")
    time.sleep(random.uniform(1.0, 2.0))  # 시뮬레이션 지연
    stock_info={
            "KOSPI": 2500.75,
            "NASDAQ": 13000.50
    }
    print(f"주식 시장: KOSPI {stock_info['KOSPI']}, NASDAQ {stock_info['NASDAQ']}")
    return {"stock_data": stock_info}

def aggregator(state: DashboardState) -> Dict[str, Any]:
    print("대시보드 보고서 생성 중...")
    parallel_time = time.time() - state.start_time
    report = f"""
    날씨: {state.weather_data.get('condition')}, 온도: {state.weather_data.get('temperature')}°C, 습도: {state.weather_data.get('humidity')}%
    뉴스: {state.news_data.get('count')}개의 기사 수집 완료.
    주식 시장: KOSPI {state.stock_data.get('KOSPI')}, NASDAQ {state.stock_data.get('NASDAQ')}
    병렬 처리 시간: {parallel_time:.2f}초
    """
    return {"dashboard_report": report}

def create_graph():
    workflow = StateGraph(DashboardState)
    workflow.add_node("coordinator", coordinator)
    workflow.add_node("weather_checker", weather_checker)
    workflow.add_node("news_fetcher", new_fetcher)
    workflow.add_node("stock_analyzer", stock_analyzer)
    workflow.add_node("aggregator", aggregator)

    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "weather_checker")
    workflow.add_edge("coordinator", "news_fetcher")
    workflow.add_edge("coordinator", "stock_analyzer")
    workflow.add_edge("weather_checker", "aggregator")
    workflow.add_edge("news_fetcher", "aggregator")
    workflow.add_edge("stock_analyzer", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile()

def main():
    print("===대시보드 생성 워크플로우 실행===")
    initial = DashboardState(user_location="Seoul")
    app = create_graph()
    result_state = app.invoke(initial)
    print("===대시보드 보고서===")
    print(result_state['dashboard_report'])
if __name__ == "__main__":
    main()