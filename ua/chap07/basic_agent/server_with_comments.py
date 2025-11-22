"""
A2A(Agent to Agent) 프로토콜 기반 에이전트 서버 구현

핵심 개념(초보자용):
- A2A: 에이전트끼리 표준 프로토콜로 통신하도록 정의한 명세
- AgentCard: 에이전트의 정보(이름, 설명, 제공 기능)를 표준 JSON으로 나타낸 것
- AgentSkill: AgentCard에 담기는 "에이전트가 할 수 있는 작업 단위"
- Pydantic: 타입 검증과 JSON 직렬화를 자동화하는 라이브러리

라이브러리 역할:
- uvicorn: ASGI 웹 서버로, FastAPI 앱을 HTTP로 실행
- a2a.server.apps (A2AFastAPIApplication): A2A 프로토콜을 FastAPI로 구현한 기본 앱
- a2a.server.request_handlers (DefaultRequestHandler): 표준 요청 처리 로직
- a2a.server.tasks (InMemoryTaskStore): 메모리 기반 작업 저장소
"""

# ===== 라이브러리 임포트 =====

# A2A 프로토콜 기본 타입들: 에이전트 정보를 구조화/검증하기 위한 Pydantic 모델
# - AgentCard: 에이전트 전체 정보를 표준 JSON으로 나타낸 것
# - AgentSkill: 에이전트가 제공하는 개별 기능/작업을 정의
# - AgentCapabilities: 에이전트가 지원하는 추가 기능(스트리밍, 알림 등)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# pathlib: 파일 경로를 객체 지향적으로 다루는 라이브러리
# - Path(__file__).parent.parent로 현재 파일의 상위 디렉토리 경로를 안전하게 구성
from pathlib import Path

# uvicorn: ASGI(비동기) 웹 서버
# - Python의 비동기 웹 앱(FastAPI, Starlette)을 HTTP 엔드포인트로 실행하는 역할
# - 높은 동시성으로 여러 요청을 동시에 처리 가능
import uvicorn
import sys

# ===== A2A 프로토콜 서버 컴포넌트 =====

# 상위 디렉토리를 Python 모듈 검색 경로에 추가(상대 import를 위해)
# - Path(__file__).parent.parent: 현재 파일 위치의 상위 폴더
# - sys.path.append(): 이 경로를 모듈 검색 경로에 등록
sys.path.append(str(Path(__file__).parent.parent))

# A2AFastAPIApplication: A2A 프로토콜을 FastAPI로 구현한 기본 웹 애플리케이션
# - 에이전트 카드 조회, 스킬 실행, 메시지 수신 등 표준 HTTP 엔드포인트 제공
# - 다른 에이전트가 HTTP 요청으로 이 서버에 접근 가능
from a2a.server.apps import A2AFastAPIApplication

# DefaultRequestHandler: A2A 클라이언트(다른 에이전트)의 요청을 처리하는 표준 로직
# - 스킬 실행 요청 수신, 결과 계산, 응답 반환
# - 오류 처리, 예외 상황 관리
from a2a.server.request_handlers import DefaultRequestHandler

# InMemoryTaskStore: 진행 중인 작업 상태를 메모리에 저장
# - 작업 ID, 진행 상태, 결과 등을 관리
# - 데이터베이스 대신 메모리 사용 (개발/테스트용)
from a2a.server.tasks import InMemoryTaskStore

# 에이전트 실행 로직이 담긴 모듈
# - 실제 "인사 기능"을 수행하는 HelloWorldAgentExecutor 클래스 포함
from basic_agent.agent_executor import HelloWorldAgentExecutor

# ===== 에이전트 정보 정의 =====

def create_agent_card() -> AgentCard:
    """에이전트 카드 생성 함수
    
    목적: 
    - A2A 프로토콜을 따르는 표준 JSON 형식으로 
      "이 에이전트가 뭘 할 수 있는가"를 명시
    
    반환: 
    - AgentCard(Pydantic BaseModel)로, 
      다른 에이전트/클라이언트가 이해할 수 있는 정보 제공
    """
    
    # ===== Step 1: 에이전트가 수행할 수 있는 "스킬"(작업) 정의 =====
    # AgentSkill: Pydantic의 BaseModel을 상속해 필드 타입/필수 여부를 자동 검증
    greeting_skill = AgentSkill(
        # 필수 필드들
        id="basic_greeting",
        # - 스킬을 구별하는 고유 식별자
        # - 다른 스킬과 중복되면 안 됨
        
        name="Basic Greeting",
        # - 사람이 읽기 좋은 스킬 이름
        # - UI나 로그에서 표시되는 이름
        
        description="간단한 인사와 기본적인 대화를 제공합니다.",
        # - 스킬의 기능을 설명하는 텍스트
        # - 클라이언트가 이 스킬을 선택할 때 참고
        
        # 선택 필드들 (optional): 메타데이터로 검색/필터링에 사용
        tags=["greeting", "hello", "basic"],
        # - 카테고리/검색용 태그
        # - 클라이언트가 비슷한 스킬들을 찾을 때 사용
        
        examples=["안녕하세요", "hello", "hi", "고마워요"],
        # - 사용 예시
        # - 클라이언트가 이 스킬을 어떻게 사용할지 알 수 있게 함
        
        input_modes=["text"],
        # - 입력 형식: 텍스트 기반
        # - 다른 옵션: "voice", "image" 등도 가능
        
        output_modes=["text"]
        # - 출력 형식: 텍스트 기반
        # - 클라이언트가 어떤 형식의 응답을 받을지 알 수 있음
    )
    
    # ===== Step 2: 전체 에이전트 정보(카드)를 구성 =====
    # AgentCard도 Pydantic BaseModel: 모든 필드가 타입 검증되고 JSON으로 직렬화 가능
    agent_card = AgentCard(
        # 필수 정보
        name="Basic Hello World Agent",
        # - 에이전트의 이름
        # - 다른 에이전트/클라이언트에서 식별할 때 사용
        
        description="A2A 프로토콜을 학습하기 위한 기본적인 Hello World 에이전트입니다.",
        # - 에이전트의 전체 목적/기능 설명
        # - 검색 결과에도 표시될 수 있음
        
        url="http://localhost:9999/",
        # - 다른 에이전트가 이 서버에 접근할 HTTP 주소
        # - 클라이언트는 이 URL로 HTTP 요청을 보냄
        # - 로컬 개발: localhost, 프로덕션: 실제 서버 주소
        
        version="1.0.0",
        # - 에이전트의 버전
        # - 업데이트 추적이나 호환성 확인에 사용
        
        # 선택 정보
        default_input_modes=["text"],
        # - 기본 입력 형식
        # - 클라이언트가 명시하지 않으면 이 형식 사용
        
        default_output_modes=["text"],
        # - 기본 출력 형식
        # - 클라이언트가 명시하지 않으면 이 형식으로 응답
        
        capabilities=AgentCapabilities(streaming=True),
        # - 지원 기능 정의
        # - streaming=True: 점진적 응답 가능 (채팅처럼 한 줄씩 출력)
        # - 다른 옵션: pushNotifications, stateTransitionHistory 등
        
        skills=[greeting_skill],
        # - 이 에이전트가 제공하는 모든 스킬 목록
        # - 위에서 정의한 greeting_skill을 포함
        # - 여러 스킬이 있으면 리스트에 추가: [skill1, skill2, skill3]
        
        supports_authentication_extended_card=False
        # - 인증 기능 여부
        # - False: 인증 없이 누구나 이 에이전트 사용 가능
        # - True: 특정 사용자/권한만 사용 가능 (향후 기능)
    )
    
    return agent_card

# ===== 메인 실행 로직 =====

def main():
    """에이전트 카드를 생성해 표준 JSON 형식으로 출력
    
    프로세스:
    1. create_agent_card()로 AgentCard 객체 생성
    2. model_dump_json()으로 표준 JSON 문자열 변환
    3. print()로 콘솔에 출력 (또는 파일 저장, HTTP 응답 등)
    """
    agent_card = create_agent_card()
    
    # model_dump_json(): Pydantic 모델을 JSON 문자열로 변환
    # - AgentCard의 모든 필드가 표준 A2A JSON 형식으로 변환됨
    # - 다른 에이전트가 HTTP GET으로 이 정보를 받아 상호작용 가능
    # - indent=2를 추가하면 보기 좋게 포맷팅: agent_card.model_dump_json(indent=2)
    print(agent_card.model_dump_json())

# ===== 진입점 =====

if __name__ == "__main__":
    # 이 파일을 직접 실행할 때만 main() 호출
    # - import로 사용될 때는 실행 안 됨
    # - 다른 파일에서 create_agent_card()만 임포트해서 사용 가능
    main()

# ===== 출력 예시 및 각 필드 설명 =====

"""
json 형식 출력 샘플:

{
    "additionalInterfaces": null,
    # - 추가 인터페이스 (null = 없음)
    
    "capabilities": {
        "extensions": null,              # 확장 기능 미지원
        "pushNotifications": null,       # 푸시 알림 미지원
        "stateTransitionHistory": null,  # 상태 전환 이력 미지원
        "streaming": true                # 스트리밍 응답 지원 (우리가 설정함)
    },
    
    "defaultInputModes": ["text"],       # 기본 입력 형식: 텍스트
    "defaultOutputModes": ["text"],      # 기본 출력 형식: 텍스트
    
    "description": "A2A 프로토콜을 학습하기 위한 기본적인 Hello World 에이전트입니다.",
    # - 에이전트 설명
    
    "documentationUrl": null,            # 문서 URL 없음
    "iconUrl": null,                     # 아이콘 이미지 URL 없음
    
    "name": "Basic Hello World Agent",   # 에이전트 이름
    
    "preferredTransport": "JSONRPC",
    # - 선호하는 통신 방식: JSONRPC (A2A 표준)
    
    "protocolVersion": "0.3.0",
    # - A2A 프로토콜 버전
    
    "provider": null,                    # 제공자 정보 없음
    "security": null,                    # 보안 설정 없음
    "securitySchemes": null,             # 보안 스키마 없음
    "signatures": null,                  # 서명 검증 없음
    
    "skills": [{
        "description": "간단한 인사와 기본적인 대화를 제공합니다.",
        "examples": ["안녕하세요", "hello", "hi", "고마워요"],
        "id": "basic_greeting",
        "inputModes": ["text"],
        "name": "Basic Greeting",
        "outputModes": ["text"],
        "security": null,
        "tags": ["greeting", "hello", "basic"]
    }],
    # - 이 에이전트가 제공하는 스킬 목록
    
    "supportsAuthenticatedExtendedCard": null,
    # - 인증 확장 카드 미지원
    
    "url": "http://localhost:9999/",     # 에이전트 접근 URL
    "version": "1.0.0"                   # 에이전트 버전
}
"""
