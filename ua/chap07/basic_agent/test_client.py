"""
===================================
A2A(Agent to Agent) 프로토콜 클라이언트 테스트
===================================

[ 목적 ]
원격 A2A 에이전트 서버와 통신하는 클라이언트 구현
- HTTP 기반 에이전트 간 상호작용
- 표준 메시지 형식 (A2A 프로토콜) 사용
- 스트리밍/논스트리밍 응답 처리

[ 주요 기능 ]
1. 에이전트 카드 조회 (GET /agent/card)
   - 에이전트 메타정보 (이름, 설명, 스킬) 조회
   - A2ACardResolver로 자동 파싱

2. 클라이언트 설정 (ClientConfig + ClientFactory)
   - 스트리밍 여부 설정 (streaming=True/False)
   - HTTP 클라이언트 커넥션 풀 재사용
   
3. 메시지 송수신 (send_message)
   - A2A 표준 Message 포맷 사용
   - 비동기 이터레이터로 응답 처리 (async for)
   - 예외 처리 (서버 미실행, 네트워크 에러 등)

[ 아키텍처 ]
┌─────────────────┐
│  Test Client    │  (이 파일: test_client.py)
└────────┬────────┘
         │ HTTP 요청/응답
         │ A2A 프로토콜
         │
┌────────▼────────┐
│  A2A Server     │  (server.py)
│  Port: 9999     │  - AgentCard 제공
└────────┬────────┘  - 메시지 수신/응답
         │
┌────────▼────────┐
│  LLM (Gemini)   │  (agent_executor.py)
│  또는 로컬 LM   │  - 텍스트 생성
└─────────────────┘

[ 실행 흐름 ]
1. asyncio.run(main())
   └─ await test_basic_agent()

2. test_basic_agent() 단계별 실행:
   Step 1: A2ACardResolver로 에이전트 카드 조회
   Step 2: 논스트리밍 ClientConfig 생성
   Step 3: 스트리밍 ClientConfig 생성
   Step 4: 테스트 메시지 준비
   Step 5: 논스트리밍 클라이언트로 메시지 송수신
   Step 6: 스트리밍 클라이언트로 메시지 송수신 (실시간)

3. 예외 처리:
   - 서버 미실행: "check server is running" 안내
   - 네트워크 에러: 에러 메시지 출력
   - 검증 에러: Pydantic 검증 오류 메시지

[ 사용할 라이브러리 및 개념 ]
표준:
  - asyncio: 비동기 프로그래밍
  - uuid: 고유 ID 생성
  - typing: 타입 힌트

외부:
  - httpx: 비동기 HTTP 클라이언트
  - a2a: Agent to Agent 프로토콜 구현체
    * A2ACardResolver: 에이전트 카드 조회
    * ClientFactory: 클라이언트 생성
    * ClientConfig: 클라이언트 설정
    * Message: 표준 메시지 포맷
    * get_message_text(): 텍스트 추출

[ 주요 개념 설명 ]
1. async/await
   - 비동기 함수 (async def): I/O 대기 중 제어권 반환
   - await 키워드: 비동기 작업 완료 대기
   - asyncio.run(): 이벤트 루프 생성 및 코루틴 실행

2. Context Manager (with/async with)
   - with: 리소스 자동 획득/해제
   - async with: 비동기 버전
   - as 키워드: 리소스를 변수에 할당
   
3. 타입 힌트 (typing 모듈)
   - Optional[T]: T 또는 None
   - Any: 모든 타입 (타입 검증 없음)
   - Union[T1, T2, ...]: 여러 타입 중 하나

4. A2A 프로토콜
   - REST/JSONRPC 기반 에이전트 간 통신
   - AgentCard: 에이전트 메타정보
   - Message: 일관된 메시지 형식
   - Event: 메시지, 에러, 상태 등 이벤트 타입

5. 스트리밍 vs 논스트리밍
   - 논스트리밍: 전체 응답을 한 번에 수신
   - 스트리밍: 응답을 부분씩 수신 (ChatGPT 스타일)

[ 파일 구조 ]
1. import 섹션 (라인 1-120)
   - 표준 라이브러리
   - 타입 힌트 설명
   - A2A 프로토콜 라이브러리
   
2. 함수 정의 (라인 120-500)
   - create_user_message(): 사용자 메시지 생성
   - test_basic_agent(): 주요 테스트 함수
   - main(): 진입점 함수
   
3. 진입점 (라인 500-끝)
   - if __name__ == "__main__": asyncio.run(main())

[ 실행 방법 ]
# 방법 1: 직접 실행
$ python ua/chap07/basic_agent/test_client.py

# 방법 2: 모듈로 실행
$ python -m ua.chap07.basic_agent.test_client

# 방법 3: 다른 스크립트에서 import
>>> import asyncio
>>> from ua.chap07.basic_agent.test_client import test_basic_agent
>>> asyncio.run(test_basic_agent())

[ 서버 실행 (전제 조건) ]
터미널에서 먼저 에이전트 서버를 실행해야 함:
$ python -m ua.chap07.basic_agent.server
또는
$ uvicorn ua.chap07.basic_agent.server:app --port 9999

[ 예상 결과 ]
Basic Hello world A2A Agent 테스트 시작...
Server URL: http://localhost:9999
--------------------------------------------------
agent card 조회 중...
agent name: HelloAgent
agent description: A simple agent that responds with greetings
agent skills: ['basic_greeting']

=== Non-Streaming Client 테스트 ===

1. User: 안녕하세요
(에이전트 응답...)

=== Streaming Client 테스트 ===

1. User: 안녕하세요
agent streaming: (실시간 응답...)

[ 에러 처리 ]
1. Connection refused
   → 서버가 실행 중이지 않음
   → 해결: python -m ua.chap07.basic_agent.server 실행

2. Timeout
   → 네트워크 느림 또는 서버 응답 지연
   → ClientConfig의 timeout 값 증가

3. ValidationError
   → Message 형식 오류
   → create_user_message() 사용해서 올바른 형식 생성

4. RuntimeError: Event loop is already running
   → Jupyter 노트북에서 발생
   → nest_asyncio.apply() 사용

[ 참고자료 ]
- A2A 프로토콜: https://github.com/openinterpreter/open-interpreter/tree/main/src/open_interpreter/server/types
- asyncio 문서: https://docs.python.org/3/library/asyncio.html
- httpx 문서: https://www.python-httpx.org/
- Pydantic: https://docs.pydantic.dev/

===================================
"""

# ===== 표준 라이브러리 =====

# asyncio: 비동기 프로그래밍을 위한 Python 표준 라이브러리
# - async/await 문법을 사용해 I/O 대기 중에도 다른 작업 처리 가능
# - 네트워크 요청, 파일 I/O 등 느린 작업에 최적화
# - asyncio.run(): 비동기 함수를 실행하는 진입점
import asyncio

# uuid: 고유 식별자(UUID) 생성 라이브러리
# - uuid4(): 무작위로 생성된 128비트 고유 식별자
# - 메시지 ID, 세션 ID 등 고유성이 필요할 때 사용
from uuid import uuid4

# ===== typing 모듈: 타입 힌트 (타입 검증 및 IDE 자동완성) =====
# 타입 힌트: 함수 인자와 반환값의 타입을 명시해 코드 안정성과 가독성 향상
from typing import Any, Optional

# 자주 사용되는 typing 구성요소 설명:
# - Any: "모든 타입" 허용 (타입 검증 안 함, 마지막 수단)
#   예: def process(data: Any) -> None: ...
#       어떤 타입이든 받을 수 있음
#
# - Optional[T]: T 또는 None 허용 (필수 아님)
#   예: def get_user(user_id: Optional[str] = None) -> User:
#       user_id가 없어도 됨 (기본값 None)
#
# - List[T]: T 타입 요소의 리스트
#   예: def process_items(items: List[str]) -> None:
#       문자열 리스트만 받음
#
# - Dict[K, V]: 키 K, 값 V인 딕셔너리
#   예: def parse_config(config: Dict[str, Any]) -> None:
#       문자열 키와 모든 타입의 값을 가진 딕셔너리
#
# - Union[T1, T2, ...]: T1 또는 T2 중 하나 (Python 3.10+는 T1 | T2 가능)
#   예: def handle_response(data: Union[str, dict]) -> None:
#       문자열 또는 딕셔너리 받음
#
# - Callable[[P1, P2], R]: P1, P2를 인자로 받아 R을 반환하는 함수
#   예: def apply_func(func: Callable[[int, int], int]) -> int:
#       두 정수를 받아 정수를 반환하는 함수만 받음
#
# - TypedDict: 구조화된 딕셔너리 타입 (필드명과 타입 고정)
#   예: class Config(TypedDict):
#           name: str
#           port: int
#       name(str), port(int) 필드를 가진 딕셔너리
#
# - Generic[T]: 제네릭 타입 (여러 타입을 받을 수 있음)
#   예: class Container(Generic[T]):
#           def get(self) -> T: ...
#       Container[str], Container[int] 등 유연하게 사용

# ===== 외부 라이브러리 =====

# httpx: 비동기 HTTP 클라이언트 라이브러리
# - requests의 비동기 버전
# - httpx.AsyncClient로 비동기 HTTP 요청 가능
# - GET, POST, PUT, DELETE 등 모든 HTTP 메서드 지원
# - 사용: async with httpx.AsyncClient() as client: ...
import httpx

# ===== A2A 프로토콜 클라이언트 라이브러리 =====

# A2ACardResolver: 에이전트 카드 정보를 원격으로 조회하고 파싱하는 클래스
# - URL에서 에이전트 카드 JSON 다운로드
# - 에이전트 정보(이름, 스킬, 기능) 추출
# - 여러 에이전트 카드 캐싱 지원
# - 사용: resolver = A2ACardResolver(httpx_client, base_url)
#        agent_card = await resolver.get_agent_card()
from a2a.client import A2ACardResolver

# ClientFactory: A2A 클라이언트 인스턴스를 생성하는 팩토리
# - 에이전트 카드와 설정을 기반으로 클라이언트 생성
# - 여러 유형의 클라이언트 생성 가능 (스트리밍/논스트리밍)
# - Pydantic 기반: 타입 검증 자동 수행
# - 사용: factory = ClientFactory(config)
#        client = factory.create(agent_card)
from a2a.client.client_factory import ClientFactory

# ClientConfig: A2A 클라이언트의 설정 정보
# - httpx_client: 재사용할 HTTP 클라이언트 (중요: connection pool 공유)
# - streaming: True면 스트리밍(응답 부분씩), False면 한 번에 수신
# - timeout: 요청 타임아웃 (초)
# - 사용: config = ClientConfig(httpx_client=client, streaming=True)
from a2a.client.client import ClientConfig

# Message: A2A 표준 메시지 형식 (타입 정의 클래스)
# - role: 메시지 발신자 역할 ("user", "agent", "system")
# - parts: 메시지 콘텐츠 배열 (여러 타입 혼합 가능)
# - messageId: 고유 식별자 (서버에서 추적/재시도용)
# - metadata: 선택사항, 타임스탬프 등 추가 정보
# - 직렬화: model_dump_json()로 JSON 문자열 변환
from a2a.types import Message 

# get_message_text(): Message 객체에서 텍스트 추출 유틸함수
# - Message.parts 배열을 순회하며 "text" kind인 부분 수집
# - 이미지, 음성 등은 제외
# - 여러 텍스트 부분이 있으면 연결 반환
# - 사용: text = get_message_text(message)
from a2a.utils import get_message_text

def create_user_message(text: str, message_id: Optional[str] = None) -> Message:
    """A2A 표준 형식의 사용자 메시지 생성 함수
    
    목적: 
    - 사용자 입력을 A2A 프로토콜 표준 Message 형식으로 변환
    - 에이전트 서버로 보낼 메시지 준비
    
    인자:
    - text: str 
      사용자가 입력한 메시지 텍스트
    - message_id: Optional[str] = None
      메시지 고유 식별자 (선택사항)
      - None이면 uuid4().hex로 자동 생성 (무작위 UUID)
      - 명시하면 그 값 사용 (서버에서 메시지 추적/재시도 처리용)
    
    반환:
    - Message: A2A 표준 메시지 객체
      다음 필드 포함:
      - role: "user" (발신자 역할: 사용자)
      - parts: 메시지 내용 배열 (텍스트, 이미지, 음성 등)
      - messageId: 고유 식별자
    
    사용 예:
    >>> msg = create_user_message("안녕하세요")
    >>> # Message(role="user", parts=[...], messageId="abc123...")
    """
    return Message(
        # role: 메시지 발신자 역할
        # - "user": 사용자 입력
        # - "agent": 에이전트 응답
        # - "system": 시스템 메시지
        role="user",
        
        # parts: 메시지 콘텐츠 배열
        # - "kind": 콘텐츠 타입 ("text", "image", "audio" 등)
        # - "text": 실제 메시지 텍스트
        # - 여러 부분을 배열로 지정해 복합 메시지 가능
        parts=[{"kind": "text", "text": text}],
        
        # messageId: 메시지 고유 식별자
        # - message_id가 제공되면 사용, 아니면 uuid4().hex로 생성
        # - uuid4().hex: UUID를 16진수 문자열로 변환 (36자 → 32자)
        # - 서버에서 메시지 추적, 재시도, 중복 검사 등에 사용
        messageId=message_id or uuid4().hex
    )

async def test_basic_agent():
    """
    A2A 에이전트 서버와 통신하는 통합 테스트 함수
    
    목적:
    1. 원격 에이전트 서버에서 에이전트 카드(메타정보) 조회
    2. 에이전트 정보(이름, 스킬) 출력
    3. 스트리밍/논스트리밍 클라이언트로 메시지 송수신 테스트
    4. 에이전트의 응답 확인
    
    흐름:
    1. async with로 httpx.AsyncClient 생성 (연결 관리)
    2. A2ACardResolver로 /agent/card 엔드포인트에서 카드 조회
    3. ClientConfig 생성: 스트리밍 여부 설정
    4. ClientFactory로 client 인스턴스 생성
    5. send_message()로 메시지 전송 및 응답 수신
    6. 응답 텍스트 출력
    
    예외 처리:
    - 서버 연결 불가: "check server is running" 메시지
    - 요청 에러: try/except로 예외 정보 출력
    """
    base_url = "http://localhost:9999"  # A2A 에이전트 서버 URL (localhost 포트 9999)
    print("Basic Hello world A2A Agent 테스트 시작...")
    print(f"Server URL: {base_url}")
    print("-"*50)
    
    # ===== async with 문법 설명 =====
    # 
    # 1) with 문: Context Manager (컨텍스트 매니저)
    #    - 리소스의 획득과 해제를 자동으로 관리
    #    - __enter__()와 __exit__()가 자동 호출됨
    #    - 예외 발생 여부와 관계없이 __exit__() 항상 실행 (안전한 정리)
    #
    # 2) async with 문: 비동기 Context Manager
    #    - with 문의 비동기 버전
    #    - __aenter__()과 __aexit__() 자동 호출
    #    - 비동기 작업 중 리소스를 안전하게 관리
    #    - 네트워크 연결, 파일 I/O 등에 사용
    #
    # 3) as 키워드: 별칭 지정
    #    - Context Manager가 반환한 객체를 변수에 할당
    #    - 이 블록 내에서만 유효 (scope 제한)
    #    - 블록을 벗어나면 자동으로 정리됨
    #
    # 실행 순서:
    #   1) httpx.AsyncClient() 객체 생성
    #   2) __aenter__() 호출 → 연결 설정
    #   3) httpx_client 변수에 할당
    #   4) with 블록 내 코드 실행
    #   5) 블록 완료 시 __aexit__() 호출 → 연결 종료 (또는 예외 처리)
    #
    # 장점:
    #   - 자동 리소스 정리: close()를 명시적으로 호출할 필요 없음
    #   - 예외 안전성: 예외 발생 시에도 정리 코드 실행
    #   - 코드 간결성: try/finally 없이 안전한 리소스 관리
    #
    # ===== 자주 사용하는 with/async with 패턴 =====
    #
    # 1) 파일 처리
    #    with open("file.txt", "r") as f:
    #        content = f.read()
    #    # 블록 후 자동으로 f.close() 호출
    #
    # 2) 비동기 HTTP 요청
    #    async with httpx.AsyncClient() as client:
    #        response = await client.get("http://example.com")
    #    # 블록 후 자동으로 client.aclose() 호출
    #
    # 3) 데이터베이스 연결
    #    async with database.connection() as conn:
    #        result = await conn.execute(query)
    #    # 블록 후 자동으로 conn.close() 호출
    #
    # 4) Lock/Semaphore (동시성 제어)
    #    async with lock:
    #        # 임계 영역(critical section)
    #        shared_resource.modify()
    #    # 블록 후 자동으로 lock 해제
    #
    # 5) 중첩 사용
    #    async with client1 as c1:
    #        async with client2 as c2:
    #            resp1 = await c1.get(url1)
    #            resp2 = await c2.get(url2)
    #    # 가장 안쪽부터 차례로 정리됨
    #
    # 6) 여러 리소스 동시 관리
    #    async with httpx.AsyncClient() as client, \
    #              asyncio.timeout(10):
    #        response = await client.get(url)
    #    # 여러 context manager를 쉼표로 구분해 사용
    #
    # ===== 동기 vs 비동기 Context Manager =====
    #
    # 동기 (일반 파일, 동기 HTTP 라이브러리)
    #   with resource() as r:
    #       data = r.read()
    #
    # 비동기 (비동기 라이브러리)
    #   async with resource() as r:
    #       data = await r.read()
    #
    # 차이점:
    # - async with는 async 함수 내에서만 사용 가능
    # - await 키워드와 함께 비동기 작업 수행
    # - 다른 작업이 대기 중에 CPU 자원 활용 가능
    
    async with httpx.AsyncClient() as httpx_client:
        """
        httpx.AsyncClient: 비동기 HTTP 클라이언트
        - as httpx_client: Client 객체를 변수에 할당
        - 블록을 벗어나면 자동으로 연결 종료
        - 여러 요청에서 재사용하므로 connection pool 효율적
        """
        try:
            # ===== 1단계: 에이전트 카드 조회 =====
            # A2ACardResolver: 원격 서버의 에이전트 정보를 가져오는 resolver
            # - httpx_client: HTTP 요청을 수행할 client 객체
            # - base_url: 에이전트 서버의 기본 URL (예: http://localhost:9999)
            # 
            # get_agent_card() 내부 동작:
            # 1. base_url + "/agent/card" 엔드포인트로 GET 요청
            # 2. JSON 응답을 AgentCard 모델로 파싱
            # 3. AgentCard 객체 반환 (이름, 설명, 스킬 목록 포함)
            resolver = A2ACardResolver(
                httpx_client=httpx_client,  # 재사용 가능한 client
                base_url=base_url  # http://localhost:9999
            )
            
            print("agent card 조회 중...")
            
            # await resolver.get_agent_card()
            # - async 함수 호출이므로 await 필수
            # - 네트워크 요청 중 제어권을 이벤트 루프에 반환 (다른 작업 진행 가능)
            # - 응답 수신 시 제어권 반환 (계속 실행)
            agent_card = await resolver.get_agent_card()
            
            # AgentCard 정보 출력
            # AgentCard는 다음 필드 포함:
            # - name: 에이전트 이름 (예: "HelloAgent")
            # - description: 에이전트 설명
            # - skills: AgentSkill 객체의 리스트
            #   - AgentSkill.name: 스킬 이름 (예: "basic_greeting")
            #   - AgentSkill.description: 스킬 설명
            # - capabilities: 에이전트가 지원하는 기능 (streaming, tools 등)
            print(f"agent name: {agent_card.name}")
            print(f"agent description: {agent_card.description}")
            print(f"agent skills: {[skill.name for skill in agent_card.skills]}")
            # 예상 출력:
            # agent name: HelloAgent
            # agent description: A simple agent that responds with greetings
            # agent skills: ['basic_greeting']
            print()  # 빈 줄

            # ===== 2단계: 논스트리밍 클라이언트 설정 =====
            # 논스트리밍: 에이전트 응답을 완전히 수신한 후 한 번에 반환
            # - 응답 시간 좀 더 김
            # - 전체 응답 구조를 먼저 확인 가능
            # - 짧은 응답에 유용
            
            # ClientConfig: 클라이언트 동작 방식 설정 (Pydantic 모델)
            # - httpx_client: 재사용할 HTTP 클라이언트 (connection pool 공유)
            # - streaming: False = 논스트리밍, True = 스트리밍
            # - timeout: 요청 타임아웃 (선택사항, 기본값 30초)
            non_streaming_config = ClientConfig(
                httpx_client=httpx_client,  # 같은 client 재사용 (연결 풀 효율)
                streaming=False  # 전체 응답을 한 번에 받기
            )
            
            # ClientFactory: ClientConfig를 기반으로 client 인스턴스 생성
            # - Pydantic 기반: config 자동 검증
            # - create(agent_card): 에이전트 카드를 기반으로 클라이언트 생성
            #   - 에이전트의 API 엔드포인트 설정
            #   - 메시지 형식 검증 규칙 적용
            non_streaming_factory = ClientFactory(non_streaming_config)
            
            # factory.create() 반환값: A2A 클라이언트
            # - send_message(message: Message) -> AsyncIterator[Event]
            #   메시지를 보내고 이벤트 스트림 반환
            # - 이벤트 종류: Message (응답), Error, Status 등
            non_streaming_client = non_streaming_factory.create(agent_card)
            
            # ===== 3단계: 스트리밍 클라이언트 설정 =====
            # 스트리밍: 에이전트 응답을 부분씩 수신
            # - 응답이 더 빠르게 표시됨 (사용자 경험 개선)
            # - 특히 긴 응답에서 사용자가 먼저 읽기 시작 가능
            # - ChatGPT 스타일의 "타이핑하는 것처럼" 표시
            
            streaming_config = ClientConfig(
                httpx_client=httpx_client,  # 같은 client 재사용
                streaming=True  # 부분씩 응답 받기 (streaming)
            )
            
            streaming_factory = ClientFactory(streaming_config)
            streaming_client = streaming_factory.create(agent_card)

            # ===== 4단계: 테스트 메시지 준비 =====
            # 다양한 테스트 시나리오를 위한 메시지들
            # - 한국어 인사, 영어 인사, 질문, 감사 등
            # 각 메시지에 대해 에이전트의 응답 확인
            test_messages = [
                "안녕하세요",  # 한국어 기본 인사
                "Hello, how are you?",  # 영어 인사
                "오늘 날씨 어때?",  # 한국어 질문
                "고마워요"  # 한국어 감사
            ]

            # ===== 5단계: 논스트리밍 클라이언트 테스트 =====
            print("=== Non-Streaming Client 테스트 ===")
            # enumerate(test_messages, 1): 메시지에 1부터 시작하는 번호 부여
            # - i: 1, 2, 3, 4 (번호)
            # - message: 메시지 텍스트
            for i, message in enumerate(test_messages, 1):
                print(f"\n{i}. User: {message}")
                
                # create_user_message(message)
                # - 입력 텍스트를 A2A 표준 Message 객체로 변환
                # - Message(role="user", parts=[...], messageId="...")
                user_message = create_user_message(message)
                
                # send_message(user_message)
                # - 비동기 메서드: 메시지를 에이전트에 전송하고 응답 스트림 반환
                # - 반환: AsyncIterator (비동기 이터레이터)
                #   - 논스트리밍에서도 이터레이터 사용 (A2A 표준)
                #   - 하지만 보통 1개 이벤트만 반환 (전체 응답)
                #   - 스트리밍에서는 여러 부분 이벤트 반환
                #
                # async for event in non_streaming_client.send_message(user_message):
                # - 비동기 이터레이터 순회 (async for)
                # - await를 자동으로 처리 (편리함)
                # - 이벤트가 발생할 때마다 루프 본문 실행
                async for event in non_streaming_client.send_message(user_message):
                    # isinstance(event, Message)
                    # - event 타입 확인
                    # - Message: 에이전트 응답 메시지 (role="agent")
                    # - Error: 에러 발생
                    # - Status: 상태 업데이트 (예: "processing", "complete")
                    if isinstance(event, Message):
                        # get_message_text(event)
                        # - Message 객체에서 텍스트만 추출
                        # - event.parts 배열의 "text" 타입 항목만 수집
                        # - 이미지, 음성 등은 제외
                        response_text = get_message_text(event)
                        
                        # 응답 출력
                        print(response_text)
                        
                        # break: 첫 응답만 처리 후 다음 메시지로 이동
                        # (논스트리밍이므로 보통 1개 이벤트만 발생)
                        break
            print("\n" + "="*50)

            # ===== 6단계: 스트리밍 클라이언트 테스트 =====
            # 스트리밍 응답의 차이점:
            # - 논스트리밍: 응답 완성 → 한 번에 표시
            # - 스트리밍: 응답이 점진적으로 도착 → 부분씩 표시 (실시간 감)
            #
            # 예: 응답이 "안녕하세요! 반갑습니다."라면
            # - 논스트리밍: [Message(text="안녕하세요! 반갑습니다.")]
            # - 스트리밍: [Message(text="안녕"),
            #             Message(text="하세요"),
            #             Message(text="! 반갑"),
            #             Message(text="습니다")]
            #   (여러 이벤트로 분산)
            
            print("=== Streaming Client 테스트 ===")
            
            # test_messages[:3]
            # - 첫 3개 메시지만 테스트 (스트리밍은 처리 시간이 좀 더 걸려서)
            # - ["안녕하세요", "Hello, how are you?", "오늘 날씨 어때?"]
            for i, message in enumerate(test_messages[:3], 1):
                print(f"\n{i}. User: {message}")
                
                user_message = create_user_message(message)
                
                # "agent streaming" 프롬프트 출력
                # - end="": 줄바꿈 없음 (다음 출력이 같은 줄에 계속)
                # - flush=True: 버퍼 즉시 플러시 (화면에 즉시 나타남)
                #   - 스트리밍 효과를 보기 위해 필수
                #   - 없으면 버퍼에서 대기했다가 한번에 출력됨
                print("agent streaming: ", end="", flush=True)
                
                response_text = ""
                
                # 스트리밍 응답 순회
                # - 비동기 이터레이터에서 이벤트 하나씩 받음
                # - 논스트리밍과 달리 여러 이벤트 발생 가능
                async for event in streaming_client.send_message(user_message):
                    # 이벤트 타입 확인
                    if isinstance(event, Message):
                        # 이번 이벤트의 텍스트 부분 추출
                        part_text = get_message_text(event)
                        
                        # 부분 텍스트를 그대로 출력 (누적 안 함)
                        # - ChatGPT 스타일: "타이핑하는 것처럼" 한 글자씩 나타남
                        # - 실제로는 여러 글자씩 일괄 전송되지만, 분산되어 표시됨
                        # - end="": 줄바꿈 없음
                        # - flush=True: 즉시 출력 (지연 없음)
                        print(part_text, end="", flush=True)
                        
                        # response_text에 누적 (최종 응답 확인용)
                        # - 현재 코드에서는 사용 안 하지만, 나중에 응답 저장할 때 유용
                        response_text += part_text
                
                # 줄바꿈 (다음 메시지로 이동)
                print()  # "\n" 같은 효과
            print("\n" + "="*50)

        # ===== 예외 처리 =====
        # try/except로 감싼 이유:
        # - 서버가 실행 중이지 않으면 연결 에러
        # - 메시지 포맷이 잘못되면 검증 에러
        # - 네트워크 문제로 타임아웃 가능
        except Exception as ex:
            # 에러 정보 출력
            # - str(ex): 에러 메시지 문자열로 변환
            # - 예: "Connection refused", "Timeout", "ValidationError"
            print("에러 발생:", str(ex))
            
            # 디버깅 힌트 제공
            print("check server is running at", base_url)
            print("server execute: python basic_agent/__main__.py") 
async def main():
    """
    비동기 테스트의 진입점 함수
    
    목적:
    - asyncio.run()이 호출할 진입 함수 제공
    - 모든 비동기 작업 조율 (현재는 test_basic_agent만 호출)
    - 향후 여러 테스트를 동시 실행하거나 순차 실행 가능
    
    사용 예:
    >>> import asyncio
    >>> asyncio.run(main())
    # 또는
    >>> import subprocess
    >>> subprocess.run([sys.executable, __file__])
    """
    # test_basic_agent() 호출
    # - async 함수이므로 await 필수
    # - 테스트 전체 완료 시까지 대기
    await test_basic_agent()

# ===== 스크립트 진입점 =====
# __name__ == "__main__": 이 파일이 직접 실행되었음을 의미
# (import될 때는 __name__이 "a2a.client.test_client" 같은 모듈 경로)
#
# 사용법:
# 1) 터미널에서 직접 실행:
#    $ python test_client.py
#    또는
#    $ python -m ua.chap07.basic_agent.test_client
#
# 2) 다른 스크립트에서 import하면 이 블록은 실행 안 됨:
#    >>> from ua.chap07.basic_agent import test_client
#    >>> # __name__ != "__main__"이므로 main() 호출 안 됨
#    >>> asyncio.run(test_client.main())  # 명시적으로 호출해야 함
#
if __name__ == "__main__":
    # asyncio.run(코루틴) 설명:
    # - Python 3.7+ 표준 비동기 진입점
    # - 새 이벤트 루프 생성
    # - 코루틴 실행
    # - 완료 후 루프 종료
    #
    # 구 방식과 비교:
    # 구: loop = asyncio.get_event_loop()
    #     loop.run_until_complete(main())
    #     loop.close()
    # 신: asyncio.run(main())  # 훨씬 간단
    #
    # 주의:
    # - asyncio.run()은 이미 실행 중인 이벤트 루프가 있으면 에러
    # - Jupyter 노트북에서는 기존 루프가 있어서 에러 발생 가능
    # - 그 경우 nest_asyncio 라이브러리로 해결:
    #   import nest_asyncio
    #   nest_asyncio.apply()
    #   asyncio.run(main())
    # asyncio.run(main())로 비동기 테스트 시작
    asyncio.run(main())        

"""
Basic Hello world A2A Agent 테스트 시작...
Server URL: http://localhost:9999
--------------------------------------------------
agent card 조회 중...
agent name: Basic Hello World Agent
agent description: A2A 프로토콜을 학습하기 위한 기본적인 Hello World 에이전트입니다.
agent skills: ['Basic Greeting']

=== Non-Streaming Client 테스트 ===

1. User: 안녕하세요
안녕하세요! 오늘 어떻게 도와드릴까요?

2. User: Hello, how are you?
Hello there! I'm doing well, thank you for asking. How can I help you today?

3. User: 오늘 날씨 어때?
안녕하세요! 오늘 날씨에 대해 알려드릴게요. 어디 날씨를 알고 싶으신가요? 위치를 알려주시면 자세한 날씨 정보를 알려드릴 수 있습니다. 😊

4. User: 고마워요
천만에요! 도울 일이 있으면 언제든지 말씀해주세요.

==================================================
=== Streaming Client 테스트 ===

1. User: 안녕하세요
agent streaming: 안녕하세요! 오늘 무엇을 도와드릴까요?

2. User: Hello, how are you?
agent streaming: Hello there! I'm doing well, thank you for asking. How can I help you today?

3. User: 오늘 날씨 어때?
agent streaming: 안녕하세요! 오늘 날씨에 대해 알려드릴게요. 어느 지역 날씨가 궁금하신가요?

==================================================
"""