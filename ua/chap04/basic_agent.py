from agents import Agent, Runner
import os

# 참고: Agent.__init__에는 base_url 인자가 없습니다.
# OpenAI 호환 엔드포인트를 쓰려면 SDK/환경변수로 전역 설정해야 합니다.
# 1) 엔드포인트 설정: OPENAI_BASE_URL
# 2) API 키 설정: OPENAI_API_KEY (LM Studio는 임의의 문자열도 허용)
os.environ.setdefault("OPENAI_BASE_URL", "http://192.168.45.167:50505/v1")
if "OPENAI_API_KEY" not in os.environ:
    # LMS_API_KEY가 있다면 재사용, 없으면 더미 키로 설정
    os.environ["OPENAI_API_KEY"] = os.getenv("LMS_API_KEY", "lm-studio")

# (선택) OpenAI Python SDK가 설치되어 있다면 전역 설정을 적용합니다.
try:
    import openai  # type: ignore
    openai.base_url = os.environ["OPENAI_BASE_URL"]
    openai.api_key = os.environ["OPENAI_API_KEY"]
except ImportError:
    pass

# 사용할 모델 식별자만 전달합니다. (예: lm-studio에서 노출되는 모델 이름)
hello_agent = Agent(
    name="HelloAgent",
    instructions="Say hello to the world",
    model="model-identifier",
)

result = Runner.run_sync(hello_agent, "프랑스어로만 인사해주세요.")
print(result.final_output)