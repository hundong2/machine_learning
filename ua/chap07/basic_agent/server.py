from a2a.types import AgentCapabilities, AgentCard, AgentSkill

def create_agent_card() -> AgentCard:
    """에이전트 카드 생성 함수"""
    greeting_skill = AgentSkill( #pydantic 의 BaseModel에서 상속
        id="basic_greeting", #agent skill 고유 id
        name="Basic Greeting", #agent skill 이름
        description="간단한 인사와 기본적인 대화를 제공합니다.", #agent skill 설명
        tags=["greeting", "hello", "basic"], #optional
        examples=["안녕하세요", "hello", "hi", "고마워요"], #optional
        input_modes=["text"], #optional
        output_modes=["text"] #optional
    )
    #agent card construction 
    agent_card = AgentCard(
        name="Basic Hello World Agent",
        description="A2A 프로토콜을 학습하기 위한 기본적인 Hello World 에이전트입니다.",
        url="http://localhost:9999/",#에이전트 서비스 URL
        version="1.0.0",
        default_input_modes=["text"], #optional
        default_output_modes=["text"], #optional
        capabilities=AgentCapabilities(streaming=True), #에이전트가 지원하는 추가 기능, streaming 지원
        skills=[greeting_skill],
        supports_authentication_extended_card=False #인증 된 에이전트 확장 카드를 지원하는지 여부
    )
    return agent_card

def main():
    agent_card=create_agent_card()
    print(agent_card.model_dump_json())
if __name__ == "__main__":
    main()

"""
{
    "additionalInterfaces":null,
    "capabilities": { 
        "extensions":null,
        "pushNotifications":null,
        "stateTransitionHistory":null,
        "streaming":true
    },
    "defaultInputModes":["text"],
    "defaultOutputModes":["text"],
    "description":"A2A 프로토콜을 학습하기 위한 기본적인 Hello World 에이전트입니다.",
    "documentationUrl":null,
    "iconUrl":null,
    "name":"Basic Hello World Agent",
    "preferredTransport":"JSONRPC",
    "protocolVersion":"0.3.0",
    "provider":null,
    "security":null,
    "securitySchemes":null,
    "signatures":null,
    "skills":[{
        "description":"간단한 인사와 기본적인 대화를 제공합니다.",
        "examples":
            ["안녕하세요","hello","hi","고마워요"],
        "id":"basic_greeting",
        "inputModes":["text"],
        "name":"Basic Greeting",
        "outputModes":["text"],
        "security":null,
        "tags":["greeting","hello","basic"]}],
        "supportsAuthenticatedExtendedCard":null,
        "url":"http://localhost:9999/",
        "version":"1.0.0"}
"""