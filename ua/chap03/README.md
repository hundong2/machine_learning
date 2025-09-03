# langchain

## key point 

- prompt, vector store, output parser, llm, tools, memory, etc..
  - 위 요소들을 체인 형태로 연결하여 다양하고 복잡한 작업들은 단순하게 진행 

```sh
uv add "langchain[openai]"
uv add "langchain[google-genai]"
```

## key concept : message 

- 메세지는 contents와 role을 가지며 구조화된 정보를 통해 모델은 맥락을 파악, 적절한 응답을 만들어 낼 수 있다. 
- `BaseMessage`를 상속 받은 클래스들 이 있음. 
  - `SystemMessage` : type이 system인 메시지로써 시스템의 역할 지정, 페르소나 지저엥 사용 됨, 
  - `HumanMessage` : type이 Human인 메시지로써 사용자의 입력이나 질문
  - `AIMessage` : type이 ai인 메세지, 채팅 모델의 응답에 사용
  - `ToolMessage` : type이 tool인 메시지, 도구 호출의 결과를 AI에게 전달 될 떄 사용. 
- 이름 뒤에 `Chunk`가 있다면 streaming에 사용 한다는 의미. 
- reference : https://python.langchain.com/api_reference/core/messages.html

### PromptTemplate

- Prompt를 객체로 관리할 수 있도록 제공
- 동적으로 프롬프트를 생성하는 템플릿
- 변수를 포함하는 템플릿을 정의하고 실행 시점에 변수의 값들을 실젯값으로 채워서 프롬프트로 완성 가능
- 재사용성 증가, 유지보수성 향상 
- [runnable](./runnable.md) 이고 LCEL(pip line `|`)을 사용할 수 있다. 

| name | explain |
| :--- | :--- |
| **PromptTemplate** | 가장 기본적인 템플릿입니다. 하나의 문자열 프롬프트를 생성하며, `{변수}`를 사용해 동적인 값을 채워 넣습니다. (예: "Tell me about {topic}.") |
| **ChatPromptTemplate** | 채팅 모델을 위한 템플릿입니다. `SystemMessage`, `HumanMessage` 등 역할이 있는 메시지 목록을 생성하여 대화형 프롬프트 구성에 사용됩니다. |
| **FewShotPromptTemplate** | 모델에게 몇 가지 예시(few-shot)를 함께 제공하여 원하는 출력 형식이나 스타일을 학습시키는 템플릿입니다. 예시를 통해 모델의 응답 정확도를 높일 수 있습니다. |
| **PipelinePromptTemplate** | 여러 프롬프트 템플릿을 순차적으로 연결하여 최종 프롬프트를 만드는 템플릿입니다. 복잡한 프롬프트를 여러 단계로 나누어 구성할 때 유용합니다. |
| **MessagesPlaceholder** | `ChatPromptTemplate` 내에서 사용되며, 대화 기록(memory)과 같은 동적인 메시지 목록이 들어갈 자리를 지정합니다. 채팅 히스토리를 프롬프트에 주입할 때 필수적입니다. |

- [PromptTemplate](./promptTemplateEx.py). 
  - `from_template()` : 문자열로 바로 생성
  - `Constructor` : 변수 목록, 템플릿을 명시하여 세밀한 제어가 가능
  - `load_prompt` : 파일에서 템플릿을 불러올 수 있음. 
  - `partial_variable`: 일부 변수를 고정하여 하위-프롬프트 생성 가능 

## example code 

- [langchain example 1](./langchainExample.py)
- [langchain depth HumanMessage](./langchainHumanMessage.py)
- [langchain using conversation list](./langchain_chat_list.py)
- 