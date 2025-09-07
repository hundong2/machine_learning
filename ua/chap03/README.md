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
    - [from_template()](./promptTemplateEx.py)
  - `Constructor` : 변수 목록, 템플릿을 명시하여 세밀한 제어가 가능
    - [Constructor](./promptTemplateConstruct.py). 
  - `load_prompt` : 파일에서 템플릿을 불러올 수 있음. 
    - [load_prompt](./promptTemplateLoadEnv.py)
  - `partial_variable`: 일부 변수를 고정하여 하위-프롬프트 생성 가능 
    - [partial prompt](./promptPartialVar.py)

## Output Parser

- LLM 은 기본적으로 텍스트 문자열을 반환
- OutputParser 는 텍스트 응답을 애플리케이션에서 사용하기 편리한 구조화 된 데이터 ( example: JSON, pydantic )으로 변환하는 역할. 

| name | description |
| :--- | :--- |
| **StrOutputParser** | 모델의 응답(AIMessage)을 간단한 문자열(string)로 변환합니다. |
| **SimpleJsonOutputParser** | 모델이 생성한 JSON 형식의 문자열을 파이썬 딕셔너리로 변환합니다. |
| **PydanticOutputParser** | 모델의 응답을 미리 정의된 Pydantic 모델 객체로 변환하여, 데이터 유효성 검사와 타입 안전성을 보장합니다. |
| **CommaSeparatedListOutputParser** | 쉼표로 구분된 문자열(예: "사과, 바나나, 오렌지")을 파이썬 리스트로 변환합니다. |
| **DatetimeOutputParser** | 날짜/시간 형식의 문자열을 파이썬 `datetime` 객체로 변환합니다. |
| **XMLOutputParser** | 모델이 생성한 XML 형식의 문자열을 파이썬 딕셔너리나 객체로 변환합니다. |
| **MarkdownOutputParser** | 마크다운 형식의 텍스트에서 특정 코드 블록(예: ```python ... ```)을 추출합니다. |... ```)을 추출합니다. |

### Example of OutputParser

- [output parser using with prompt template](./OutputParserEx1.py)
  - langchain(`|`) 으로 선언 된 객체, 위 예시에서의 `chain` 변수는 (`RunnableSequence`)는 Runnable을 상속. 
  - `invoke()`실행 가능한 객체들은 모두 Runnable() -> Runnable은 모델의 실행, 배치, 스트리밍 등을 담당. 

### Example Json OutputParser 

- [Json Output Parser](./OutputParserEx2.py). 
  - `|` 파이프 연산자는 랭체인 표현 언어 LCEL의 기능 중 하나, 컴포넌트를 직관적으로 연결 해줌. 

### Runnable and LCEL

- Runnable 
- `LCEL(LangChain Expression Language)`
- Langchain의 모든 컴포넌트들을 표준화된 방식으로 연결할 수 있음. 

```python
# previous llm prompting example
def process_query(user_input):
  #1 step: generate prompt
  prompt = f"question : {user_input}\nresponse:"
  #2 setp: call LLM
  response = llm.generate(prompt)
  #3 step : parsing result
  parsed_result = parse_reponse(reponse)
  #4 step : post processing
  final_result = postprocess(parsed_result)
  return final_result
```

```python
#Using runnable processing 
chain = prompt_template | llm | parser | postprocessor
result = chain.invoke({"user_prompt: user_input"})
```

#### Runnable interface 

- 실행 가능한 무언가를 나타내는 추상 인터페이스 
- `input` -> `output` 이 있는 모든 것을 Runnable로 만들 수 있다. 

```python
#runnable interface 
from typing import Any, List, AsyncIterator, Iterator
from langchain_core.runnables import Runnable

#Runnable important methodes
class RunnableInterface:
    def invoke(self, input: Any) -> Any:
      """synchronous running"""
      pass
    async def ainvoke(self, input: Any) -> Any:
      """asynchrounous running"""
      pass
    def stream(self, input: Any) -> Iterator[Any]:
      """using streaming"""
      pass
    async def astream(self, input:Any) -> AsyncIterator[Any]:
      pass
    def batch(self, input: List[Any]) -> List[Any]:
      pass
```

- `Runnable`  Characteristic
  - 통일 된 인터페이스 : 모든 Runnable은 같은 메서드를 제공, 어떤 컴포넌트든 예측 가능한 방식으로 사용 가능 
  - 조합 가능성 : Runnable들은 `|` pipe 연산자로 쉽게 연결 가능 
  - 다양한 실행 모드 : 동기, 비동기, 스트리밍, 배치 처리를 모두 지원
  - 타입 안정성 : 입력과 출력 타입을 명시할 수 있어 타입 체커의 도움을 받을 수 있음. 

- [example Runnable lambda](./langchain_runnable_lamda.py). 
  - 일반 function을 runnable로 만들어 주는 예제 

#### LCEL ( langchain expression language )

- Runnable들을 조합하여 복잡한 로직을 만들 수 있게 하는 선억적 언어 
- `|` pipeline 

```python
chain = compoent1 | component2 | component3
```

- Unix shell의 파이프와 비슷하게 동작 
- 앞 컴포넌트의 출력이 다음 컴포넌트의 입력이 됨. 

##### pros

- 가독성 : 코드가 데이터의 흐름을 그대로 보여줌. 
- 재사용성 : 만든 체인은 그 자체로 Runnable 이므로, 다른 체인의 일부로 사용할 수 있음. 
- 유연성 : 실행 시점에 동적으로 체인을 구성하거나 수정할 수 있음. 
- [LCEL Example](./langchain_runnable_lcel.py). 

- runnable 주요 타입들 
  - [Runnable vairous type](./langchain_runnable_passthrough.py). 
  - [Runnable Lambda](./langchain_runnable_type-lambda.py). 
  - [Runnable Branch](./langchain_runnable_branch.py)  
    - RunnableBranch의 첫번째 파라미터로는 ( 조건 함수, 실행할 체인) 

### Langchain Tools

- langchain의 도구는 외부 세계와 상호 작용하는 강력한 기능 
- 가장 많이 도입 된 도구는 database 와 web browser 
- LLM을 제공해주는 업체에서 `function calling`이라는 기능으로 도구 사용을 제공하지만 `불편`

#### langchain tools using sequence 

1. Tool Creation : 사용할 도구를 준비 
2. Tool Building : 사용할 도구를 LLM에 연결 
3. Tool Calling : LLM이 사용할 도구를 선택
4. Tool Execution : LLM이 도구를 실행하여 작업 

- `@tool decorator`
  - [langchain tool decorator](./langchain_tool_decorator.py). 
  - 

## example code 

- [langchain example 1](./langchainExample.py)
- [langchain depth HumanMessage](./langchainHumanMessage.py)
- [langchain using conversation list](./langchain_chat_list.py)
- [pydantic example](./pydantic.md). 
- [Any](./Any.md). 
- 