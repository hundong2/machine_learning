# LangChain의 PromptTemplate, Runnable, 그리고 LCEL 이란?

`PromptTemplate`은 동적 프롬프트를 만드는 템플릿이고, LCEL은 이 템플릿과 모델 같은 구성요소들을 파이프(`|`)로 연결하는 방법이며, `Runnable`은 이렇게 연결될 수 있는 모든 객체의 표준 인터페이스입니다. 이 세 가지는 LangChain을 강력하고 유연하게 만드는 핵심 개념입니다.

## 예시 파일

아래 예시 코드는 `PromptTemplate`, 모델, 출력 파서를 LCEL을 사용해 연결하고, `Runnable` 객체로서 실행하는 전체 과정을 보여줍니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 1. PromptTemplate: 'topic'이라는 변수를 포함하는 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)

# 2. Model: 사용할 LLM 모델 초기화
model = ChatOpenAI()

# 3. Output Parser: 모델의 출력(AIMessage)을 간단한 문자열로 변환
output_parser = StrOutputParser()

# 4. LCEL을 사용하여 체인 생성
# PromptTemplate, Model, OutputParser는 모두 'Runnable' 객체이므로
# 파이프(|) 연산자로 간단하게 연결하여 새로운 체인을 만들 수 있습니다.
chain = prompt | model | output_parser

# 5. 체인 실행 (invoke)
# 생성된 'chain' 역시 'Runnable' 객체이므로 invoke, stream, batch 등의 메서드를 사용할 수 있습니다.
response = chain.invoke({"topic": "ice cream"})

print(response)

```

## 답변

### 1. PromptTemplate

`PromptTemplate`은 **재사용 가능한 프롬프트의 설계도**입니다. 단순히 문자열로 프롬프트를 만드는 대신, `{변수}`와 같은 플레이스홀더를 포함하는 템플릿을 만들어 실행 시점에 실제 값을 채워 넣을 수 있습니다.

-   **왜 필요한가?**: 매번 다른 주제나 내용으로 프롬프트를 만들어야 할 때, 코드의 중복을 줄이고 프롬프트의 구조를 일관되게 유지할 수 있어 유지보수가 매우 편리해집니다.

### 2. LCEL (LangChain Expression Language)

LCEL은 LangChain의 여러 구성요소(Component)들을 **선언적으로 연결하여 체인(Chain)을 만드는 방법**입니다. 파이썬의 파이프(`|`) 연산자를 사용하여 마치 데이터가 흘러가는 것처럼 직관적으로 체인을 구성할 수 있습니다.

-   **핵심 아이디어**: `첫 번째 단계 | 두 번째 단계 | 세 번째 단계`
-   **예시**: `prompt | model | output_parser`
    1.  `prompt`: 사용자 입력을 받아 `PromptTemplate`을 완성합니다.
    2.  `model`: 완성된 프롬프트를 LLM 모델에 전달합니다.
    3.  `output_parser`: 모델의 응답을 우리가 원하는 형식(예: 문자열)으로 변환합니다.

-   **왜 강력한가?**: LCEL로 만든 체인은 스트리밍, 배치(batch), 비동기(async) 실행을 별도의 복잡한 코드 없이 자동으로 지원합니다.

### 3. Runnable 객체

`Runnable`은 LCEL 체인을 구성하는 **모든 요소의 표준 인터페이스(규격)**입니다. `PromptTemplate`, `ChatOpenAI` 모델, `StrOutputParser` 등 체인에 연결될 수 있는 모든 것은 `Runnable`입니다.

-   **표준 메서드**: 모든 `Runnable` 객체는 다음과 같은 공통된 메서드를 가집니다.
    -   `invoke()`: 입력을 받아 결과를 반환 (단일 실행)
    -   `stream()`: 결과를 스트리밍 형태로 실시간 반환
    -   `batch()`: 여러 입력을 한 번에 처리
    -   `ainvoke()`, `astream()`, `abatch()`: 비동기 버전의 메서드들

-   **왜 중요한가?**: 모든 구성요소가 동일한 `Runnable` 규격을 따르기 때문에, 우리는 어떤 조합으로든 이들을 파이프(`|`)로 자유롭게 연결할 수 있습니다. 그리고 그 결과로 만들어진 **체인 자체도 또 하나의 `Runnable` 객체**가 되어, 동일한 표준 메서드로 실행할 수 있습니다. 이것이 LangChain의 유연성과 확장성의 핵심입니다.

### 추가 자료

-   [LangChain 공식 문서: LangChain Expression Language (LCEL)](https://python.langchain.com/v0.2/docs/concepts/#langchain-expression-language-lcel)
-   [LangChain 공식 문서: Prompts](https://python.langchain.com/v0.2/docs/concepts/#prompts)
-   [LangChain 공식 문서: The Runnable interface](https://python.langchain.com/v0.2/docs/concepts/#runnable-interface)