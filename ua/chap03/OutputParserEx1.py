from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage

load_dotenv()

chat_model = ChatOpenAI(
    model="model-identifier",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1"
)

# Chat Prompt Template just using list messages 
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
    ("system", "당신은 까칠한 AI입니다. 사용자의 질문에 최대 3줄로 답해주세요."),
    ("human", "{question}"),
    ]
)

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="당신은 친절하고 상냥한 AI입니다. 사용자의 질문에 최대한 상세히 설명하세요."),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)



string_output_parser = StrOutputParser()

result: AIMessage = chat_model.invoke(
    chat_prompt_template.format_messages(
        question="파이썬에서 리스트를 정렬하는 방법은?"
    )
)
parsed_result: str = string_output_parser.parse(result)

print(parsed_result.content)
"""
파이썬에서 리스트를 정렬하려면 `my_list.sort()`(인제)나 `sorted(my_list)`(새 객체)를 사용하면 됩니다.  
필요하면 `key=`이나 `reverse=True` 옵션도 넣어줄 수 있어요.
"""

print("--------------------------------")
chain = chat_prompt_template | chat_model | string_output_parser
print(type(chain)) # <class 'langchain_core.runnables.base.RunnableSequence'>

result = chain.invoke({"question": "파이썬에서 딕셔너리를 정렬하는 방법은?"})

print(type(result)) #<class 'str'>
print(result)

"""
gpt-oss-20b-Q3_K_S Model ( local )
파이썬에서 딕셔너리를 정렬하려면:

```python
d = {'b':1,'a':2,'c':3}
sorted_items = sorted(d.items(), key=lambda kv: kv[0])   # 키 기준 정렬
# 또는 값 기준이면 key=lambda kv: kv[1]
```

`sorted()`는 리스트를 반환하므로 결과를 다시 딕셔너리로 변환할 수 있다.  
(필요 시 `collections.OrderedDict(sorted_items)` 사용)
"""