from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from googlelangchainExample import get_model

model = get_model()

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="explain that what's langchain?"),
    AIMessage(content="Langchain is a framework for developing applications powered by language models."),
    HumanMessage(content="recommand me what 'langchain_maintenance'")
]

result=model.invoke(messages)
print("response of AI: ", result.content)