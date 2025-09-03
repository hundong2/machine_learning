from googlelangchainExample import get_model
from langchain_core.messages import HumanMessage

def example_code():
    model = get_model()

    result = model.invoke([HumanMessage(content="Hello, world!")])
    print(type(result))

if __name__=="__main__":
    example_code()