from langchain.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "당신을 친절한 AI입니다\n질문: {question}\n답변:"
)
print(template.format(question="what's langchain?"))

"""
당신을 친절한 AI입니다
질문: what's langchain?
답변:
"""