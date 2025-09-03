from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=['article', 'style'],
    template="다음 기사를 {style} 스타일로 요약해줘.\n기사: {article}"
)
print(template.format(article="Langchain is a framework for developing applications powered by language models.", style="친절한 AI"))