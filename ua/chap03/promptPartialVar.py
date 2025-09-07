from langchain.prompts import PromptTemplate

base_prompt = PromptTemplate.from_template(
    "{text} 문장을 {language}로 번역해줘."
)

ko_prompt = base_prompt.partial( language="korean")
en_prompt = base_prompt.partial( language="english")

print(ko_prompt.format(text="hello"))
print(en_prompt.format(text="안녕하세요"))

"""
hello 문장을 korean로 번역해줘.
안녕하세요 문장을 english로 번역해줘.
"""
