from langchain.prompts import PromptTemplate, load_prompt
import os

current_dir_path = os.path.dirname(os.path.abspath(__file__))

file_prompt = load_prompt(f'{current_dir_path}/prompt_template.yaml')
print(file_prompt.format(context="Langchain is a framework for developing applications powered by language models.", question="what's langchain?"))

"""
컨텍스트: Langchain is a framework for developing applications powered by language models.
질문: what's langchain?
답변: 
"""