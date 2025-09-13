from langchain_community.tools import DuckDuckGoSearchResults
from ua.chap03.googlelangchainExample import get_model
from langchain_core.prompts import ChatPromptTemplate
import time

class RealTimeWebSearch:
    def __init__(self):
        self.search = DuckDuckGoSearchResults()
        self.llm = get_model()
        message ="""
        당신은 유능한 연구원입니다. 웹에서 검색한 최신 정보를 바탕으로 답변하세요
        검색 결과:
        {search_results}
        질문 {question}
        
        중요: 검색 결과에 있는 정보만 사용하여 답변하세요. 
        답변:"""
        self.qa_prompt = ChatPromptTemplate.from_messages(
            ("human", message)
        )
    def answer(self, question):
        """realtime web search"""
        print(f"Searching the web for: {question}")
        search_results = self.search.run(question)
        time.sleep(5) # 5 second delay to avoid rate limiting

        qa_chain = self.qa_prompt | self.llm
        answer = qa_chain.invoke({"search_results": search_results, "question": question})
        return answer

if __name__=="__main__":
    web_rag = RealTimeWebSearch()
    questions = ["오늘의 주요 뉴스는?", "오늘의 야구 순위는?", "최신 AI 기술 동향은?"]
    for q in questions:
        response = web_rag.answer(q)
        print(f"Question: {q}\nAnswer: {response.content}\n\n")