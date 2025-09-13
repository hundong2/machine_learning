from ua.template.ollama_embedding import get_ollama_embeddings_model
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ua.chap03.googlelangchainExample import get_model

embeddings = get_ollama_embeddings_model()

text_splitter = CharacterTextSplitter(separator=".", chunk_size=50, chunk_overlap=20)
# 문서 특성에 따라 separator를 적절하게 설정해야 한다. 기본값은 "\n\n"
#chunk_size는 문서의 길이에 따라 적절하게 설정 필요
#chunk_overlap은 문서의 중복되는 부분을 설정, 일반적으로 0으로 설정
#chunk_overlap은 문맥을 파악하는 용도로 사용 됨. 너무 높게 설정하면 중복된 정보가 많아져 비효율적일 수 있음.
documents=[
    Document(page_content="파이썬은 배우기 쉬운 프로그래밍 언어입니다." \
    "다양한 분야에서 활용되며, 특히 데이터 과학과 AI개발에 인기가 많습니다.", metadata={"source": "python_intro.txt", "topic": "programming"}),
    Document(page_content="자바스크립트는 웹브라우저에서 실행되는 프로그래밍 언어로 시작했지만, " \
    "현재는 서버 사이드 개발에도 널리 사용됩니다. Node.js가 대표적입니다.", metadata={"source": "javascript_intro.txt", "topic": "programming"}),
    Document(page_content="머신러닝은 데이터에서 패턴을 학습하는 AI의 한 분야 입니다." \
    "지도학습 비지도학습 강화 학습 등 다양한 방법론이 있습니다.", metadata={"source": "ml_intro.txt", "topic": "AI"}),
]

# seperator documents 
split_docs = text_splitter.split_documents(documents)
for doc in split_docs:
    print(f"document: {doc.page_content[:50]}... | reference: {doc.metadata['source']} | 주제: {doc.metadata['topic']}")

retriever = FAISS.from_documents(split_docs, embeddings).as_retriever(search_type="similarity",
                                                                      search_kwargs={"k": 2}) #의미적으로 가까운 문서들을 벡터화하여 저장
#vector store를 retriever로 변환, search_type은 similarity, 
# mmr(maximum marginal relevance), hybrid, similarity_score_threshold 등이 있음. 기본값은 similarity
# search_kwargs는 retriever의 검색 옵션을 지정, k는 반환할 유사한 문서의 개수
# mmr일 경우 lambda_mult는 다양성 제어, hybrid일 경우 alpha는 두 벡터의 가중치 조절, similarity_score_threshold는 유사도 점수 임계값 설정

results = retriever.get_relevant_documents("초보자가 배우기 좋은 프로그래밍 언어는?")
for i, doc in enumerate(results,1):
    print(f"{i}\n{doc.page_content[:100]}\n(source: {doc.metadata['source']}\ntopic: {doc.metadata['topic']})")

# retriever를 사용한 LLM 호출 
llm = get_model()

message="""
질문에 대한 답변을 작성할 때, 리트리버에서 가져온 문서를 참고하여 답변을 작성해 주세요. 
질문:
{question}

참고:
{context}
"""

# ChatPromptTemplate 생성: "human" 역할의 메시지로 프롬프트 템플릿을 만듭니다.
# 이 템플릿은 나중에 "context"와 "question" 변수를 받아 완전한 프롬프트를 생성합니다.
prompt = ChatPromptTemplate.from_messages([("human", message)])

# LCEL(LangChain Expression Language)을 사용하여 체인을 구성합니다.
# {"context": retriever, "question": RunnablePassthrough()}:
# - "context": retriever 객체를 할당. 체인이 실행될 때 retriever가 입력 질문을 받아 관련 문서를 검색합니다.
# - "question": RunnablePassthrough()를 할당. RunnablePassthrough는 입력 데이터를 그대로 통과시키는 역할을 합니다.
#   즉, 사용자의 입력 질문을 변경하지 않고 다음 단계(prompt)로 전달합니다.
#   만약 RunnablePassthrough가 없었다면, "question" 키에 특정 값을 할당해야 하지만,
#   여기서는 사용자의 입력 질문을 그대로 사용하기 위해 RunnablePassthrough를 사용합니다.
# | prompt: 앞 단계에서 생성된 {"context": 검색된 문서들, "question": 입력 질문}을 prompt 템플릿에 채워 완전한 프롬프트를 만듭니다.
# | llm: 완전한 프롬프트를 LLM에 전달하여 응답을 생성합니다.
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = chain.invoke("초보자가 배우기 좋은 프로그래밍 언어는?")
print("\n\n질문에 대한 답변:")
print(response.content)

"""
질문에 대한 답변:
제공된 문서를 참고하면, 초보자가 배우기 좋은 프로그래밍 언어는 **파이썬(Python)**입니다.

문서에 따르면 "파이썬은 배우기 쉬운 프로그래밍 언어"라고 명시되어 있습니다.
"""