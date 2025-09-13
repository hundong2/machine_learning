from ua.template.ollama_embedding import get_ollama_embeddings_model
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

embeddings = get_ollama_embeddings_model()

text_spliter = CharacterTextSplitter(separator=".", chunk_size=50, chunk_overlap=20)
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
split_docs = text_spliter.split_documents(documents)
for doc in split_docs:
    print(f"document: {doc.page_content[:50]}... | reference: {doc.metadata['source']} | 주제: {doc.metadata['topic']}")

vectorstore = FAISS.from_documents(split_docs, embeddings) #의미적으로 가까운 문서들을 벡터화하여 저장

query = "초보자가 배우기 좋은 프로그래밍 언어는?"
results = vectorstore.similarity_search(query, k=2)
print("question:", query)
print("results:")
for i, res in enumerate(results,1):
    print(f"{i}\n{res.page_content[:100]}\n(source: {res.metadata['source']}\ntopic: {res.metadata['topic']})")

result_with_score = vectorstore.similarity_search_with_score(qeuery, k=2)
print("\n\n유사도 점수와 함께 결과 출력:")
for i, (res, score) in enumerate(result_with_score, 1):
    print(f"{i}\n{res.page_content[:100]}\n(source: {res.metadata['source']}\ntopic: {res.metadata['topic']})\nscore: {score}")