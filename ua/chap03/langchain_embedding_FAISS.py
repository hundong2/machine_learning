from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
# Ollama 서버의 엔드포인트를 지정합니다.
# Ollama는 기본적으로 11434 포트에서 실행되며,
# OpenAI 호환을 위해 /v1 경로를 추가합니다.
ollama_api_base = "http://192.168.45.167:11434"

# Ollama API에 대한 더미 API 키입니다. 실제 키는 필요하지 않습니다.
# OpenAIEmbeddings에서 키가 비어있으면 오류가 발생할 수 있으므로 임의의 값을 넣어줍니다.
ollama_api_key = "ollama"

# OpenAIEmbeddings 클래스 인스턴스 생성
embeddings = OllamaEmbeddings(
    base_url=ollama_api_base,
    model="nomic-embed-text"  # Ollama에 다운로드한 임베딩 모델 이름을 지정합니다.
)

texts =[
    "인공지능과 머신러닝의 차이점은 무엇일까?",
    "Python을 활용한 데이터 분석 예제.",
    "자연어 처리를 위한 딥러닝 기초.",
    "강아지와 고양이의 행동 차이.",
    "우주 탐사의 역사와 미래 전망.",
    "기후 변화와 환경 보호의 중요성.",
    "블록체인 기술의 원리와 응용 분야.",
    "인간의 뇌와 인공지능의 비교.",
    "빅데이터 분석을 위한 도구와 기법.",
    "로봇 공학의 최신 동향과 발전 방향.",
    "어리석은 자는 멀리서 행복을 찾고, 현명한 자는 가까이서 그것을 키운다.",
    "성공은 최선을 다한 결과이며, 실패는 최선을 다하지 않은 결과이다.",
    "삶은 우리가 만드는 것이며, 우리의 태도가 그것을 결정한다."
]

vectorstore = FAISS.from_texts(texts, embeddings)

query = "힘이 나는 명언 알려주세요."
docs = vectorstore.similarity_search(query, k=2) # k는 반환할 유사한 문서의 개수 2개를 반환 

print("검색 결과:")
for i, doc in enumerate(docs):
    print(f"{i+1}. {doc.page_content}")

#결괏값은 langchain.schema.Document로 page_content(본문)와 metadata(메타데이터) 속성을 가집니다.

"""
검색 결과:
1. 어리석은 자는 멀리서 행복을 찾고, 현명한 자는 가까이서 그것을 키운다.
2. 인간의 뇌와 인공지능의 비교.
"""