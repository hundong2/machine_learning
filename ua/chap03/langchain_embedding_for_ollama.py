# 필요한 라이브러리 설치
# pip install langchain_openai

#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
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

# 임베딩을 생성할 텍스트
text_list = ["인공지능과 머신러닝의 차이점은 무엇일까?", "Python을 활용한 데이터 분석 예제."]

# 문서 리스트에 대한 임베딩 생성
embedded_vectors = embeddings.embed_documents(text_list)

# 결과 확인
print(f"생성된 임베딩 벡터 개수: {len(embedded_vectors)}")
print("-" * 30)

# 첫 번째 텍스트의 임베딩 정보 출력
print(f"텍스트: '{text_list[0]}'")
print(f"임베딩 벡터 길이: {len(embedded_vectors[0])}")
print(f"임베딩 벡터 (일부): {embedded_vectors[0][:10]}...")
print("-" * 30)

# 두 번째 텍스트의 임베딩 정보 출력
print(f"텍스트: '{text_list[1]}'")
print(f"임베딩 벡터 길이: {len(embedded_vectors[1])}")
print(f"임베딩 벡터 (일부): {embedded_vectors[1][:10]}...")