from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_ollama_embeddings_model():
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
    return embeddings

