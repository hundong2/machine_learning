from langchain_community.embeddings import OllamaEmbeddings
import os
import numpy as np

ollama_api_base = "http://192.168.45.167:11434"
ollama_api_key = "ollama"

# OpenAIEmbeddings 클래스 인스턴스 생성
embeddings = OllamaEmbeddings(
    base_url=ollama_api_base,
    model="nomic-embed-text"  # Ollama에 다운로드한 임베딩 모델 이름을 지정합니다.
)

words = ["강아지", "고양이", "토끼", "비행기"]
words_embeddings = embeddings.embed_documents(words)

query = "동물"
query_embedding = embeddings.embed_query(text=query)

# 코사인 유사도(Cosine Similarity) 계산 함수
# 두 벡터(vec1, vec2)의 방향이 얼마나 비슷한지를 측정하는 함수입니다.
# 값이 1에 가까울수록 두 벡터의 방향이 같고, -1에 가까울수록 반대 방향입니다.
# 임베딩 벡터의 유사도를 계산할 때 자주 사용됩니다.
def cosine_similarity(vec1, vec2):
    # np.dot(vec1, vec2): 두 벡터의 내적(dot product)을 계산합니다.
    # 내적은 두 벡터의 각 요소를 곱한 후 모두 더한 값으로,
    # 벡터의 방향이 얼마나 비슷한지를 수치적으로 표현합니다.
    dot_product = np.dot(vec1, vec2)
    
    # np.linalg.norm(vec1): vec1 벡터의 노름(norm)을 계산합니다.
    # 노름은 벡터의 크기(길이)를 나타내는 값으로,
    # 피타고라스 정리를 이용해 계산됩니다 (예: sqrt(x² + y²)).
    norm_vec1 = np.linalg.norm(vec1)
    
    # np.linalg.norm(vec2): vec2 벡터의 노름을 계산합니다.
    # vec1과 동일하게 vec2의 크기를 계산합니다.
    norm_vec2 = np.linalg.norm(vec2)
    
    # 코사인 유사도 공식: 내적을 두 벡터의 노름의 곱으로 나누어 계산합니다.
    # 1e-9는 매우 작은 값(0.000000001)으로, 분모가 0이 되는 것을 방지합니다.
    # (두 벡터 중 하나가 0벡터일 때 노름이 0이 되어 0으로 나누는 오류를 막기 위함)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-9) # 작은 값 추가로 0 나누기 방지

print(f"'{query}'에 대한 유사도: ")
for word, word_embedding in zip(words, words_embeddings):
    similarity = cosine_similarity(query_embedding, word_embedding)
    print(f"{word}: {similarity:.4f}")

"""
'동물'에 대한 유사도: 
강아지: 0.5491
고양이: 0.5465
토끼: 0.5420
비행기: 0.5565
"""