from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-nomic-embed-text-v1.5",
    api_key=os.getenv("LMS_API_KEY"),
    base_url="http://192.168.45.167:50505/v1" # /embeddings 포함
)

words = ["강아지", "고양이", "토끼", "비행기"] 
words_embeddings = embeddings.embed_documents(words)


query = "동물"
query_embedding = embeddings.embed_query(input=query)

#cosine similarity
def consine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-9) # 작은 값 추가로 0 나누기 방지 

print(f'{query}에 대한 유사도: ')
for word, word_embedding in zip(words, words_embeddings):
    similarity = consine_similarity(query_embedding, word_embedding)
    print(f'{word}: {similarity:.4f}')