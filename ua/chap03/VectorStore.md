### PostgreSQL과 Elasticsearch 연동 예제

PostgreSQL과 Elasticsearch를 Ollama Embedding API와 연동하는 예제는 벡터를 생성하고 저장하는 방식에서 차이가 있습니다.

#### 1\. PostgreSQL (pgvector 사용)

PostgreSQL은 **pgvector** 확장 기능을 통해 벡터를 저장하고 검색할 수 있습니다. 이는 RDBMS의 안정성과 ACID 트랜잭션 기능을 유지하면서 벡터 검색을 할 수 있다는 장점이 있습니다.

**설치 및 설정**:

1.  **pgvector 확장 설치**: PostgreSQL에 `pgvector` 확장을 설치해야 합니다.
2.  **테이블 생성**: 벡터를 저장할 테이블을 생성합니다.

<!-- end list -->

```sql
-- pgvector 확장 설치 (한 번만 실행)
CREATE EXTENSION vector;

-- 데이터와 벡터를 저장할 테이블 생성
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1024) -- ollama 모델에 따라 차원(dimension)을 맞춰야 함
);
```

**Python 연동 예제**:

```python
import ollama
import psycopg2

# Ollama를 사용하여 텍스트를 벡터로 변환
def get_embedding(text):
    response = ollama.embeddings(model='llama3', prompt=text)
    return response['embedding']

# PostgreSQL에 연결
conn = psycopg2.connect(
    host="localhost",
    database="your_db",
    user="your_user",
    password="your_password"
)
cur = conn.cursor()

# 텍스트 데이터와 벡터 저장
text_data = "Ollama는 로컬에서 실행되는 언어 모델입니다."
embedding = get_embedding(text_data)
cur.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    (text_data, str(embedding))
)
conn.commit()

# 유사도 검색 (코사인 유사도)
query_text = "로컬 LLM"
query_embedding = get_embedding(query_text)
cur.execute(
    "SELECT content FROM documents ORDER BY embedding <-> %s LIMIT 5",
    (str(query_embedding),)
)
results = cur.fetchall()
print("PostgreSQL 검색 결과:")
for row in results:
    print(row[0])

cur.close()
conn.close()
```

-----

#### 2\. Elasticsearch (Dense Vector 필드 사용)

Elasticsearch는 **dense\_vector** 필드 타입을 사용하여 벡터를 저장하고, `script_score`나 KNN(k-Nearest Neighbor) 검색을 통해 유사도 검색을 수행합니다. 이는 로그 분석이나 전문 검색과 결합하여 사용할 때 유리합니다.

**설치 및 설정**:

1.  **인덱스 생성**: 벡터 필드 타입을 정의한 인덱스를 생성합니다.

<!-- end list -->

```json
PUT /my_documents
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text"
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

**Python 연동 예제**:

```python
import ollama
from elasticsearch import Elasticsearch

# Ollama를 사용하여 텍스트를 벡터로 변환
def get_embedding(text):
    response = ollama.embeddings(model='llama3', prompt=text)
    return response['embedding']

# Elasticsearch에 연결
es = Elasticsearch("http://localhost:9200")

# 텍스트 데이터와 벡터 저장
doc = {
    "content": "Ollama는 로컬에서 실행되는 언어 모델입니다.",
    "embedding": get_embedding("Ollama는 로컬에서 실행되는 언어 모델입니다.")
}
es.index(index="my_documents", document=doc)

# 유사도 검색 (KNN 검색)
query_text = "로컬 LLM"
query_embedding = get_embedding(query_text)
search_results = es.search(index="my_documents", body={
    "knn": {
        "field": "embedding",
        "query_vector": query_embedding,
        "k": 5,
        "num_candidates": 10
    },
    "_source": ["content"]
})

print("Elasticsearch 검색 결과:")
for hit in search_results['hits']['hits']:
    print(hit['_source']['content'])
```

-----

### FAISS 상세 설명

\*\*FAISS(Facebook AI Similarity Search)\*\*는 **효율적인 유사성 검색과 클러스터링을 위한 라이브러리**입니다. 메타(Meta)에서 개발했으며, 특히 대규모 벡터 집합(수십억 개 이상)에 대해 빠른 속도로 \*\*근접 이웃 탐색(Approximate Nearest Neighbor, ANN)\*\*을 수행하도록 최적화되어 있습니다.

FAISS는 단독으로 데이터베이스 역할을 하는 것이 아니라, **벡터 인덱스를 생성하고 관리**하는 라이브러리입니다. 따라서 자체적인 영속성(Persistence)이나 서버 기능이 없으므로, 일반적으로 로컬 메모리에 데이터를 로드하거나 다른 시스템과 통합하여 사용합니다.

**주요 특징**:

  - **다양한 인덱스 유형**: HNSW, IVFFlat 등 다양한 ANN 알고리즘을 지원하여 데이터의 특성과 성능 요구사항에 맞게 선택할 수 있습니다.
  - **메모리 효율성**: 벡터를 압축하는 기술(예: Product Quantization)을 사용하여 메모리 사용량을 줄입니다.
  - **높은 성능**: C++로 작성되어 매우 빠르며, GPU를 활용한 병렬 처리를 지원합니다.

FAISS는 "FAISS는 유사성 검색을 위한 도구 모음"으로 이해할 수 있습니다. 데이터의 양이 매우 많을 때, 빠르고 효율적인 검색을 위해 사용됩니다.

-----

### 인기 있는 Vector Store 3가지 추천

1.  **Weaviate**: Weaviate는 **그래프 기반의 벡터 검색**을 제공하는 오픈 소스 Vector DB입니다.
      - **특징**: 스키마(Schema)를 통해 데이터를 모델링하고, 벡터와 함께 메타데이터를 저장하여 필터링과 벡터 검색을 결합한 하이브리드 검색이 강력합니다. 자체적으로 **`text2vec`** 같은 모듈을 내장하고 있어, 별도의 임베딩 서버 없이도 데이터를 바로 벡터화할 수 있습니다.
      - **예제**: `Weaviate`는 **`ollama` 모듈**을 내장하고 있어, 간단한 설정으로 `ollama` 모델을 사용하여 데이터를 벡터화하고 검색할 수 있습니다.
2.  **Milvus**: Milvus는 **대규모 벡터 데이터에 특화된 오픈 소스 Vector DB**입니다.
      - **특징**: 분산 시스템으로 설계되어 수십억 개의 벡터를 효율적으로 관리할 수 있습니다. 다양한 인덱스 타입과 복잡한 필터링 쿼리를 지원하며, 엔터프라이즈 환경에서 많이 사용됩니다.
      - **예제**: Milvus는 **`PyMilvus`** 라이브러리를 통해 파이썬에서 쉽게 연동할 수 있습니다.
3.  **Pinecone**: Pinecone은 **매니지드(Managed) 서비스** 형태의 Vector DB입니다.
      - **특징**: 사용자가 직접 인프라를 구축하고 관리할 필요 없이 API를 통해 벡터를 업로드하고 검색할 수 있습니다. 뛰어난 확장성과 사용 편의성을 제공하며, 엔터프라이즈 환경에서 빠르게 프로덕션을 구축할 때 유용합니다.
      - **예제**: `Pinecone`은 **`pinecone-client`** 라이브러리를 통해 사용합니다.

-----

**예제**: Pinecone과 Ollama 연동 예제

```python
import ollama
from pinecone import Pinecone, ServerlessSpec

# Pinecone API 키와 환경 설정
api_key = "YOUR_API_KEY"
pc = Pinecone(api_key=api_key)

index_name = "ollama-index"
if index_name not in pc.list_indexes().names:
    pc.create_index(
        name=index_name,
        dimension=1024, # Ollama 모델 차원
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# 텍스트를 벡터로 변환
def get_embedding(text):
    response = ollama.embeddings(model='llama3', prompt=text)
    return response['embedding']

# 데이터 업로드
documents = [
    {"id": "doc1", "text": "Ollama는 로컬에서 실행되는 언어 모델입니다."},
    {"id": "doc2", "text": "AI 모델을 쉽게 배포하고 실행할 수 있습니다."}
]
vectors_to_upsert = []
for doc in documents:
    embedding = get_embedding(doc['text'])
    vectors_to_upsert.append(
        {"id": doc['id'], "values": embedding, "metadata": {"text": doc['text']}}
    )
index.upsert(vectors=vectors_to_upsert)

# 유사도 검색
query_text = "로컬 LLM"
query_embedding = get_embedding(query_text)
search_results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

print("Pinecone 검색 결과:")
for match in search_results['matches']:
    print(f"점수: {match['score']}, 내용: {match['metadata']['text']}")
```