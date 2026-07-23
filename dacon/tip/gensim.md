# Gensim 실무 가이드

## 1. Gensim이란?

Gensim은 대규모 텍스트에서 **벡터 표현과 토픽을 학습하고 검색하는 데 특화된 NLP 라이브러리**다. 모든 데이터를 한꺼번에 메모리에 올리지 않고 반복 가능한 말뭉치(corpus)를 순차적으로 처리할 수 있다는 점이 강점이다.

주요 기능은 다음과 같다.

| 목적 | 대표 클래스 | 결과 |
|---|---|---|
| 단어 임베딩 | `Word2Vec` | 단어마다 고정 길이 벡터 |
| 미등록 단어 대응 임베딩 | `FastText` | 단어와 문자 n-gram 기반 벡터 |
| 문서 임베딩 | `Doc2Vec` | 문서마다 고정 길이 벡터 |
| 단어 빈도 기반 표현 | `Dictionary`, `TfidfModel` | 희소한 문서 벡터 |
| 토픽 모델링 | `LdaModel` | 문서의 토픽 분포와 토픽별 단어 |
| 벡터 검색 | `KeyedVectors`, `MatrixSimilarity` | 유사 단어 또는 유사 문서 |
| 공개 모델·데이터 다운로드 | `gensim.downloader` | GloVe 벡터, `text8` 등의 데이터 |

Gensim은 다음과 같은 업무에 특히 적합하다.

- 상품명, 검색어, 게시글에서 비슷한 단어 찾기
- 도메인 전용 단어 임베딩 학습
- 문서 분류 모델에 사용할 특징 벡터 생성
- 문서 군집화와 유사 문서 검색의 빠른 프로토타입
- 대규모 문서에서 주요 토픽 탐색
- 희소 행렬 기반 TF-IDF 검색

반면 문맥에 따라 같은 단어의 표현이 달라져야 하는 문제, 복잡한 질의응답, 생성형 모델이 필요한 문제에는 BERT나 GPT 계열의 Transformer가 더 적합하다. Gensim의 Word2Vec 벡터에서 `bank`는 어떤 문장에서도 기본적으로 하나의 벡터를 사용하지만, 문맥형 Transformer는 문장에 따라 다른 표현을 만든다.

> 공식 문서: [Gensim API Reference](https://radimrehurek.com/gensim/apiref.html)

---

## 2. 설치와 버전 확인

UV 프로젝트에서는 다음과 같이 설치한다.

```bash
uv add gensim
```

문서의 분류 예제까지 실행하려면 다음 패키지도 사용한다.

```bash
uv add scikit-learn
```

설치 결과를 확인한다.

```python
import gensim

print(gensim.__version__)
```

이 문서의 코드는 Gensim 4.x API를 기준으로 한다. 인터넷에 남아 있는 Gensim 3.x 예제와는 다음과 같은 차이가 있다.

| 과거 Gensim 3.x | 현재 Gensim 4.x |
|---|---|
| `size=100` | `vector_size=100` |
| `model["word"]` | `model.wv["word"]` |
| `model.wv.vocab` | `model.wv.key_to_index` |
| `model.wv.index2word` | `model.wv.index_to_key` |

---

## 3. Gensim이 기대하는 데이터 형태

Word2Vec과 FastText가 입력으로 기대하는 기본 형태는 다음과 같다.

```python
sentences = [
    ["machine", "learning", "is", "useful"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["machine", "learning", "uses", "data"],
]
```

즉, 전체 입력은 **문장들의 반복 가능한 객체**이고, 각 문장은 **토큰 문자열의 리스트**다.

잘못된 입력은 다음과 같다.

```python
# 문자열 하나를 그대로 전달하면 문자 단위로 처리될 가능성이 있다.
sentences = [
    "machine learning is useful",
    "deep learning uses neural networks",
]
```

최소한 다음과 같이 토큰화해야 한다.

```python
sentences = [
    sentence.split()
    for sentence in [
        "machine learning is useful",
        "deep learning uses neural networks",
    ]
]
```

영어의 간단한 실험에서는 `simple_preprocess()`를 사용할 수 있다.

```python
from gensim.utils import simple_preprocess

text = "Gensim makes large-scale NLP relatively simple!"
tokens = simple_preprocess(
    text,
    deacc=True,  # 악센트 기호 제거
    min_len=2,
    max_len=30,
)

print(tokens)
```

`simple_preprocess()`는 소문자화와 간단한 토큰화를 제공하지만 모든 언어와 도메인에 맞는 만능 전처리기는 아니다. 이메일 주소, 상품 코드, 해시태그, 이모지, 숫자가 중요한 서비스에서는 업무 규칙에 맞는 전처리가 필요하다.

---

## 4. Word2Vec

### 4.1 핵심 개념

Word2Vec은 주변 문맥이 비슷한 단어는 의미도 비슷하다는 분포 가설을 이용해 단어 벡터를 학습한다.

#### CBOW

주변 단어들로 가운데 단어를 예측한다.

$$
P(w_t \mid w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c})
$$

- 일반적으로 빠르고 자주 등장하는 단어에 안정적이다.
- Gensim에서는 `sg=0`으로 설정한다.

#### Skip-gram

가운데 단어로 주변 단어를 예측한다.

$$
\sum_{-c \leq j \leq c,\;j \neq 0}
\log P(w_{t+j} \mid w_t)
$$

- 희귀 단어 표현에 유리한 경우가 많다.
- 계산량은 CBOW보다 큰 편이다.
- Gensim에서는 `sg=1`로 설정한다.

실무에서는 전체 vocabulary에 대한 softmax를 매번 계산하는 대신 negative sampling을 자주 사용한다. 관측된 단어 쌍의 점수는 높이고 임의로 뽑은 음성 단어 쌍의 점수는 낮추도록 학습한다.

$$
\log \sigma(v_{w_o}^{\mathsf T}v_{w_i})
+
\sum_{k=1}^{K}
\log \sigma(-v_{w_k}^{\mathsf T}v_{w_i})
$$

여기서 $w_i$는 중심 단어, $w_o$는 실제 주변 단어, $w_k$는 음성 표본, $K$는 `negative` 값이다.

### 4.2 기본 학습 예제

아래 예제는 API 확인용 작은 데이터다. 의미 있는 임베딩을 만들려면 훨씬 많은 문장이 필요하다.

```python
from gensim.models import Word2Vec

sentences = [
    ["cat", "is", "a", "small", "animal"],
    ["dog", "is", "a", "friendly", "animal"],
    ["cat", "and", "dog", "are", "pets"],
    ["tiger", "is", "a", "large", "animal"],
    ["lion", "and", "tiger", "are", "wild", "animals"],
    ["puppy", "is", "a", "young", "dog"],
    ["kitten", "is", "a", "young", "cat"],
]

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    sg=1,
    negative=10,
    epochs=50,
    seed=42,
)
```

### 4.3 주요 하이퍼파라미터

| 파라미터 | 의미 | 실무 판단 기준 |
|---|---|---|
| `vector_size` | 임베딩 차원 | 100~300부터 검증. 크면 표현력과 메모리 사용량이 함께 증가 |
| `window` | 좌우 문맥 범위 | 구문 관계는 작게, 주제 유사성은 상대적으로 크게 시작 |
| `min_count` | 최소 출현 빈도 | 오타와 노이즈 제거. 작은 말뭉치는 낮추고 대규모 말뭉치는 높임 |
| `sg` | `0`: CBOW, `1`: Skip-gram | 속도는 CBOW, 희귀 단어는 Skip-gram부터 비교 |
| `negative` | 음성 표본 수 | 보통 5~20 범위에서 검증 |
| `sample` | 고빈도 단어 다운샘플링 | 조사·관사처럼 지나치게 흔한 단어의 영향 조절 |
| `epochs` | 말뭉치 반복 횟수 | 작은 데이터는 늘릴 수 있으나 과적합과 시간 확인 |
| `workers` | 병렬 작업 수 | 속도는 증가하지만 완전한 재현성이 어려워질 수 있음 |
| `seed` | 난수 시드 | 실험 재현을 돕지만 멀티스레드에서는 이것만으로 충분하지 않음 |

### 4.4 벡터와 유사 단어 조회

학습 알고리즘을 포함한 전체 모델은 `model`, 조회용 단어 벡터는 `model.wv`에 있다.

```python
# 단어가 vocabulary에 있는지 확인
if "cat" in model.wv:
    cat_vector = model.wv["cat"]
    print(cat_vector.shape)

# 두 단어의 코사인 유사도
similarity = model.wv.similarity("cat", "dog")
print(similarity)

# 가장 가까운 단어
similar_words = model.wv.most_similar(
    "cat",
    topn=5,
)

for word, score in similar_words:
    print(word, score)
```

코사인 유사도는 다음과 같다.

$$
\operatorname{cosine}(a,b)
=
\frac{a \cdot b}
{\lVert a \rVert_2 \lVert b \rVert_2}
$$

값이 1에 가까울수록 방향이 비슷하다. 다만 높은 코사인 유사도가 반드시 사람이 판단하는 동의어 관계를 뜻하지는 않는다. 같은 문맥에 함께 등장하는 반의어도 가까워질 수 있다.

벡터 연산도 가능하다.

```python
result = model.wv.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=5,
)
```

이 코드는 해당 단어들이 vocabulary에 있고 학습 데이터가 충분할 때 의미가 있다.

---

## 5. 공개 사전학습 임베딩 사용

직접 학습할 말뭉치가 부족하면 Gensim Downloader로 공개 데이터나 임베딩을 불러올 수 있다.

```python
import gensim.downloader as api

# 사용할 수 있는 데이터와 모델 이름 확인
available = api.info(name_only=True)
print(available)

# 비교적 작은 공개 GloVe 벡터
vectors = api.load("glove-twitter-25")

print(vectors["computer"])
print(vectors.most_similar("computer", topn=5))
```

첫 실행에서는 파일을 다운로드하고 이후에는 로컬 캐시를 사용한다. 기본 캐시 위치와 데이터별 라이선스·크기는 배포 환경에 들어가기 전에 확인해야 한다.

데이터셋도 반복 가능한 형태로 불러올 수 있다.

```python
import gensim.downloader as api
from gensim.models import Word2Vec

text8 = api.load("text8")

model = Word2Vec(
    sentences=text8,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=5,
)
```

> 공식 문서: [Gensim Downloader API](https://radimrehurek.com/gensim/downloader.html)

### 사전학습 벡터를 사용할 때 확인할 사항

1. **언어와 도메인**: 일반 뉴스 임베딩이 의료 약어나 사내 상품 코드를 잘 표현한다는 보장은 없다.
2. **전처리 일치 여부**: 대소문자, 공백, 구두점 처리 방식이 vocabulary 적중률을 바꾼다.
3. **OOV 비율**: 실제 데이터에서 vocabulary에 없는 단어 비율을 측정한다.
4. **라이선스**: 원본 말뭉치와 배포 벡터의 사용 조건을 모두 확인한다.
5. **다운로드 의존성**: 운영 서버가 시작할 때 인터넷에서 모델을 받지 않도록 빌드 단계에서 고정한다.

OOV 비율은 다음과 같이 측정할 수 있다.

```python
def calculate_oov_rate(tokenized_documents, keyed_vectors):
    tokens = [
        token
        for document in tokenized_documents
        for token in document
    ]

    if not tokens:
        return 0.0

    oov_count = sum(
        token not in keyed_vectors
        for token in tokens
    )

    return oov_count / len(tokens)
```

---

## 6. FastText: 미등록 단어에 더 강한 임베딩

Word2Vec은 vocabulary에 없는 단어를 바로 조회할 수 없다.

```python
# 등록되지 않은 단어이면 KeyError 발생
vector = model.wv["unseen_word"]
```

FastText는 단어를 문자 n-gram으로 나누어 학습한다. 예를 들어 `playing`의 일부 정보는 `<pl`, `pla`, `lay`, `ayi`, `yin`, `ing` 같은 부분 문자열에서 얻는다. 따라서 학습 때 보지 못한 단어도 알려진 문자 조각을 조합해 벡터를 만들 수 있다.

```python
from gensim.models import FastText

sentences = [
    ["play", "playing", "player", "game"],
    ["run", "running", "runner", "race"],
    ["walk", "walking", "walker", "road"],
]

fasttext_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=3,
    max_n=6,
    workers=1,
    sg=1,
    epochs=50,
    seed=42,
)

# 정확히 학습하지 않은 단어라도 문자 n-gram으로 벡터 추론 가능
vector = fasttext_model.wv["playfully"]
print(vector.shape)
```

주의할 점은 벡터를 만들 수 있다는 사실이 의미 품질을 보장하지는 않는다는 것이다. 오타나 형태 변화에는 도움이 되지만, 전혀 다른 의미의 문자열이 우연히 비슷한 문자 조각을 가지면 부정확한 결과가 나올 수 있다.

FastText가 적합한 경우:

- 활용과 굴절이 많은 언어
- 오탈자가 자주 등장하는 리뷰와 검색어
- 상품 코드처럼 변형이 많은 문자열
- 학습 후에도 신규 단어가 계속 유입되는 서비스

> 공식 튜토리얼: [FastText Model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html)

---

## 7. 실무 예제: 평균 Word2Vec 벡터로 문서 분류

Word2Vec은 단어 벡터를 만들기 때문에 문서 분류 모델에 넣으려면 문서 하나를 고정 길이 벡터로 변환해야 한다. 가장 단순한 기준선은 문서에 포함된 단어 벡터의 평균이다.

$$
v_{\text{document}}
=
\frac{1}{N}
\sum_{i=1}^{N} v_{w_i}
$$

### 7.1 데이터 준비와 임베딩 학습

```python
import numpy as np

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


train_texts = [
    "The delivery was fast and the product was excellent",
    "I love this product and would buy it again",
    "The quality is great and I am very satisfied",
    "The package arrived broken and late",
    "This product is terrible and disappointing",
    "The quality is poor and I want a refund",
]
train_labels = [1, 1, 1, 0, 0, 0]

test_texts = [
    "Fast delivery and great quality",
    "The item was broken and terrible",
]
test_labels = [1, 0]

tokenized_train = [
    simple_preprocess(text)
    for text in train_texts
]

tokenized_test = [
    simple_preprocess(text)
    for text in test_texts
]

embedding_model = Word2Vec(
    sentences=tokenized_train,
    vector_size=50,
    window=3,
    min_count=1,
    workers=1,
    sg=1,
    epochs=100,
    seed=42,
)
```

엄격한 실험에서는 임베딩도 훈련 데이터만 사용해 학습한다. 검증 또는 테스트 문장을 임베딩 학습에 포함하면 정답 라벨을 직접 쓰지 않더라도 데이터 분포 정보가 새어 들어갈 수 있다.

### 7.2 문서 벡터 변환 함수

```python
def mean_embedding(tokens, keyed_vectors):
    valid_vectors = [
        keyed_vectors[token]
        for token in tokens
        if token in keyed_vectors
    ]

    if not valid_vectors:
        return np.zeros(
            keyed_vectors.vector_size,
            dtype=np.float32,
        )

    return np.mean(
        valid_vectors,
        axis=0,
    )


X_train = np.vstack([
    mean_embedding(tokens, embedding_model.wv)
    for tokens in tokenized_train
])

X_test = np.vstack([
    mean_embedding(tokens, embedding_model.wv)
    for tokens in tokenized_test
])
```

### 7.3 분류기 학습과 평가

```python
classifier = LogisticRegression(
    max_iter=1_000,
    random_state=42,
)

classifier.fit(
    X_train,
    train_labels,
)

predictions = classifier.predict(X_test)

print(
    classification_report(
        test_labels,
        predictions,
        zero_division=0,
    )
)
```

평균 벡터는 단어 순서를 잃고 모든 단어를 같은 중요도로 취급한다는 한계가 있다. 그래도 학습과 추론이 빠르므로 TF-IDF, 선형 분류기와 함께 좋은 기준선이 된다.

개선 방법으로는 다음을 검토할 수 있다.

- 불용어를 제거하거나 업무상 중요한 단어는 유지
- 단순 평균 대신 TF-IDF 가중 평균 사용
- FastText로 OOV 문제 완화
- 문장 길이, OOV 비율 등의 보조 특징 추가
- 더 복잡한 문맥 이해가 필요하면 Transformer 기준선과 비교

---

## 8. Doc2Vec: 문서 자체의 벡터 학습

Doc2Vec은 각 문서에 고유 태그를 붙이고 단어와 문서 표현을 함께 학습한다.

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

documents = [
    "Machine learning learns patterns from data.",
    "Deep learning uses multilayer neural networks.",
    "The football team won the championship.",
    "The player scored a goal in the match.",
]

tagged_documents = [
    TaggedDocument(
        words=simple_preprocess(text),
        tags=[f"DOC_{index}"],
    )
    for index, text in enumerate(documents)
]

doc_model = Doc2Vec(
    documents=tagged_documents,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    epochs=50,
    seed=42,
)
```

학습 문서의 벡터는 태그로 조회한다.

```python
document_vector = doc_model.dv["DOC_0"]
print(document_vector.shape)

similar_documents = doc_model.dv.most_similar(
    "DOC_0",
    topn=3,
)
print(similar_documents)
```

새 문서는 `infer_vector()`로 벡터를 추론한다.

```python
new_tokens = simple_preprocess(
    "Neural networks learn useful data representations."
)

new_vector = doc_model.infer_vector(
    new_tokens,
    epochs=30,
)

similar_documents = doc_model.dv.most_similar(
    [new_vector],
    topn=3,
)

print(similar_documents)
```

`infer_vector()`는 최적화 과정을 수행하므로 단순 벡터 조회보다 느리다. 실시간 서비스에서는 자주 사용하는 문서 벡터를 미리 계산해 저장하는 것이 좋다.

> 공식 문서: [Doc2Vec](https://radimrehurek.com/gensim/models/doc2vec.html)

---

## 9. TF-IDF와 유사 문서 검색

문서에 등장한 단어가 중요한 정도를 간단하고 설명 가능하게 표현하려면 TF-IDF가 유용하다.

역문서 빈도는 흔히 다음과 같이 정의한다.

$$
\operatorname{IDF}(t)
=
\log
\frac{N}
{\operatorname{DF}(t)}
$$

$N$은 전체 문서 수이고, $\operatorname{DF}(t)$는 단어 $t$가 등장한 문서 수다. 여러 문서에 흔하게 등장하는 단어의 가중치는 낮아지고 특정 문서에 집중된 단어는 높아진다.

```python
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from gensim.utils import simple_preprocess

documents = [
    "machine learning uses data",
    "deep learning uses neural networks",
    "football players score goals",
    "baseball players hit the ball",
]

tokenized_documents = [
    simple_preprocess(text)
    for text in documents
]

# 단어와 정수 ID의 대응 관계
dictionary = Dictionary(tokenized_documents)

# 지나치게 희귀하거나 흔한 단어 제거
dictionary.filter_extremes(
    no_below=1,
    no_above=0.8,
)

# 문서를 (token_id, count) 형태의 BoW로 변환
bow_corpus = [
    dictionary.doc2bow(tokens)
    for tokens in tokenized_documents
]

tfidf_model = TfidfModel(bow_corpus)
tfidf_corpus = list(tfidf_model[bow_corpus])

# 작은 말뭉치용 인메모리 유사도 인덱스
index = MatrixSimilarity(
    tfidf_corpus,
    num_features=len(dictionary),
)
```

새 질의와 가까운 문서를 검색한다.

```python
query = "neural machine learning"
query_tokens = simple_preprocess(query)
query_bow = dictionary.doc2bow(query_tokens)
query_tfidf = tfidf_model[query_bow]

scores = index[query_tfidf]

ranked_results = sorted(
    enumerate(scores),
    key=lambda item: item[1],
    reverse=True,
)

for document_index, score in ranked_results:
    print(
        f"{score:.4f}",
        documents[document_index],
    )
```

실제 검색 시스템에서는 문서 ID와 원문 또는 데이터베이스 키의 대응 관계도 반드시 같은 순서로 관리해야 한다. 문서가 추가·삭제되면 인덱스와 메타데이터의 정렬이 어긋나지 않도록 버전을 함께 갱신한다.

---

## 10. LDA 토픽 모델링

LDA는 문서를 여러 토픽의 혼합으로, 토픽을 여러 단어의 확률 분포로 본다.

$$
P(w \mid d)
=
\sum_{k=1}^{K}
P(w \mid z=k)
P(z=k \mid d)
$$

여기서 $K$는 토픽 수, $P(z=k \mid d)$는 문서 $d$의 토픽 분포, $P(w \mid z=k)$는 토픽 $k$의 단어 분포다.

```python
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

documents = [
    "machine learning model data prediction",
    "deep neural network model training",
    "artificial intelligence data algorithm",
    "football team player match goal",
    "baseball team player game ball",
    "sports match championship player",
]

tokenized_documents = [
    simple_preprocess(text)
    for text in documents
]

dictionary = Dictionary(tokenized_documents)
dictionary.filter_extremes(
    no_below=1,
    no_above=0.8,
)

corpus = [
    dictionary.doc2bow(tokens)
    for tokens in tokenized_documents
]

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    passes=20,
    iterations=100,
    random_state=42,
)

for topic_id, description in lda_model.print_topics(
    num_topics=-1,
    num_words=5,
):
    print(topic_id, description)
```

새 문서의 토픽 분포를 구한다.

```python
new_document = simple_preprocess(
    "neural network training data"
)
new_bow = dictionary.doc2bow(new_document)

topic_distribution = lda_model.get_document_topics(
    new_bow,
    minimum_probability=0.0,
)

print(topic_distribution)
```

토픽 번호 자체에는 의미가 없다. `topic 0`이 항상 스포츠라는 보장은 없으며 재학습하면 번호가 달라질 수 있다. 토픽에 이름을 붙이는 작업은 상위 단어와 대표 문서를 사람이 검토해 수행한다.

실무에서 토픽 수는 다음을 함께 고려해 결정한다.

- 토픽 일관성(coherence)
- 사람이 해석할 수 있는가
- 서로 중복되는 토픽이 지나치게 많지 않은가
- 시간에 따른 토픽 비중 변화가 안정적인가
- 최종 업무 목표에 실제 도움이 되는가

---

## 11. 대용량 데이터 스트리밍

Gensim의 중요한 장점은 모든 문장을 리스트로 만들지 않아도 된다는 점이다.

한 줄에 하나의 토큰화된 문장이 저장된 파일이 있다고 가정한다.

```text
machine learning uses data
deep learning uses neural networks
word embeddings represent semantic relationships
```

`LineSentence`를 사용하면 파일을 순차적으로 읽는다.

```python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence(
    "data/tokenized_corpus.txt"
)

model = Word2Vec(
    sentences=sentences,
    vector_size=200,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=5,
)
```

데이터베이스나 여러 파일을 읽는 경우 반복 가능한 클래스를 만들 수 있다.

```python
from pathlib import Path

from gensim.utils import simple_preprocess


class CorpusIterator:
    def __init__(self, data_directory):
        self.data_directory = Path(data_directory)

    def __iter__(self):
        for file_path in sorted(
            self.data_directory.glob("*.txt")
        ):
            with file_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line in file:
                    tokens = simple_preprocess(line)

                    if tokens:
                        yield tokens


sentences = CorpusIterator("data/corpus")
```

`__iter__()`가 호출될 때마다 처음부터 다시 읽을 수 있어야 한다. Word2Vec은 vocabulary 구축과 학습 과정에서 말뭉치를 여러 번 순회하기 때문이다. 한 번 소비하면 끝나는 generator 객체를 그대로 재사용하면 데이터가 비어 학습이 제대로 되지 않을 수 있다.

---

## 12. 모델 저장, 불러오기와 배포

### 12.1 추가 학습이 필요한 경우

전체 Word2Vec 모델을 저장한다.

```python
from gensim.models import Word2Vec

model.save(
    "artifacts/word2vec.model"
)

loaded_model = Word2Vec.load(
    "artifacts/word2vec.model"
)
```

전체 모델에는 단어 벡터뿐 아니라 추가 학습에 필요한 내부 상태도 포함된다.

### 12.2 조회와 추론만 필요한 경우

`KeyedVectors`만 저장하면 파일과 메모리 사용량을 줄일 수 있다.

```python
from gensim.models import KeyedVectors

model.wv.save(
    "artifacts/word_vectors.kv"
)

word_vectors = KeyedVectors.load(
    "artifacts/word_vectors.kv",
    mmap="r",
)
```

`mmap="r"`은 큰 벡터를 읽기 전용 메모리 매핑으로 열어 여러 프로세스가 공유하기 쉽게 해준다. 다만 `KeyedVectors`만 저장하면 Word2Vec 학습을 이어갈 수 없다.

> 공식 문서: [KeyedVectors](https://radimrehurek.com/gensim/models/keyedvectors.html)

### 12.3 다른 도구와 교환할 때

Word2Vec 호환 형식으로 내보낼 수 있다.

```python
model.wv.save_word2vec_format(
    "artifacts/word_vectors.bin",
    binary=True,
)
```

다시 불러온다.

```python
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format(
    "artifacts/word_vectors.bin",
    binary=True,
)
```

이 형식도 조회용 벡터만 포함하므로 일반적으로 추가 학습은 불가능하다.

### 12.4 모델과 함께 보관할 메타데이터

다음 정보를 JSON, 모델 레지스트리 또는 실험 관리 도구에 함께 기록하는 것이 좋다.

- Gensim과 Python 버전
- 학습 데이터 버전과 기간
- 토큰화·정규화 코드 버전
- 전체 문장 수와 토큰 수
- 하이퍼파라미터
- vocabulary 크기와 OOV 기준
- 랜덤 시드와 `workers`
- 오프라인 평가 결과
- 데이터 및 사전학습 벡터 라이선스

임베딩과 분류기를 함께 사용하는 경우 둘을 동일한 릴리스 단위로 관리한다. 임베딩만 다시 학습하면 벡터 좌표계가 달라지므로 기존 분류기에 그대로 연결해서는 안 된다.

---

## 13. 증분 학습

새 데이터가 들어왔을 때 전체 `Word2Vec` 모델을 저장해 두었다면 vocabulary를 확장하고 추가 학습할 수 있다.

```python
new_sentences = [
    ["new", "product", "category", "appeared"],
    ["customers", "use", "new", "search", "terms"],
]

# 기존 vocabulary를 유지하면서 새 단어 반영
model.build_vocab(
    new_sentences,
    update=True,
)

model.train(
    new_sentences,
    total_examples=len(new_sentences),
    epochs=model.epochs,
)
```

증분 학습은 편리하지만 다음 문제가 있다.

- 최근 데이터에 지나치게 맞춰져 과거 의미 관계가 변할 수 있다.
- 새 단어 출현량이 적으면 벡터 품질이 낮다.
- 기존 단어와 신규 단어의 학습량이 불균형할 수 있다.
- 같은 단어의 의미가 변하는 semantic drift가 발생할 수 있다.

따라서 운영에서는 고정된 평가 단어 쌍, 유사 검색 결과, 다운스트림 모델 성능을 이전 모델과 비교한 후 배포한다. 데이터가 크게 달라졌다면 부분 업데이트보다 전체 재학습이 더 안정적일 수 있다.

---

## 14. 한국어 데이터에서 사용할 때

한국어는 조사와 어미가 단어에 붙기 때문에 단순 공백 분리만 사용하면 vocabulary가 불필요하게 커질 수 있다.

```python
sentence = "상품의 배송이 빨라서 만족했습니다"
print(sentence.split())
```

이 경우 `상품의`, `배송이`, `빨라서`, `만족했습니다`가 각각 별개의 토큰이 된다. 일반적으로는 형태소 분석기를 이용해 의미 있는 단위로 나눈다.

형태소 분석 결과가 다음과 같이 준비되었다고 가정하면 Gensim에는 그대로 전달할 수 있다.

```python
tokenized_sentences = [
    ["상품", "배송", "빠르다", "만족"],
    ["포장", "상태", "좋다"],
    ["배송", "늦다", "환불"],
]

model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10,
)
```

실무 전처리 정책에서는 다음을 결정해야 한다.

- 명사만 사용할지, 동사와 형용사의 원형도 유지할지
- 조사와 어미를 제거할지
- 숫자, 단위, 상품 코드와 브랜드명을 유지할지
- 띄어쓰기 오류와 반복 문자를 어떻게 정규화할지
- 개인정보를 어떤 단계에서 제거할지
- 동의어와 표기 변형을 통합할지

검색에서는 상품 코드와 브랜드명이 중요할 수 있고, 감성 분석에서는 형용사와 부정 표현이 중요하다. 따라서 품사 제거 규칙은 목적에 따라 달라야 한다.

FastText는 한국어의 변형과 미등록 단어를 어느 정도 완화하지만 올바른 형태소 분석과 데이터 정제를 완전히 대체하지는 않는다.

---

## 15. 평가와 운영 점검

### 15.1 임베딩 자체 평가

- 업무 전문가가 정의한 유사 단어 쌍의 순위
- `most_similar()` 결과의 정성 평가
- 단어 유추 문제
- 특정 속성에 대한 편향 검사
- 빈도 구간별 품질 비교

### 15.2 다운스트림 평가

임베딩 자체의 유사도가 좋아 보여도 실제 분류나 검색 성능이 좋아진다는 보장은 없다. 최종 업무 지표로 평가해야 한다.

- 분류: accuracy, macro F1, 클래스별 recall
- 검색: Precision@K, Recall@K, MRR, NDCG
- 군집화: silhouette score와 사람의 군집 해석
- 토픽 모델: coherence와 전문가 해석

### 15.3 운영 모니터링

- 신규 데이터의 OOV 비율
- 토큰 빈도와 문서 길이 분포 변화
- 주요 질의의 최근접 단어 변화
- 빈 벡터가 생성되는 문서 비율
- 추론 시간과 메모리 사용량
- 데이터 기간별 다운스트림 성능

특히 평균 Word2Vec 방식에서 모든 단어가 OOV이면 영벡터가 생성될 수 있다. 이 비율을 로그로 남기지 않으면 모델이 입력을 이해하지 못한 상태를 발견하기 어렵다.

---

## 16. 자주 발생하는 실수

### 실수 1: 문자열을 문장 토큰 리스트로 착각

```python
# 잘못된 형태
model = Word2Vec(["machine learning"])
```

```python
# 올바른 형태
model = Word2Vec([["machine", "learning"]])
```

### 실수 2: 너무 작은 말뭉치에서 의미를 과도하게 해석

몇 문장으로도 코드는 실행되지만 안정적인 의미 관계는 학습되지 않는다. 작은 예제의 유사도 값은 API 동작 확인용으로만 사용한다.

### 실수 3: `min_count` 때문에 중요한 단어가 사라짐

```python
print("important_term" in model.wv)
```

도메인 희귀어가 중요하다면 빈도 분포를 확인한 뒤 `min_count`를 정한다.

### 실수 4: 테스트 데이터까지 임베딩 학습에 사용

경진대회 규정과 실험 목적에 따라 허용 범위가 다를 수 있지만, 일반적인 성능 추정에서는 학습 split으로만 임베딩을 훈련하는 것이 안전하다.

### 실수 5: 모델과 전처리 버전이 달라짐

학습 때는 소문자로 변환했는데 추론 때 대소문자를 유지하면 OOV가 급증할 수 있다. 토큰화 코드를 모델과 같은 버전으로 배포한다.

### 실수 6: 임베딩만 교체하고 기존 분류기를 유지

새로 학습한 100차원 임베딩은 차원 수가 같아도 이전 임베딩과 좌표축 의미가 다르다. 다운스트림 분류기도 함께 다시 학습하고 검증한다.

### 실수 7: `KeyedVectors`로 추가 학습 시도

`KeyedVectors`는 조회에 최적화된 객체다. 추가 학습 가능성이 있다면 전체 `Word2Vec` 또는 `FastText` 모델도 보관한다.

---

## 17. 실무 선택 가이드

| 상황 | 우선 검토할 방법 |
|---|---|
| 빠르고 설명 가능한 문서 검색 기준선 | TF-IDF |
| 비슷한 단어와 연관어 검색 | Word2Vec |
| 오타·활용형·신규 단어가 많음 | FastText |
| 문서 단위의 고정 길이 벡터 필요 | Doc2Vec 또는 단어 벡터 평균 |
| 문서 집합의 주제를 사람이 탐색 | LDA |
| 문맥에 따른 다의어 구분 필요 | Transformer 임베딩 |
| 높은 검색 품질과 문장 의미 이해 필요 | Sentence Transformer 계열과 비교 |

Gensim 모델을 선택하는 가장 현실적인 방식은 단순한 기준선부터 비교하는 것이다.

1. TF-IDF와 선형 모델을 먼저 측정한다.
2. Word2Vec 평균 벡터를 비교한다.
3. OOV가 문제라면 FastText를 비교한다.
4. 필요하면 Doc2Vec 또는 Transformer 임베딩으로 확장한다.
5. 가장 복잡한 모델이 아니라 품질, 지연시간, 메모리, 운영 비용을 함께 만족하는 모델을 선택한다.

---

## 18. 참고 자료

- [Gensim 공식 문서](https://radimrehurek.com/gensim/)
- [Word2Vec API](https://radimrehurek.com/gensim/models/word2vec.html)
- [FastText 튜토리얼](https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html)
- [Doc2Vec API](https://radimrehurek.com/gensim/models/doc2vec.html)
- [KeyedVectors API](https://radimrehurek.com/gensim/models/keyedvectors.html)
- [Gensim Downloader API](https://radimrehurek.com/gensim/downloader.html)
