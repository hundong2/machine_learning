**Hugging Face의 `datasets` 라이브러리**를 사용하면 파이썬(Python) 코드 몇 줄만으로 아주 쉽게 다운로드하고 불러올 수 있습니다.

먼저 라이브러리가 설치되어 있지 않다면 아래 명령어로 설치해야 합니다.

```bash
pip install datasets

```

다음은 각 데이터셋을 불러오는 파이썬 코드 예시입니다.

---

### 1. GLUE

GLUE는 여러 하위 태스크(MRPC, SST2, QNLI 등)로 구성되어 있으므로, 불러올 때 특정 태스크 이름을 함께 지정해야 합니다.

```python
from datasets import load_dataset

# 예: MRPC 태스크 불러오기
glue_dataset = load_dataset("glue", "mrpc")

# 예: SST2 태스크 불러오기
# glue_dataset = load_dataset("glue", "sst2")

```

### 2. SQuAD

```python
from datasets import load_dataset

# SQuAD v1.1 버전
squad_dataset = load_dataset("squad")

# 답변할 수 없는 질문이 포함된 SQuAD v2.0 버전을 원할 경우
# squad_dataset = load_dataset("squad_v2")

```

### 3. Common Crawl

Common Crawl 자체는 페타바이트급의 방대한 원시 웹 데이터입니다. 따라서 보통 이를 NLP용으로 정제한 C4 (Colossal Clean Crawled Corpus)나 **OSCAR**, **CC-100** 형태로 불러오는 것이 일반적입니다.

```python
from datasets import load_dataset

# Common Crawl을 기반으로 정제된 C4 데이터셋 (영어) 불러오기
c4_dataset = load_dataset("c4", "en") 

```

### 4. WMT

WMT는 기계 번역 데이터셋이므로, 해당하는 **연도**와 언어 쌍(Language Pair)을 지정해 주어야 합니다.

```python
from datasets import load_dataset

# 예: 2014년 독일어(de) - 영어(en) 번역 데이터셋 불러오기
wmt_dataset = load_dataset("wmt14", "de-en")

```

### 5. SNLI (Stanford Natural Language Inference)

```python
from datasets import load_dataset

snli_dataset = load_dataset("snli")

```

### 6. IMDb (감성 분석)

```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")

```

---

> **참고 사항**
> 허깅페이스 데이터셋을 처음 불러올 때는 데이터를 다운로드하고 캐시(Cache)하는 과정이 진행되므로 시간이 조금 걸릴 수 있습니다. 한 번 다운로드된 후에는 로컬에 저장되어 매우 빠르게 불러와집니다.