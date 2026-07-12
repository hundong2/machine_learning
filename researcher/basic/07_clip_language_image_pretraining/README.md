# 07. CLIP

**논문:** Learning Transferable Visual Models From Natural Language Supervision  
**한국어 제목:** 자연어 감독으로 전이 가능한 시각 모델 학습하기  
**저자:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh et al.  
**학회/연도:** ICML 2021  
**원문:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

## 이 폴더에서 얻을 것

이미지와 텍스트를 같은 벡터 공간에 놓고, 맞는 쌍은 가깝게 틀린 쌍은 멀게 만드는 대조 학습 논문입니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_clip_contrastive_embedding.ipynb](./practice_clip_contrastive_embedding.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 임베딩
- 코사인 유사도
- 대조 학습
- zero-shot
- 멀티모달

## 다음 논문으로 이어지는 질문

이제 CNN, Transformer, 자기지도, 생성 모델, 언어-이미지 정렬이 하나의 현대 비전 연구 지도로 연결됩니다.
