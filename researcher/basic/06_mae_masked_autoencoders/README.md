# 06. Masked Autoencoder (MAE)

**논문:** Masked Autoencoders Are Scalable Vision Learners  
**한국어 제목:** 마스크드 오토인코더는 확장 가능한 비전 학습자다  
**저자:** Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick  
**학회/연도:** CVPR 2022, arXiv 2021  
**원문:** [https://arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)

## 이 폴더에서 얻을 것

이미지 패치의 대부분을 가리고 남은 조각만 보고 복원하게 해 라벨 없이 시각 표현을 학습합니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_mae_mask_reconstruct.ipynb](./practice_mae_mask_reconstruct.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 자기지도 학습
- 마스킹
- 오토인코더
- 비대칭 encoder-decoder
- 복원 손실

## 다음 논문으로 이어지는 질문

CLIP은 라벨 대신 이미지와 자연어 캡션 쌍을 이용해 텍스트와 이미지를 같은 임베딩 공간에 정렬합니다.
