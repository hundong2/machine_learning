# 04. Vision Transformer (ViT)

**논문:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale  
**한국어 제목:** 이미지는 16x16 단어들의 문장이다: 대규모 이미지 인식을 위한 Transformer  
**저자:** Alexey Dosovitskiy et al.  
**학회/연도:** ICLR 2021, arXiv 2020  
**원문:** [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

## 이 폴더에서 얻을 것

이미지를 작은 패치로 잘라 단어처럼 나열하고, self-attention으로 멀리 떨어진 패치들의 관계를 한 번에 봅니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_vit_patch_attention.ipynb](./practice_vit_patch_attention.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 패치
- 토큰
- 위치 인코딩
- self-attention
- 사전학습

## 다음 논문으로 이어지는 질문

MAE는 ViT의 패치 구조를 이용해 이미지 일부를 가리고 복원하는 자기지도 학습을 만듭니다.
