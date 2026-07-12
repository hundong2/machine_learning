# 05. Latent Diffusion Model (LDM)

**논문:** High-Resolution Image Synthesis with Latent Diffusion Models  
**한국어 제목:** 잠재 확산 모델을 이용한 고해상도 이미지 합성  
**저자:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer  
**학회/연도:** CVPR 2022, arXiv 2021  
**원문:** [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)

## 이 폴더에서 얻을 것

픽셀 전체가 아니라 압축된 잠재 공간에서 노이즈를 더하고 제거해 고해상도 생성 비용을 크게 낮춘 논문입니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_latent_diffusion_noise.ipynb](./practice_latent_diffusion_noise.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 정방향 확산
- 역방향 확산
- 노이즈 예측
- 잠재 공간
- 오토인코더

## 다음 논문으로 이어지는 질문

CLIP 같은 텍스트-이미지 임베딩은 확산 모델이 사용자의 문장 방향으로 이미지를 만들도록 조건을 제공합니다.
