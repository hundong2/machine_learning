# 03. ResNet

**논문:** Deep Residual Learning for Image Recognition  
**한국어 제목:** 이미지 인식을 위한 깊은 잔차 학습  
**저자:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**학회/연도:** arXiv 2015, CVPR 2016  
**원문:** [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

## 이 폴더에서 얻을 것

층을 더 깊게 쌓아도 정보와 기울기가 흐르도록 입력을 다음 층에 더해 주는 스킵 커넥션을 제안했습니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_resnet_skip_connection.ipynb](./practice_resnet_skip_connection.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 잔차 학습
- 스킵 커넥션
- 항등 매핑
- 기울기 흐름
- degradation

## 다음 논문으로 이어지는 질문

ViT는 CNN을 더 깊게 다듬는 흐름에서 벗어나, 이미지를 패치 토큰 시퀀스로 보고 Transformer를 적용합니다.
