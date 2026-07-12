# 01. LeNet-5

**논문:** Gradient-based learning applied to document recognition  
**한국어 제목:** 문서 인식에 적용한 기울기 기반 학습  
**저자:** Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner  
**학회/연도:** Proceedings of the IEEE, 1998  
**원문:** [https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

## 이 폴더에서 얻을 것

이미지를 1차원 표로 펴지 말고, 작은 필터가 훑으며 2차원 모양 정보를 보존하자는 CNN의 출발점입니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_lenet5_convolution_pooling.ipynb](./practice_lenet5_convolution_pooling.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- 합성곱
- 가중치 공유
- 수용 영역
- 풀링
- 공간 정보

## 다음 논문으로 이어지는 질문

AlexNet은 LeNet의 구조를 훨씬 크게 키우고 ReLU, GPU, 데이터 증강을 결합해 실사 이미지 분류로 확장합니다.
