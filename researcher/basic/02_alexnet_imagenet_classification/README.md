# 02. AlexNet

**논문:** ImageNet Classification with Deep Convolutional Neural Networks  
**한국어 제목:** 깊은 합성곱 신경망을 이용한 ImageNet 분류  
**저자:** Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**학회/연도:** NeurIPS, 2012  
**원문:** [https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

## 이 폴더에서 얻을 것

CNN을 대규모 데이터, GPU, ReLU, 드롭아웃과 결합해 현대 딥러닝 붐을 폭발시킨 논문입니다.

## 읽는 순서

1. [translation_ko.md](./translation_ko.md): 논문의 문제의식과 핵심 주장을 한국어 의역으로 읽습니다.
2. [core_notes.md](./core_notes.md): 핵심 개념, 수식 직관, 실무 연결을 정리합니다.
3. [practice_alexnet_relu_dropout.ipynb](./practice_alexnet_relu_dropout.ipynb): 작은 파이썬 실험으로 논문 아이디어를 손으로 확인합니다.

## 먼저 잡아야 할 핵심 단어

- ReLU
- GPU 학습
- 드롭아웃
- 데이터 증강
- ImageNet

## 다음 논문으로 이어지는 질문

ResNet은 AlexNet 이후 더 깊게 쌓으려 할 때 생긴 최적화 난제를 스킵 커넥션으로 풉니다.
