# Basic Vision Paper Roadmap

초보자가 연구자로 첫발을 내딛기 위해 만든 컴퓨터 비전 핵심 논문 학습 자료집입니다. 목표는 논문을 외우는 것이 아니라, 각 논문이 어떤 문제를 보고 어떤 가정을 바꾸었는지 이해하는 것입니다.

## 공부 순서

| 순서 | 논문/모델 | 한국어 제목 | 실습 노트북 |
|---|---|---|---|
| 01 | [LeNet-5](./01_lenet5_document_recognition/README.md) | 문서 인식에 적용한 기울기 기반 학습 | [practice_lenet5_convolution_pooling.ipynb](./01_lenet5_document_recognition/practice_lenet5_convolution_pooling.ipynb) |
| 02 | [AlexNet](./02_alexnet_imagenet_classification/README.md) | 깊은 합성곱 신경망을 이용한 ImageNet 분류 | [practice_alexnet_relu_dropout.ipynb](./02_alexnet_imagenet_classification/practice_alexnet_relu_dropout.ipynb) |
| 03 | [ResNet](./03_resnet_deep_residual_learning/README.md) | 이미지 인식을 위한 깊은 잔차 학습 | [practice_resnet_skip_connection.ipynb](./03_resnet_deep_residual_learning/practice_resnet_skip_connection.ipynb) |
| 04 | [Vision Transformer (ViT)](./04_vit_image_as_words/README.md) | 이미지는 16x16 단어들의 문장이다: 대규모 이미지 인식을 위한 Transformer | [practice_vit_patch_attention.ipynb](./04_vit_image_as_words/practice_vit_patch_attention.ipynb) |
| 05 | [Latent Diffusion Model (LDM)](./05_latent_diffusion_models/README.md) | 잠재 확산 모델을 이용한 고해상도 이미지 합성 | [practice_latent_diffusion_noise.ipynb](./05_latent_diffusion_models/practice_latent_diffusion_noise.ipynb) |
| 06 | [Masked Autoencoder (MAE)](./06_mae_masked_autoencoders/README.md) | 마스크드 오토인코더는 확장 가능한 비전 학습자다 | [practice_mae_mask_reconstruct.ipynb](./06_mae_masked_autoencoders/practice_mae_mask_reconstruct.ipynb) |
| 07 | [CLIP](./07_clip_language_image_pretraining/README.md) | 자연어 감독으로 전이 가능한 시각 모델 학습하기 | [practice_clip_contrastive_embedding.ipynb](./07_clip_language_image_pretraining/practice_clip_contrastive_embedding.ipynb) |

## 빠른 시작

1. 각 폴더의 `README.md`에서 배경과 읽는 순서를 확인합니다.
2. `translation_ko.md`로 논문 초록과 핵심 주장을 초보자용 의역으로 읽습니다.
3. `core_notes.md`에서 연구 아이디어, 수식 직관, 실무 연결을 정리합니다.
4. `practice_*.ipynb`를 실행해 손으로 작은 실험을 해봅니다.
5. 모르는 단어는 [glossary.md](./glossary.md)를 먼저 확인합니다.

## 보완 자료

- [glossary.md](./glossary.md): 논문 용어와 코드 변수 용어를 함께 정리한 사전입니다.
- [code_function_guide.md](./code_function_guide.md): 노트북에 나오는 NumPy/Matplotlib 함수 해설입니다.
- [study_questions.md](./study_questions.md): 논문별 자기점검 질문입니다.
- [beginner_review_log.md](./beginner_review_log.md): 초보자 관점 3회 보완 기록입니다.

## 큰 흐름

- LeNet-5: 이미지는 표가 아니라 2차원 구조라는 생각에서 CNN이 시작됩니다.
- AlexNet: CNN이 대규모 데이터와 GPU를 만나 현대 딥러닝의 주류가 됩니다.
- ResNet: 깊어질수록 죽는 기울기를 스킵 커넥션으로 살립니다.
- ViT: 이미지를 패치 토큰의 문장처럼 읽습니다.
- MAE: 라벨 없이도 가려진 이미지를 복원하며 표현을 배웁니다.
- CLIP: 텍스트와 이미지를 같은 임베딩 공간에 연결합니다.
- Latent Diffusion: 텍스트 조건과 잠재 공간 확산으로 생성형 이미지 AI의 실용성을 높입니다.

## 실습 환경

노트북은 무거운 GPU 학습을 요구하지 않도록 `numpy`와 `matplotlib` 중심의 작은 실험으로 작성했습니다.

```bash
pip install numpy matplotlib jupyter
jupyter notebook
```

## 참고 원문

| 모델 | 원문 | 보조 링크 |
|---|---|---|
| LeNet-5 | [Gradient-based learning applied to document recognition](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | [Proceedings of the IEEE, 1998](https://ieeexplore.ieee.org/document/726791) |
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | [NeurIPS, 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) |
| ResNet | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | [arXiv 2015, CVPR 2016](https://arxiv.org/pdf/1512.03385) |
| Vision Transformer (ViT) | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | [ICLR 2021, arXiv 2020](https://arxiv.org/pdf/2010.11929) |
| Latent Diffusion Model (LDM) | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | [CVPR 2022, arXiv 2021](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
| Masked Autoencoder (MAE) | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | [CVPR 2022, arXiv 2021](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html) |
| CLIP | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | [ICML 2021](https://proceedings.mlr.press/v139/radford21a.html) |

## 학습 체크리스트

- [ ] 합성곱이 왜 공간 정보를 보존하는지 설명할 수 있다.
- [ ] ReLU가 tanh보다 학습에 유리한 이유를 기울기 관점에서 말할 수 있다.
- [ ] residual block의 `F(x) + x`가 왜 깊은 모델을 돕는지 설명할 수 있다.
- [ ] ViT의 patch, position embedding, self-attention 흐름을 그릴 수 있다.
- [ ] diffusion의 forward noise와 reverse denoising을 구분할 수 있다.
- [ ] MAE가 왜 라벨 없이도 표현을 배우는지 이해한다.
- [ ] CLIP의 cosine similarity와 contrastive learning을 작은 행렬로 계산할 수 있다.

## 저작권과 번역 범위

이 자료의 `translation_ko.md`는 논문 원문 전체의 직역이 아니라, 초보자가 논문을 읽기 전에 맥락을 잡을 수 있도록 작성한 한국어 의역과 해설입니다. 정확한 문장과 세부 실험은 각 폴더의 원문 링크를 기준으로 확인하세요.
