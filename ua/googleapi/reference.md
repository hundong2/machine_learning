# `temperature` 매개변수: AI의 창의성 조절하기

`temperature`는 AI 모델이 텍스트를 생성할 때 얼마나 무작위적이고 창의적인 답변을 만들지를 조절하는 설정값입니다. 값이 낮으면 예측 가능하고 일관된 답변을, 높으면 더 다양하고 창의적인 답변을 생성합니다.

## 예시 파일

[ua/googleapi/text_generation.py](./text_generation.py)

## 답변

### 1. `temperature`의 의미 (초보자 눈높이 설명)

`temperature`를 **'단어 뽑기 복권'**에 비유할 수 있습니다. AI가 다음 단어를 예측할 때, 가능한 모든 단어에 대해 '이 단어가 정답일 확률'을 계산합니다.

-   **`temperature`가 낮으면 (예: 0.1)**: 가장 확률이 높은 단어에 복권을 몰아주는 것과 같습니다. AI는 거의 항상 가장 예측 가능한, 안전한 단어만 선택합니다.
    -   **결과**: 사실 요약, 코드 생성처럼 정해진 답변이 필요할 때 유용합니다.

-   **`temperature`가 높으면 (예: 1.5)**: 확률이 낮은 단어들에게도 복권을 많이 나눠주는 것과 같습니다. AI는 가끔 엉뚱하지만 창의적인 단어를 선택할 수 있습니다.
    -   **결과**: 시나 소설 쓰기처럼 독창적인 아이디어가 필요할 때 유용하지만, 가끔 말이 안 되는 문장을 만들 수도 있습니다.

### 2. `temperature`의 기능과 이론 (심층 탐구)

AI 언어 모델은 내부적으로 다음 단어를 예측하기 위해 **Softmax 함수**를 사용합니다. `temperature`는 이 함수의 동작을 조절하는 핵심적인 역할을 합니다.

1.  **로짓 (Logits) 생성**: 모델은 어휘 사전에 있는 모든 단어에 대해 다음 단어로 얼마나 적합한지를 나타내는 점수(로짓)를 계산합니다. 이 점수는 정규화되지 않은 원시 값입니다.

2.  **Temperature 적용**: Softmax 함수에 로짓을 넣기 전에, 모든 로짓 값을 `temperature` 값으로 나눕니다.
    `조정된 로짓 = 원래 로짓 / temperature`

3.  **Softmax 계산**: 조정된 로짓을 Softmax 함수에 넣어 각 단어의 최종 확률을 계산합니다.

#### Temperature 값에 따른 변화

-   **`temperature` → 0**: 로짓을 매우 작은 수로 나누면, 가장 높은 점수를 가진 로짓과 다른 로짓들의 차이가 극대화됩니다. Softmax 결과, 가장 확률이 높은 단어의 확률은 거의 1에 가까워지고 나머지는 0에 가까워집니다. 모델은 매우 **결정론적(deterministic)**으로 행동합니다.

-   **`temperature` = 1**: 로짓을 1로 나누므로 아무 변화가 없습니다. 모델은 자신이 계산한 원래 확률 분포를 그대로 사용합니다.

-   **`temperature` > 1**: 로짓을 큰 수로 나누면, 로짓들 간의 차이가 줄어듭니다. Softmax 결과, 확률 분포가 전반적으로 평평해져 확률이 낮았던 단어들도 선택될 가능성이 커집니다. 모델은 더 **무작위적(random)**으로 행동합니다.

결론적으로, `temperature`는 모델의 예측 확률 분포를 얼마나 뾰족하게(확신에 차게) 또는 평평하게(불확실하게) 만들지 결정하는 **수학적 조절 장치**입니다.

### 추가 자료

-   [Google AI for Developers: GenerateContentConfig](https://ai.google.dev/api/python/google/generativeai/types/GenerateContentConfig)
-   [Hugging Face Blog: How to generate text with LLMs (Temperature 설명 포함)](https://huggingface.co/blog/how-to-generate)