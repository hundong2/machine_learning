# 학습 가이드

## 대상 독자

이 자료는 Python을 한 번 이상 실행해 본 기술자를 기준으로 작성했다. 논문식 최적화 용어를 모두 알고 있을 필요는 없다. 다만 "설정을 바꾸고 성능을 비교한다"는 머신러닝 실험 흐름은 알고 있다고 가정한다.

## 30분 빠른 학습 경로

시간이 많지 않다면 아래만 진행한다.

1. README의 핵심 비유를 읽는다.
2. `paper_summary.ko.md`의 "초급 독자용 먼저 보기"를 읽는다.
3. `python .\code\bilevel_autoresearch_demo.py`를 실행한다.
4. 출력 표에서 A, B, C, D의 `mean_delta`를 비교한다.
5. `python .\code\bilevel_autoresearch_demo.py --trace C`를 실행해 Level 2가 언제 켜지는지 본다.

## 추천 학습 순서

1. `paper_summary.ko.md`에서 문제의식과 세 수준 구조를 먼저 읽는다.
2. 논문 원문에서 Introduction, Methods 3.1-3.6, Results 4.1-4.4, Limitations 5.4를 읽는다.
3. `code/bilevel_autoresearch_demo.py`를 실행해 Group A, B, C, D의 차이를 확인한다.
4. trace 출력으로 어떤 후보가 반복되고, Level 2 이후 어떤 후보가 새로 등장하는지 본다.
5. 직접 새 메커니즘을 하나 추가해 결과가 어떻게 변하는지 비교한다.

## 선행 개념

- Gradient-free optimization: 미분값을 쓰지 않고 후보를 직접 평가하며 찾는 최적화. 여기서는 "여러 설정을 실행해 보고 좋은 것을 고르는 방식"으로 이해하면 된다.
- Hyperparameter search: learning rate, batch size 같은 설정을 바꾸며 성능을 찾는 과정이다.
- Bilevel optimization: 위층 문제가 아래층 문제의 절차나 선택에 영향을 주는 구조다. 여기서는 외부 루프가 내부 루프의 탐색 방식을 바꾸는 구조로 보면 된다.
- Multi-armed bandit: 여러 선택지 중 좋은 선택지를 찾되, 아직 모르는 선택지도 가끔 시험하는 문제다.
- Tabu search: 최근 실패했거나 이미 본 후보를 잠시 금지해 같은 곳을 반복해서 보지 않게 하는 방법이다.
- Design of experiments: 여러 변수를 한쪽에 치우치지 않게 체계적으로 시험하는 설계 방법이다.

## 용어 정리

| 용어 | 의미 |
| --- | --- |
| Autoresearch | LLM이 제안, 실행, 평가, 유지/폐기를 반복하는 자동 연구 루프 |
| Inner loop | 실제 과제 성능을 개선하는 Level 1 루프 |
| Outer loop | 내부 루프의 전략이나 메커니즘을 개선하는 상위 루프 |
| Mechanism carrier | 메커니즘을 담는 표현. 논문에서는 Python 코드가 대표 예시 |
| Trace | 각 iteration의 제안, 결과, 채택 여부 기록 |
| Code injection | 생성된 코드를 런타임에 로딩해 기존 루프에 연결하는 방식 |
| Validate and revert | 새 코드가 검증에 실패하면 원래 코드로 되돌리는 안전장치 |

## 출력 결과 읽는 법

실습 코드는 다음과 비슷한 표를 출력한다.

```text
group  mean_delta  std_delta   repeats
A         -0.0152     0.0008         3
B         -0.0170     0.0004         3
C         -0.0547     0.0004         3
D         -0.0416     0.0229         3
```

읽는 법은 간단하다.

- `mean_delta`는 평균 개선량이다. 더 음수일수록 좋다.
- `std_delta`는 반복 실행 간 흔들림이다. 클수록 결과가 불안정하다.
- `repeats`는 몇 번 반복했는지다.
- 위 예에서는 C가 A보다 더 크게 개선된다. 이는 Level 2 메커니즘이 핵심 방향을 찾게 만들었기 때문이다.

논문도 같은 방향으로 해석하면 된다. 수치 자체보다 "어떤 구조가 반복 탐색을 끊었는가"를 보는 것이 중요하다.

## 핵심 질문

- Level 1.5는 왜 batch size 감소를 찾지 못했는가?
- Level 2가 생성한 Tabu Search는 어떤 실패 모드를 막는가?
- Group C가 Group B보다 나아진 이유는 "더 많은 계산" 때문인가, "다른 탐색 분포" 때문인가?
- Python 코드가 아니라 prompt, skill, memory가 메커니즘 carrier라면 어떤 검증이 필요할까?
- recursive bootstrapping을 주장하려면 어떤 추가 실험이 필요할까?

## 막히기 쉬운 지점

- `bilevel`이라는 말 때문에 수식 최적화를 먼저 떠올릴 수 있다. 이 자료에서는 "위층 루프가 아래층 루프의 탐색 방식을 바꾼다" 정도로 시작하면 된다.
- `mechanism`은 거창한 말처럼 보이지만, 후보 생성 규칙이나 반복 방지 규칙이라고 보면 된다.
- `code injection`은 위험한 기능처럼 들릴 수 있다. 논문에서도 그래서 import 검증과 rollback을 둔다.
- `recursive self-improvement`는 이 논문에서 완료된 실험 결과가 아니다. 가능성 또는 다음 단계로 읽어야 한다.

## 실습 1: 기본 실행

```powershell
cd D:\workspace\machine_learning\Research\Bilevel_Autoresearch_Meta-Autoresearching_Itself
python .\code\bilevel_autoresearch_demo.py
```

확인할 것:

- Group A는 같은 방향을 반복하는가?
- Group B는 freeze 때문에 탐색 방향이 바뀌지만 핵심 방향을 놓치는가?
- Group C는 Level 2 이후 `total_batch_size` 감소를 발견하는가?

## 실습 2: trace 보기

```powershell
python .\code\bilevel_autoresearch_demo.py --trace C
python .\code\bilevel_autoresearch_demo.py --trace A
```

trace에서 볼 것:

- `mechanism` 컬럼이 언제 바뀌는가?
- `accepted`가 `yes`인 iteration은 어떤 변경을 했는가?
- Level 2 활성화 전후로 제안되는 파라미터 분포가 달라지는가?

## 실습 3: objective 바꾸기

`code/bilevel_autoresearch_demo.py`의 `evaluate` 함수를 수정해 보자.

- `total_batch_size=512`의 이점을 줄인다.
- `lr=0.8`의 이점을 크게 만든다.
- `window_pattern="SLSL"`이 가장 좋게 만든다.

수정 후 어떤 그룹이 가장 빨리 새 optimum을 찾는지 비교한다.

## 실습 4: 새 메커니즘 추가

다음 아이디어 중 하나를 구현해 보자.

- `RandomRestartMechanism`: 10회 실패하면 완전히 다른 파라미터 조합을 시도한다.
- `CoverageMechanism`: 아직 한 번도 바꾸지 않은 파라미터를 우선한다.
- `PairwiseMechanism`: 두 파라미터를 동시에 바꾸는 후보를 만든다.

추가 후 `choose_level2_mechanism`에서 새 메커니즘이 선택되도록 연결하고 결과를 비교한다.

## 체크 질문과 답

Q1. 이 논문에서 "bilevel"이라는 말은 고전적인 미분 가능한 nested optimization을 정확히 푸는 뜻인가?

A1. 아니다. 논문은 구조적, 알고리즘적 의미의 bilevel을 사용한다. 상위 루프가 하위 루프의 탐색 메커니즘을 바꾸지만, 고전적 bilevel 최적화 문제를 정확히 푸는 것은 아니다.

Q2. Level 1.5와 Level 2의 가장 중요한 차이는 무엇인가?

A2. Level 1.5는 기존 메커니즘 안에서 search config를 조정한다. Level 2는 proposal, selection, update 같은 메커니즘 구조를 바꾸는 코드를 생성한다.

Q3. 논문이 증명한 것은 recursive self-improvement인가?

A3. 아니다. 논문이 실험적으로 보인 것은 첫 번째 bilevel 단계, 즉 외부 루프가 내부 루프의 탐색 행동을 개선할 수 있다는 점이다. recursive self-application은 future work에 가깝다.

## 최종 이해도 기준

아래 네 문장을 자기 말로 설명할 수 있으면 초급 기술자 기준으로 충분히 이해한 것이다.

- 기존 autoresearch는 좋은 후보를 찾지만, 후보를 찾는 방식은 고정되어 있다.
- Level 1.5는 탐색 방향을 조정하지만 탐색 알고리즘 자체는 크게 바꾸지 않는다.
- Level 2는 반복을 막거나 다른 축을 보게 하는 새 메커니즘을 만들어 내부 루프에 연결한다.
- 좋은 성능 개선을 주장하려면 평균 결과뿐 아니라 실패한 패치, 작은 반복 수, 단일 benchmark 같은 한계도 같이 봐야 한다.
