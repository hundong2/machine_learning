# 실습 코드

`bilevel_autoresearch_demo.py`는 논문의 구조를 작은 탐색 문제로 축소한 교육용 시뮬레이터입니다. 외부 패키지는 필요하지 않습니다.

이 코드는 실제 LLM이나 GPU 훈련을 사용하지 않습니다. 대신 "기본 탐색기는 익숙한 후보를 반복한다"는 상황을 작은 함수로 만들고, Level 2 메커니즘이 그 반복을 끊는 모습을 보여 줍니다.

## 실행

```powershell
cd D:\workspace\machine_learning\Research\Bilevel_Autoresearch_Meta-Autoresearching_Itself
python .\code\bilevel_autoresearch_demo.py
```

상세 trace를 보려면 다음처럼 실행합니다.

```powershell
python .\code\bilevel_autoresearch_demo.py --trace C
python .\code\bilevel_autoresearch_demo.py --trace A --seed 13
```

반복 수나 iteration 수를 바꿀 수 있습니다.

```powershell
python .\code\bilevel_autoresearch_demo.py --repeats 5 --iterations 40
```

## 그룹 의미

- A: Level 1 only
- B: Level 1 + Level 1.5
- C: Level 1 + Level 1.5 + Level 2
- D: Level 1 + Level 2

초급 독자는 먼저 A와 C만 비교하면 됩니다. A는 기본 반복 탐색이고, C는 외부 루프가 탐색 메커니즘을 바꾸는 경우입니다.

## 결과 해석

출력 예시는 다음과 같습니다.

```text
group  mean_delta  std_delta   repeats
A         -0.0152     0.0008         3
B         -0.0170     0.0004         3
C         -0.0547     0.0004         3
D         -0.0416     0.0229         3
```

- `mean_delta`: 평균 개선량입니다. 더 음수일수록 좋습니다.
- `std_delta`: 반복 실행 사이의 차이입니다. 값이 크면 안정성이 낮습니다.
- C가 A보다 더 낮게 나오면 Level 2가 탐색 방식을 바꿔 더 좋은 방향을 찾았다는 뜻입니다.

`--trace C`를 붙이면 C 그룹의 iteration별 기록을 볼 수 있습니다. `mechanism`이 `default-prior`에서 `tabu-search`나 `orthogonal-exploration`으로 바뀌는 시점을 확인하세요.

## 코드에서 관찰할 부분

- `DefaultPrior`: 같은 후보를 반복하는 내부 루프의 기본 prior
- `update_strategy`: Level 1.5의 freeze 기반 전략 조정
- `TabuMechanism`, `BanditMechanism`, `OrthogonalMechanism`: Level 2가 발견했다고 가정한 메커니즘들
- `choose_level2_mechanism`: trace를 보고 활성 메커니즘을 교체하는 외부 루프
- `evaluate`: 숨겨진 objective. 논문에서 `TOTAL_BATCH_SIZE` 감소가 핵심이었던 상황을 toy objective로 모델링

## 논문과 다른 점

- 논문은 LLM이 실제로 코드를 읽고 새 코드를 생성한다.
- 이 실습은 교육용이라 사람이 미리 작성한 메커니즘 후보 중 하나를 선택한다.
- 논문은 GPT pretraining benchmark를 실행하지만, 이 실습은 빠르게 실행되는 toy objective를 사용한다.
- 따라서 이 코드는 논문 재현이 아니라 개념 이해용이다.
