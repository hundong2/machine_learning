# 1장. 실습에 필요한 Python과 NumPy

## 1. 변수와 자료형

```python
count = 3          # int: 정수(integer)
sigma = 1.5        # float: 실수(floating-point number)
name = "gaussian" # str: 문자열(string)
visible = True     # bool: 참/거짓(boolean)
```

`=`는 오른쪽 값을 왼쪽 이름에 저장하는 대입 연산자입니다. 수학의 등호와 달리 “같다”를 검사하려면 `==`를 씁니다.

## 2. 함수

```python
def square(value: float) -> float:
    """입력값의 제곱을 반환한다."""
    return value * value
```

- `def`: define, 함수를 정의합니다.
- `value: float`: 입력값을 float로 기대한다는 type hint입니다.
- `-> float`: 반환값을 float로 기대한다는 표시입니다.
- `return`: 계산한 값을 호출한 곳으로 돌려줍니다.
- type hint는 설명과 정적 검사에 도움을 주지만 실행 중 자료형을 강제로 바꾸지는 않습니다.

## 3. NumPy 배열

```python
import numpy as np

point = np.array([1.0, 2.0, 3.0], dtype=np.float64)
matrix = np.eye(3, dtype=np.float64)
```

- `import ... as ...`: 긴 모듈 이름에 별칭(alias)을 붙입니다.
- `np`: NumPy의 관례적 별칭입니다.
- `array`: Python 목록을 수치 배열로 만듭니다.
- `dtype`: data type, 각 원소의 저장 형식입니다.
- `float64`: 64-bit 부동소수점 수입니다.
- `eye(3)`: 3×3 identity matrix, 즉 단위행렬을 만듭니다.

## 4. shape와 축

```python
points = np.zeros((5, 3))
```

`points.shape == (5, 3)`은 점 5개가 있고 각 점에 x, y, z 세 좌표가 있다는 뜻입니다. axis 0은 점의 방향, axis 1은 좌표 성분의 방향입니다.

## 5. 반드시 알아야 할 축약 문법

| 문법 | 풀어쓴 의미 | 3DGS에서의 용도 |
|---|---|---|
| `a @ b` | `np.matmul(a, b)` | 행렬 곱 |
| `x ** 2` | `x`의 2제곱 | 거리와 Gaussian 지수 |
| `x.T` | `np.transpose(x)` | 행과 열 교환 |
| `x[:, 0]` | 모든 행의 0번 열 | 모든 점의 x 좌표 |
| `x[..., None]` | 마지막에 길이 1인 축 추가 | broadcasting 준비 |
| `a * b` | 원소별 곱 | 색과 alpha 곱 |
| `np.exp(x)` | 각 원소에 지수함수 | Gaussian 값 |
| `np.linalg.inv(m)` | 행렬 역행렬 | covariance 역행렬 |
| `np.clip(x, a, b)` | 값을 `[a,b]`에 제한 | alpha 안정화 |
| `f"{x:.3f}"` | 소수점 셋째 자리 문자열 | 결과 출력 |

## 6. broadcasting

shape가 `(H, W, 2)`인 픽셀 좌표에서 shape가 `(2,)`인 중심점을 빼면 NumPy는 중심점을 모든 픽셀에 자동으로 적용합니다. 이를 broadcasting이라고 합니다.

```python
delta = pixel_grid - center
```

반복문 없이 모든 픽셀의 상대 좌표를 동시에 계산하므로 Gaussian 영상 계산에 유용합니다.

## 7. 인덱스 순서 주의

수학 좌표는 `(x, y)`라고 쓰지만 NumPy 영상은 `image[y, x]`로 접근합니다.

- `x`: 가로 방향, 열(column)
- `y`: 세로 방향, 행(row)
- `image[y, x, channel]`: 특정 픽셀의 특정 색 채널

이 순서를 바꾸는 실수가 실습에서 가장 흔합니다.
