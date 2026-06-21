# 차분(diff)의 원리
차분은 **현재 값에서 바로 이전 값의 차이**를 구하는 것입니다.  
시계열에서 데이터의 변화량을 보기 위해 자주 씁니다.

## 예시 파일
14.1.2.DateTimePandas2.ipynb

## 답변
수학적으로는 이렇게 표현합니다.

$$
diff_t = x_t - x_{t-1}
$$

예를 들어 값이 아래와 같다면:

```python
[10, 13, 15, 12]
```

차분 결과는:

```python
[NaN, 3, 2, -3]
```

의미는 다음과 같습니다.

- `10`은 앞 값이 없어서 `NaN`
- `13 - 10 = 3`
- `15 - 13 = 2`
- `12 - 15 = -3`

즉, 차분은 **값 자체**보다 **변화량**을 보는 방법입니다.  
시계열에서 추세를 줄이고, 정상성을 만들기 위해 많이 사용합니다.

### pandas에서 사용법
```python
df['power_diff'] = df['power'].diff()
```

### 간단한 예
```python
import pandas as pd

s = pd.Series([10, 13, 15, 12])
print(s.diff())
```

출력:
```python
0    NaN
1    3.0
2    2.0
3   -3.0
dtype: float64
```

### 추가 자료
- [Pandas Series.diff 공식 문서](https://pandas.pydata.org/docs/reference/api/pandas.Series.diff.html)