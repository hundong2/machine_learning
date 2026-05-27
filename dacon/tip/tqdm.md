# tqdm 라이브러리 설명
`tqdm`은 반복문이 실행될 때 진행률 표시줄(Progress Bar)을 화면에 직관적으로 출력해 주는 파이썬 라이브러리입니다.

## 예시 파일
9.1.1.DataRestruct.ipynb

## 답변
`from tqdm import tqdm`은 긴 시간이 걸리는 반복 작업(예: 대용량 데이터 전처리, 모델 학습 등)을 수행할 때, **전체 작업 중 얼만큼 진행되었고 시간은 얼마나 남았는지 상태바 형태로 확인하기 위해** 사용합니다. 

기존 반복문 코드에서 리스트나 `range` 객체를 `tqdm()`으로 감싸주기만 하면 손쉽게 적용할 수 있습니다.

```python
import time
from tqdm import tqdm

# 1. 기본 반복문에서의 사용
print("작업 진행 상황:")
for i in tqdm(range(100)):
    time.sleep(0.02)  # 0.02초씩 대기하며 작업을 시뮬레이션

# 2. 리스트에 적용하기
my_list = ['데이터1', '데이터2', '데이터3', '데이터4']
for data in tqdm(my_list):
    time.sleep(0.5)
```

**💡 Pandas 사용 팁:**
데이터의 구조를 변경하거나 전처리할 때 `apply` 함수를 자주 쓰는데, 이때 `tqdm.pandas()`를 선언해 주고 `apply` 대신 `progress_apply()`를 사용하면 데이터프레임 작업의 진행률도 확인할 수 있습니다.

### 추가 자료
- [tqdm 공식 깃허브 및 문서](https://github.com/tqdm/tqdm)
- [Pandas와 tqdm 함께 사용하기 (progress_apply)](https://github.com/tqdm/tqdm#pandas-qdm#pandas-integration)