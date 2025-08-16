## 결측치 탐지 방법

- `isnull(), sum()` method 사용 

```python
train.isnull().sum()
```

```sh
index              0
gender             0
car                0
reality            0
child_num          0
income_total       0
income_type        0
edu_type           0
family_type        0
house_type         0
days_birth         0
days_employed      0
flag_mobil         0
work_phone         0
phone              0
email              0
job_type         321
family_size        0
begin_month        0
credit             0
dtype: int64
```

- box plot을 활용한 이상치 탐지 

```python
# 방법2
import seaborn as sns
import matplotlib.pyplot as plt

ax = plt.boxplot(train['days_employed'])
plt.show()
```

## mode 를 이용한 결측치 제거 

- mode 를 활용하여 가장 빈번하게 나오는 값을 확인하여 그 값을 fillna로 대체하는 방법

```python
mode_value = train['job_type'].mode()[0]
train['job_type'] = train['job_type'].fillna(mode_value)
```

## 결측치 특정 값을 Na로 대체 

- 이상치의 값을 결측값(NaN)으로 대체

```python
train.loc[train['days_employed'] == 365243, 'days_employed'] = pd.NA
```

- 결측치 데이터를 mean 값으로 대체 

```python
mean_value = train['days_employed'].mean()
train['days_employed'].fillna(mean_value, inplace=True)
```

## 중복데이터 식별

```python
drop_index_df = train.drop('index', axis = 1)
duplicated_df = drop_index_df[drop_index_df.duplicated()]
duplicated_df
```

```sh
# 코드 연습장
drop_index_df.duplicated()
0      False
1      False
2      False
3      False
4      False
       ...  
995    False
996    False
997    False
998    False
999    False
Length: 1000, dtype: bool
```

- 중복데이터 처리 

```sh
print(f'중복 데이터 제거 전 train 데이터의 개수는 {len(train)} 개 입니다.')

duplicated_indices = duplicated_df.index
final_train = train.drop(duplicated_indices, axis=0)

print(f'중복 데이터 제거 이후 train 데이터의 개수는 {len(final_train)} 개 입니다.')
```

