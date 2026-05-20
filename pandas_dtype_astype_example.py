import pandas as pd

# 샘플 데이터 생성
df = pd.DataFrame({
    '과일': ['사과', '바나나', '사과', '딸기', '바나나', '사과'], # 중복되는 문자열
    '가격': [1000, 1500, 1000, 2000, 1500, 1000]
})

# 1. dtype 확인 (현재 데이터 타입)
print("--- 변경 전 dtype ---")
print(df.dtypes)
print(f"과일 컬럼 메모리 사용량: {df['과일'].memory_usage(deep=True)} bytes\n")

# 2. astype으로 데이터 타입 변경 (object -> category)
df['과일'] = df['과일'].astype('category')

# 3. 변경 후 dtype 확인 및 메모리 비교
print("--- 변경 후 dtype ---")
print(df.dtypes)
print(f"과일 컬럼 카테고리 변환 후 메모리 사용량: {df['과일'].memory_usage(deep=True)} bytes\n")

# 내부 카테고리 정보 확인
print("--- 카테고리 코드로 변환된 내부 데이터 ---")
print(df['과일'].cat.codes)