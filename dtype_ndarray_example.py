import numpy as np

# dtype 예시
print("=== dtype (data type) 예시 ===")
# 정수 배열
int_array = np.array([1, 2, 3])
print(f"정수 배열: {int_array}, dtype: {int_array.dtype}")

# 실수 배열
float_array = np.array([1.0, 2.0, 3.0])
print(f"실수 배열: {float_array}, dtype: {float_array.dtype}")

# 문자열 배열
str_array = np.array(['a', 'b', 'c'])
print(f"문자열 배열: {str_array}, dtype: {str_array.dtype}")

# 불린 배열
bool_array = np.array([True, False, True])
print(f"불린 배열: {bool_array}, dtype: {bool_array.dtype}")

print("\n=== ndarray (n-dimensional array) 예시 ===")
# 1차원 배열 (1d array)
array_1d = np.array([1, 2, 3, 4])
print(f"1차원 배열: {array_1d}")
print(f"차원 수: {array_1d.ndim}d")
print(f"형태: {array_1d.shape}")

# 2차원 배열 (2d array)
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2차원 배열:\n{array_2d}")
print(f"차원 수: {array_2d.ndim}d")
print(f"형태: {array_2d.shape}")

# 3차원 배열 (3d array)
array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3차원 배열:\n{array_3d}")
print(f"차원 수: {array_3d.ndim}d")
print(f"형태: {array_3d.shape}")

print("\n=== dtype과 ndarray 함께 사용 ===")
# 다양한 데이터 타입의 다차원 배열
mixed_2d = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
print(f"2차원 실수 배열:\n{mixed_2d}")
print(f"데이터 타입: {mixed_2d.dtype}")
print(f"차원 수: {mixed_2d.ndim}")
print(f"배열 타입: {type(mixed_2d)}")
