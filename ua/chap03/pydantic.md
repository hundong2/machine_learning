# Pydantic: 파이썬을 위한 강력한 데이터 유효성 검사 라이브러리

Pydantic은 파이썬의 타입 힌트(Type Hint)를 사용하여 데이터의 유효성을 검사하고, 설정을 관리하며, 데이터를 직렬화/역직렬화하는 라이브러리입니다. API 요청과 같이 형식이 정해진 데이터를 다룰 때, 데이터가 올바른 구조와 타입을 가졌는지 자동으로 검증하여 코드의 안정성과 가독성을 크게 높여줍니다.

## 예시 파일

아래 예시 파일은 Pydantic의 핵심 기능들을 보여줍니다. 사용자의 기본 정보와 주소를 담는 모델을 정의하고, 유효성 검사, 기본값 설정, 중첩 모델, 데이터 변환 등의 기능을 포함합니다.

```python
from pydantic import BaseModel, Field, EmailStr, field_validator, ValidationError
from typing import List, Optional

# 중첩하여 사용할 주소 모델
class Address(BaseModel):
    street_address: str
    city: str
    zip_code: str

# 메인 사용자 모델
class User(BaseModel):
    # 필수 필드: id와 username
    id: int
    username: str = Field(
        min_length=3, 
        max_length=50, 
        description="사용자 이름은 3자 이상 50자 이하여야 합니다."
    )
    
    # Pydantic이 제공하는 특별한 타입 (이메일 형식 검증)
    email: EmailStr
    
    # 선택적 필드(Optional)와 기본값(default) 설정
    full_name: Optional[str] = None
    
    # 리스트 타입 필드와 기본값으로 빈 리스트 설정
    tags: List[str] = []
    
    # 다른 Pydantic 모델을 중첩하여 사용
    address: Address

    # 커스텀 유효성 검사기: username에 공백이 포함되지 않도록 검사
    @field_validator('username')
    def username_must_not_contain_spaces(cls, v):
        if ' ' in v:
            raise ValueError('사용자 이름에 공백을 포함할 수 없습니다.')
        return v

# 1. 유효한 데이터로 객체 생성
valid_data = {
    "id": 1,
    "username": "copilot",
    "email": "contact@github.com",
    "tags": ["AI", "Python"],
    "address": {
        "street_address": "123 AI Street",
        "city": "San Francisco",
        "zip_code": "94107"
    }
}

user_instance = User(**valid_data)
print("--- 유효한 데이터로 생성된 객체 ---")
print(user_instance)

# 2. Pydantic 객체를 JSON 문자열로 변환 (직렬화)
json_output = user_instance.model_dump_json(indent=2)
print("\n--- JSON으로 변환된 결과 ---")
print(json_output)

# 3. 유효하지 않은 데이터로 객체 생성 시도 (유효성 검사 실패)
invalid_data = {
    "id": "not-a-number",  # id는 정수여야 함
    "username": "co pilot", # 공백 포함
    "email": "invalid-email", # 이메일 형식이 아님
    "address": {
        "street_address": "456 Bug Avenue",
        "city": "Bugsville"
        # zip_code 필드 누락
    }
}

print("\n--- 유효하지 않은 데이터로 생성 시도 ---")
try:
    User(**invalid_data)
except ValidationError as e:
    print("유효성 검사 오류 발생!")
    print(e)

```

## 답변

### Pydantic이란?

Pydantic을 API 서버의 "문지기"라고 생각할 수 있습니다. 서버에 데이터가 들어올 때, Pydantic은 이 데이터가 우리가 약속한 규칙(타입, 길이, 형식 등)을 모두 지켰는지 꼼꼼하게 검사합니다. 규칙을 어긴 데이터는 입장을 거부하고 어떤 규칙을 어겼는지 알려주어, 애플리케이션 내부에서는 항상 깨끗하고 신뢰할 수 있는 데이터만 사용하도록 보장합니다.

### 자주 사용하는 문법과 기능

1.  **`BaseModel` 상속**: 모든 Pydantic 모델은 `BaseModel`을 상속받아 만듭니다.
    ```python
    from pydantic import BaseModel

    class MyModel(BaseModel):
        # 필드들을 여기에 정의합니다.
        pass
    ```

2.  **타입 힌트(Type Hints)**: 파이썬의 기본 타입 힌트를 사용하여 필드의 타입을 정의합니다.
    -   `str`, `int`, `float`, `bool`: 기본 자료형
    -   `List[str]`: 문자열 리스트
    -   `Dict[str, int]`: 키는 문자열, 값은 정수인 딕셔너리
    -   `Optional[str] = None`: 값이 없을 수도 있는 선택적 필드. 기본값은 `None`입니다.

3.  **`Field`로 추가 설정하기**: 필드에 더 상세한 규칙이나 정보를 추가하고 싶을 때 사용합니다.
    ```python
    from pydantic import Field

    name: str = Field(
        default="John Doe",      # 기본값 설정
        description="사용자의 전체 이름", # 설명 추가 (문서화에 유용)
        min_length=3,            # 최소 길이 제한
        max_length=50,           # 최대 길이 제한
        ge=1,                    # 숫자일 경우, 1보다 크거나 같아야 함 (Greater than or Equal)
        le=100,                  # 100보다 작거나 같아야 함 (Less than or Equal)
    )
    ```

4.  **특별한 데이터 타입**: Pydantic은 이메일, URL 등 특정 형식을 검증하는 유용한 타입들을 제공합니다.
    -   `EmailStr`: 이메일 주소 형식인지 검사합니다.
    -   `HttpUrl`: 유효한 HTTP URL인지 검사합니다.

5.  **중첩 모델 (Nested Models)**: 모델 안에 다른 모델을 포함하여 복잡한 JSON 구조를 표현할 수 있습니다.
    ```python
    class Address(BaseModel):
        city: str
        zip_code: str

    class User(BaseModel):
        name: str
        # Address 모델을 필드 타입으로 사용
        shipping_address: Address
    ```

6.  **데이터 변환 (직렬화)**: Pydantic 객체를 파이썬 딕셔너리나 JSON 문자열로 쉽게 변환할 수 있습니다.
    -   `.model_dump()`: 객체를 딕셔너리로 변환합니다.
    -   `.model_dump_json()`: 객체를 JSON 형식의 문자열로 변환합니다.

### 추가 자료

-   [Pydantic 공식 문서](https://docs.pydantic.dev/latest/)
-   [Pydantic - Field Validation](https://docs.pydantic.dev/latest/concepts/validation/)
-   [Pydantic - Validators](https://docs.pydantic.dev/latest/concepts/validators/)