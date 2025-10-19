# Pydantic 핵심 정리와 자주 쓰는 구성 요소
타입 힌트를 기반으로 데이터 유효성 검사·변환·직렬화를 자동화하는 라이브러리(Pydantic v2 기준).

## 예시 파일
[vscode에서 열기: pydantic_examples.py](vscode://file/Users/donghun2/workspace/machine_learning/ua/chap04/structured-output-agent/pydantic_examples.py)

## 답변
Pydantic은 BaseModel에 타입 힌트를 선언하면 입력 데이터를 자동으로 검증·변환하고, Python 객체를 dict/JSON으로 쉽게 직렬화합니다. 자주 쓰는 기능은 다음과 같습니다.

- 핵심 클래스/함수
  - BaseModel: 데이터 모델 정의의 기본
  - Field(...): 기본값/제약조건(길이, 범위, 설명 등) 지정
  - ValidationError: 검증 실패시 예외
  - field_validator, model_validator: 필드/모델 단위 커스텀 검증
  - computed_field: 필드 계산(읽기 전용 파생 값)
  - model_dump(), model_dump_json(): 직렬화
  - model_json_schema(): JSON Schema 생성
  - ConfigDict, model_config: 전역 설정(strict, alias, populate_by_name 등)
- 유용한 타입
  - EmailStr, HttpUrl/AnyUrl, PastDate/FutureDate, StrictInt/StrictStr
  - typing: Optional[T], Literal[...], Union/|, Annotated[...] 등
- 설정 로딩
  - pydantic-settings(BaseSettings): .env/환경변수 기반 설정 관리

아래 예시는 핵심 기능을 한 곳에 모았습니다.

````python
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, Literal, Annotated
from pydantic import (
    BaseModel, Field, ValidationError,
    field_validator, model_validator, computed_field, ConfigDict,
    EmailStr, HttpUrl
)

# 1) 중첩 모델 + 기본 제약
class Address(BaseModel):
    street: str
    city: str
    postal_code: Annotated[str, Field(pattern=r"^\d{5}$")]  # 5자리 우편번호

# 2) 사용자 모델: 필드 제약 + 커스텀 검증 + 파생 필드
class User(BaseModel):
    # 전역 설정: 엄격 모드/별칭 허용 등 필요 시 조정
    model_config = ConfigDict(extra="forbid")  # 선언되지 않은 필드 금지

    id: int
    username: Annotated[str, Field(min_length=3, max_length=30, description="3~30자")]
    email: EmailStr
    website: Optional[HttpUrl] = None
    first_name: str = Field(min_length=1)
    last_name: str = Field(min_length=1)
    role: Literal["admin", "member", "guest"] = "member"
    tags: list[str] = Field(default_factory=list)
    address: Address
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 개별 필드 검증/정규화
    @field_validator("username")
    @classmethod
    def no_space_and_lower(cls, v: str) -> str:
        if " " in v:
            raise ValueError("username에 공백 불가")
        return v.lower()

    # 파생(계산) 필드: 저장되진 않지만 읽을 수 있음
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

# 3) 교차 필드 검증(기간 논리 등)
class Project(BaseModel):
    name: str
    start: date
    end: date

    @model_validator(mode="after")
    def check_date_range(self) -> "Project":
        if self.end < self.start:
            raise ValueError("end는 start 이후여야 합니다.")
        return self

# 4) 직렬화/스키마
def demo_serialization(u: User) -> None:
    as_dict = u.model_dump(exclude_none=True)          # None 필드 제외
    as_json = u.model_dump_json(indent=2, ensure_ascii=False)
    schema = u.model_json_schema()                      # JSON Schema
    print("DICT:", as_dict)
    print("JSON:", as_json)
    print("SCHEMA keys:", list(schema.keys()))

# 5) 사용 예 + 오류 처리
def main():
    try:
        user = User(
            id=1,
            username="CoPilot",
            email="contact@example.com",
            first_name="GitHub",
            last_name="Copilot",
            role="member",
            tags=["ai", "assistant"],
            address=Address(street="1 AI St", city="Seoul", postal_code="12345"),
        )
        print("full_name:", user.full_name)
        demo_serialization(user)
    except ValidationError as e:
        print("ValidationError:", e)

    # 교차 검증 오류 예시
    try:
        Project(name="Demo", start=date(2025, 1, 10), end=date(2024, 12, 31))
    except ValidationError as e:
        print("Project ValidationError:", e)

if __name__ == "__main__":
    main()
````

추가로 환경변수 기반 설정이 필요하면 pydantic-settings를 사용합니다.

````python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")
    openai_api_key: str
    base_url: AnyUrl | None = None

# .env 예)
# APP_OPENAI_API_KEY=sk-...
# APP_BASE_URL=http://localhost:50505/v1

if __name__ == "__main__":
    s = AppSettings()  # .env/환경변수에서 자동 로드
    print(s.model_dump())
````

요점
- 입력 → BaseModel로 검증/정규화 → 안전한 객체
- 사용 → 직렬화(model_dump/json), 스키마 생성(model_json_schema)
- 세밀한 검증 → field_validator/model_validator/computed_field
- 운영 설정 → pydantic-settings로 .env/환경변수 로딩

### 추가 자료
- https://docs.pydantic.dev/latest/
- https://docs.pydantic.dev/latest/concepts/validators/
- https://docs.pydantic.dev/latest/concepts/serialization/
- https://docs.pydantic.dev/latest/usage/pydantic_settings/