# 05. Agentic vision과 코드 실행

## 공식 기능 요약 번역

ER 2 standard endpoint는 Python 코드를 생성·실행해 이미지를 crop/zoom하고 계산을 수행한 뒤 답할 수 있습니다. 공식 예시는 object detection, 아날로그 gauge, 액체 높이, 회로 기판 마킹, 이미지 annotation을 다룹니다.

## 1. 왜 code execution을 쓰는가

VLM이 한 번 보는 것보다 다음 loop가 정확할 수 있습니다.

```text
전체 이미지 관찰
→ 작은 관심 영역 발견
→ crop/zoom 코드 실행
→ 확대 이미지 재관찰
→ 눈금 계산
→ JSON 결과
```

특히 작은 글자, 계기판, 개수 세기, 기하 계산에서 유용합니다.

## 2. Thinking level

공식 guide는 단순 spatial task에는 낮은 thinking level, 복잡한 counting·추정에는 높은 수준을 권합니다.

| 수준 | 적합한 작업 | 대가 |
| --- | --- | --- |
| minimal/low | 단순 pointing, 명확한 box | 낮은 지연·비용 |
| medium | 일반적인 균형 | 중간 지연 |
| high | 복잡한 counting, gauge 계산 | 높은 지연·비용 |

항상 high를 쓰기보다 validation set에서 latency-accuracy curve를 측정합니다.

## 3. 코드 실행과 로봇 실행은 다르다

Gemini code execution은 이미지·데이터 계산용 sandbox입니다. 실제 로봇 SDK를 임의 코드 실행에 노출해서는 안 됩니다.

- built-in code execution: 비신뢰 계산 환경
- custom function calling: 허용한 로봇 기능만 호출
- local process execution: 기본 금지

모델이 생성한 Python을 robot PC에서 `exec()`하는 설계는 command injection과 오작동 위험이 큽니다.

## 4. Gauge 실무 패턴

```text
1. 계기판 종류와 단위 확인
2. 최소·최대 눈금 추출
3. needle angle 추정
4. calibration curve 적용
5. confidence/uncertainty 계산
6. 허용 범위와 교차 검증
7. 낮은 confidence면 사람 확인
```

VLM의 단일 숫자만 저장하지 말고 원본 이미지 checksum, crop, prompt version, 결과, 실제 sensor reading을 함께 기록합니다.

## 5. Structured output

JSON 형식만 prompt로 요구해도 markdown fence나 설명이 섞일 수 있습니다. ER 2 standard endpoint가 지원하는 structured output을 사용할 수 있다면 JSON Schema를 지정합니다.

예시 schema 개념:

```json
{
  "type": "object",
  "properties": {
    "value": {"type": "number"},
    "unit": {"type": "string"},
    "confidence": {"type": "number"}
  },
  "required": ["value", "unit", "confidence"]
}
```

스키마 일치는 의미가 정확하다는 뜻이 아닙니다. 범위·단위·시간 일관성을 다시 검증합니다.

## 6. Prompt injection

카메라 속 종이, 화면, QR에 “이전 지시를 무시하라” 같은 텍스트가 있을 수 있습니다.

대응:

- 이미지 속 텍스트는 데이터이지 명령이 아니라고 system instruction에 명시
- tool 최소 권한
- 사람 승인 없이 위험 tool 호출 금지
- 외부 검색과 robot tool 동시 사용 제한
- tool call argument와 결과 audit log
- canary·red-team 이미지 시험

## 7. 실습 아이디어

1. 사진 속 물체 count를 low/high thinking으로 비교
2. 같은 gauge를 5회 질의해 분산 측정
3. crop 유무의 정확도·latency 비교
4. 이미지에 prompt injection 문구를 넣고 tool이 차단되는지 시험
5. 허용 범위 밖 숫자를 safety gate가 거부하는지 시험

## 완료 조건

- code execution과 robot function calling을 구분한다.
- 생성 코드를 local robot host에서 실행하지 않는다.
- JSON schema 뒤에 semantic validation을 둔다.
- 시각 prompt injection을 위협 모델에 포함한다.

