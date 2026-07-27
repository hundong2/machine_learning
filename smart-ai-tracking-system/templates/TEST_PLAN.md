# Test Plan

## 범위

TBD

## 테스트 환경

- commit:
- model checksum:
- 장치:
- mock/simulation/hardware:

## 단위 시험

| ID | 대상 | 입력 | 기대 결과 | 자동화 |
| --- | --- | --- | --- | --- |
| UT-01 | letterbox | TBD | TBD | 예 |

## 계약 시험

| ID | 경계 | 계약 | 실패 조건 |
| --- | --- | --- | --- |
| CT-01 | model↔runtime | shape/dtype | TBD |

## 통합 시험

| ID | 경로 | 입력 | 기대 결과 |
| --- | --- | --- | --- |
| IT-01 | video→detection | golden MP4 | TBD |

## Fault injection

| ID | 장애 | 주입 | 안전 동작 | 복구 |
| --- | --- | --- | --- | --- |
| FT-01 | camera loss | mock disconnect | TBD | TBD |

## 합격 기준

- [ ] 정확도 parity
- [ ] 성능 회귀
- [ ] 메모리 안정성
- [ ] timeout 안전 동작
- [ ] 재현 가능한 보고서

