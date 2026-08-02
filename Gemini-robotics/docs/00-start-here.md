# 00. 시작하기

## 이 자료가 번역하는 범위

원문을 문장별로 복제하는 대신, 공식 랜딩 페이지와 연결 문서의 핵심을 한국어로 해설하고 실무 설계로 확장합니다. 각 사실은 [공식 자료 지도](./12-reference-map.md)에서 원문과 연결됩니다.

## 2026년 8월 현재의 중요한 변화

원래 요청한 랜딩 페이지는 **Gemini Robotics 1.5 VLA**를 소개합니다. 그러나 개발자가 Gemini API로 직접 호출할 최신 모델은 **Gemini Robotics ER 2**입니다.

- `gemini-robotics-er-2-preview`: 정적 이미지·영상, structured output, code execution, function calling에 적합
- `gemini-robotics-er-2-streaming-preview`: Live API 기반 연속 오디오·이미지와 낮은 지연의 tool calling에 적합
- `gemini-robotics-er-1.6-preview`: 2026년 8월 말 종료 예정

튜토리얼에서 1.5나 1.6 모델 ID를 발견하면 그대로 복사하지 말고 현재 모델 페이지를 확인합니다.

## 가능한 것과 불가능한 것

### 공개 API로 가능한 것

- 이미지 속 물체 pointing
- 정규화 bounding box
- 영상 속 물체 추적과 작업 완료 시점 탐색
- 5단계 progress bracket 분류
- 계기판·액체 높이·부품 마킹 해석
- 사용자 정의 로봇 함수를 호출하는 고수준 계획
- streaming endpoint를 통한 저지연 양방향 session

### 일반 공개 API만으로 할 수 없는 것

- Gemini Robotics 1.5 VLA weight 다운로드
- VLA를 임의 로봇에 직접 fine-tune
- On-Device 2 SDK의 unrestricted 사용
- 모델 출력만으로 safety-rated 모터 제어 보장
- 단일 2D point만으로 정확한 6-DoF grasp 실행

## 준비 지식

처음 시작한다면 아래만 알면 됩니다.

- Python 함수와 클래스
- JSON이 key-value 구조라는 사실
- 이미지의 왼쪽 위가 일반적으로 `(x=0, y=0)`이라는 사실
- 로봇 행동은 실패하면 물리적 피해를 만들 수 있다는 사실

모르는 수학은 실습 중 필요한 만큼 설명합니다. 선형대수, 카메라 캘리브레이션, 역기구학은 2D 좌표 변환을 이해한 다음 확장합니다.

## 환경

오프라인 실습:

- Python 3.11 이상
- 표준 라이브러리만으로 핵심 테스트 실행 가능
- 노트북 시각화를 위해 NumPy, Matplotlib 사용

API 실습:

- `google-genai` 공식 SDK
- Gemini API authorization key
- 사용 권한이 있는 이미지 또는 영상
- 네트워크와 API 비용 한도

## 권장 실행 순서

1. `python -m unittest discover -s tests -v`
2. 오프라인 좌표 grounding 실행
3. 모의 로봇의 정상 plan 실행
4. 금지 영역 plan이 차단되는지 확인
5. API pointing 실행
6. API 결과를 파싱만 하고 실제 행동은 금지
7. capstone에서 승인·관측·복구 loop 추가

## 왜 mock부터 시작하는가

LLM/VLM 출력은 확률적입니다. 같은 입력에도 위치나 계획이 조금씩 달라질 수 있습니다. 실제 모터를 곧바로 연결하면 다음 문제가 한꺼번에 섞입니다.

- 모델 오류
- 좌표계 오류
- API timeout
- 카메라 캘리브레이션 오류
- 역기구학 실패
- 모터 드라이버 오류
- 물리적 충돌

mock은 모델과 안전 계약을 먼저 검증해 원인 범위를 줄입니다.

## 첫 번째 체크포인트

- [ ] VLA와 ER의 차이를 한 문장으로 설명한다.
- [ ] 최신 모델 ID를 공식 문서에서 확인한다.
- [ ] API 키가 Git에 포함되지 않는다.
- [ ] 실제 행동은 기본 비활성이다.
- [ ] 사람이나 개인정보가 담긴 데이터를 쓰기 전에 동의를 확인한다.

