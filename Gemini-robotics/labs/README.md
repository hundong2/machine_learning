# Gemini Robotics ER 2 실습

핵심 로직은 API 키 없이 실행됩니다. 실제 API 예제는 perception과 평가만 수행하며 모터 명령을 보내지 않습니다.

## 폴더 구조

```text
labs/
├─ src/gemini_robotics_learning/
│  ├─ geometry.py      # [y,x] → pixel → 평면 좌표
│  ├─ schemas.py       # JSON 추출과 의미 schema
│  ├─ safety.py        # workspace·속도·거리·사람 근접 정책
│  └─ mock_robot.py    # idempotent tool executor
├─ examples/
│  ├─ 01_offline_spatial_grounding.py
│  ├─ 02_api_pointing.py
│  ├─ 03_safe_mock_orchestrator.py
│  ├─ 04_video_progress.py
│  └─ 05_streaming_skeleton.py
├─ notebooks/
│  └─ 01_coordinate_grounding.ipynb
└─ tests/
```

## 1. API 없이 실행

PowerShell:

```powershell
cd Gemini-robotics/labs
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python examples/01_offline_spatial_grounding.py
python examples/03_safe_mock_orchestrator.py
python examples/03_safe_mock_orchestrator.py --unsafe-demo
python examples/05_streaming_skeleton.py
```

Bash:

```bash
cd Gemini-robotics/labs
export PYTHONPATH=src
python -m unittest discover -s tests -v
python examples/01_offline_spatial_grounding.py
```

기대 결과:

- 정상 mock plan은 8개 tool result와 최종 pose를 출력합니다.
- `--unsafe-demo`는 `camera-post` 금지 구역 때문에 거부되고 STOPPED가 됩니다.
- 같은 call ID는 `duplicate_ignored`가 됩니다.

## 2. Notebook

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab notebooks/01_coordinate_grounding.ipynb
```

Notebook은 API를 호출하지 않으며 좌표 축, pixel 변환, homography, 시각화를 다룹니다.

## 3. Gemini API 준비

1. Google AI Studio에서 새 authorization key를 만듭니다.
2. key가 Gemini API로 제한됐는지 확인합니다.
3. 환경 변수에만 저장합니다.
4. quota·billing alert를 설정합니다.
5. 식별 가능한 사람이 없는 테스트 이미지를 사용합니다.

```powershell
$env:GEMINI_API_KEY = "YOUR_RESTRICTED_KEY"
$env:GEMINI_ROBOTICS_MODEL = "gemini-robotics-er-2-preview"
```

`.env.example`은 변수 이름만 보여 주는 예시입니다. 실제 값을 `.env.example`에 쓰지 않습니다.

## 4. Pointing API

```powershell
$env:PYTHONPATH = "src"
python examples/02_api_pointing.py `
  --image private-data/scene.jpg `
  --query "파란 블록"
```

프로그램은 다음만 수행합니다.

1. 이미지 존재와 API key 확인
2. ER 2에 이미지·prompt 전송
3. raw response 출력
4. JSON array, item count, `[y,x]`, 0~1000 검증
5. pixel 좌표 출력
6. `ACTION DISABLED`로 종료

## 5. 영상 분석 API

```powershell
python examples/04_video_progress.py `
  --video private-data/task.mp4 `
  --task "파란 블록을 그릇 안에 넣기" `
  --mode progress

python examples/04_video_progress.py `
  --video private-data/task.mp4 `
  --task "파란 블록을 그릇 안에 넣기" `
  --mode moment
```

`progress`는 다섯 bracket만 허용하고, `moment`는 0 이상의 초 또는 `null`만 허용합니다.

## 6. 실제 로봇 adapter를 만들기 전

- [ ] mock test 전체 통과
- [ ] simulator에서 workspace/충돌 검증
- [ ] tool call human confirmation
- [ ] hardware E-stop
- [ ] speed/force/joint limit
- [ ] 카메라와 robot frame calibration
- [ ] stale observation 차단
- [ ] actual sensor tool result
- [ ] timeout·중복 call 시험
- [ ] 개인정보와 운영 승인

## 7. 코드 읽는 순서

1. `geometry.py`: 생성형 결과를 결정론적 좌표로 바꾸는 경계
2. `schemas.py`: JSON 모양과 의미를 검증하는 경계
3. `safety.py`: 모델과 독립된 정책
4. `mock_robot.py`: allowlist, idempotency, stop latch
5. `03_safe_mock_orchestrator.py`: 전체 수직 흐름
6. API 예제: model adapter

## 문제 해결

### `ModuleNotFoundError: gemini_robotics_learning`

`labs` 폴더에서 `$env:PYTHONPATH = "src"`를 먼저 실행합니다.

### 403 PERMISSION_DENIED

키가 unrestricted이거나 Robotics ER 2 접근·지역·billing 조건을 만족하지 않을 수 있습니다. [API key 공식 문서](https://ai.google.dev/gemini-api/docs/api-key)와 AI Studio의 key 상태를 확인합니다.

### 모델을 찾을 수 없음

Preview ID가 변경됐을 수 있습니다. [Robotics ER overview](https://ai.google.dev/gemini-api/docs/robotics-overview)의 현재 endpoint를 확인하고 `GEMINI_ROBOTICS_MODEL`을 갱신합니다.

### JSON parsing 실패

Raw response를 보관하고 prompt-only JSON 대신 structured output을 검토합니다. 파싱 오류를 자동 행동으로 복구하지 말고 실행을 중단합니다.

