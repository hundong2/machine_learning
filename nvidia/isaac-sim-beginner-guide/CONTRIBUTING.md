# 기여 가이드

이 프로젝트에 기여해주셔서 감사합니다! 🎉

## 기여하는 방법

### 1. 버그 리포트 / 개선 제안
- GitHub Issues에 등록해주세요
- 재현 가능한 예제 코드와 환경 정보(OS, GPU, Isaac Sim 버전)를 포함해주세요

### 2. Pull Request
1. 이 리포지토리를 Fork
2. 새 브랜치 생성: `git checkout -b feature/my-improvement`
3. 변경사항 커밋: `git commit -m "Add: 새 예제 추가"`
4. Push: `git push origin feature/my-improvement`
5. Pull Request 생성

## 코드 스타일

### Python
- PEP 8 준수
- 한국어 주석 환영 (이 가이드의 주 언어)
- 각 예제 파일 상단에 **학습 목표**를 명시
- `from isaacsim import SimulationApp` → `SimulationApp({"headless": False})`를 가장 먼저 호출

### 예제 템플릿
```python
"""
예제 N: 제목

학습 목표:
- 목표 1
- 목표 2

사전 요구사항:
- 이전 예제 번호

실행 방법:
    python 파일명.py
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# 나머지 import는 여기에
from isaacsim.core.api import World
# ...
```

## 문서 스타일
- 초보자 친화적으로 작성
- 수식은 LaTeX 문법 사용 (`$...$` 또는 `$$...$$`)
- 각 개념 설명에 간단한 Python 코드 예제 포함

## 질문
PR이나 이슈 생성 전에 질문이 있다면 GitHub Discussions를 이용해주세요.
