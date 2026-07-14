# Bilevel Autoresearch: Meta-Autoresearching Itself

이 폴더는 arXiv:2603.23420v2 논문 학습용 자료입니다.

- 논문: Bilevel Autoresearch: Meta-Autoresearching Itself
- 저자: Yaonan Qu, Meng Lu
- arXiv: https://arxiv.org/abs/2603.23420
- 코드 저장소: https://github.com/EdwardOptimization/Bilevel-Autoresearch
- 원문 PDF: [2603.23420v2.pdf](2603.23420v2.pdf)

## 초급 기술자를 위한 읽는 순서

논문을 바로 읽기 어렵다면 아래 순서로 보면 됩니다.

1. 이 README의 "핵심 비유"를 먼저 읽습니다.
2. [paper_summary.ko.md](paper_summary.ko.md)의 "초급 독자용 먼저 보기"와 "핵심 구조"만 읽습니다.
3. [code/README.md](code/README.md)를 보고 실습 코드를 실행합니다.
4. 결과 표에서 Group A와 Group C의 차이만 먼저 확인합니다.
5. 그 다음 [study_guide.ko.md](study_guide.ko.md)의 용어 정리와 체크 질문을 봅니다.

## 핵심 비유

이 논문은 "일을 더 잘하는 방법을 찾는 사람"과 "그 사람이 일하는 방식을 고쳐 주는 관리자"의 관계로 이해하면 쉽습니다.

- Level 1은 실험 담당자입니다. 매번 설정을 하나 바꾸고 결과가 좋아졌는지 확인합니다.
- Level 1.5는 일정 관리자입니다. 너무 많이 실패한 방향은 잠시 막고 다른 방향을 보게 합니다.
- Level 2는 작업 방식 개선 담당자입니다. 실험 담당자가 같은 실수를 반복하면, 반복을 막는 새 규칙이나 도구를 만들어 끼워 넣습니다.

논문의 핵심 주장은 단순합니다. 실험 후보만 바꾸는 것보다, 후보를 고르는 방식 자체를 바꾸면 더 좋은 결과를 찾을 수 있다는 것입니다.

## 파일 구성

- [paper_summary.ko.md](paper_summary.ko.md): 논문 핵심 요약, 방법론, 실험 결과, 한계
- [study_guide.ko.md](study_guide.ko.md): 학습 순서, 용어 정리, 읽기 질문, 실습 과제
- [beginner_readability_check.ko.md](beginner_readability_check.ko.md): 초급 기술자 관점의 이해 가능성 점검 결과
- [code/README.md](code/README.md): 실습 코드 실행 방법
- [code/bilevel_autoresearch_demo.py](code/bilevel_autoresearch_demo.py): 로컬 실행 가능한 미니 bilevel autoresearch 시뮬레이터

## 빠른 실행

```powershell
cd D:\workspace\machine_learning\Research\Bilevel_Autoresearch_Meta-Autoresearching_Itself
python .\code\bilevel_autoresearch_demo.py
python .\code\bilevel_autoresearch_demo.py --trace C
```

이 실습은 논문 전체 시스템을 재현하지 않습니다. 논문은 LLM, 런타임 코드 생성, GPU 훈련, 검증 및 롤백을 포함하지만, 이 폴더의 코드는 그 구조를 작은 탐색 문제로 축소해 Level 1, Level 1.5, Level 2의 차이를 관찰할 수 있게 만든 교육용 예제입니다.

## 이해 목표

이 자료를 다 읽은 뒤에는 최소한 다음 질문에 답할 수 있어야 합니다.

- 왜 단순 반복 탐색은 같은 실패를 되풀이할 수 있는가?
- Level 1.5는 왜 전략 조정이고, Level 2는 메커니즘 변경인가?
- 논문에서 `TOTAL_BATCH_SIZE` 감소가 왜 중요한 발견이었는가?
- 동적으로 코드를 주입할 때 검증과 되돌리기가 왜 필요한가?
