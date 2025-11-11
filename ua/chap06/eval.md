```markdown
# eval에서 {"__builtins__": {}, "math": math} 의미
eval 실행 시 사용할 ‘제한된 전역 네임스페이스(globals)’를 명시적으로 정의해 내장 기능을 차단하고 math 모듈만 노출하는 설정입니다.

## 예시 파일
[vscode에서 열기: ua/chap06/tool_calling.py](vscode://file/Users/donghun2/workspace/machine_learning/ua/chap06/tool_calling.py)

## 답변
`eval(expression, {"__builtins__": {}, "math": math})`에서 두 번째 인자는 전역 변수 사전(globals)입니다.

구성 요소 설명:
- "__builtins__": {}  
  - 파이썬 기본 내장함수/객체(예: open, len, print, __import__) 접근을 비웁니다.  
  - 악의적 입력이 파일 열기, 모듈 임포트 등을 시도해도 실패하게 하려는 최소한의 보호 장치입니다.
- "math": math  
  - math 모듈 객체를 ‘math’라는 이름으로 제공하여 사용자가 입력한 수식에서 math.sqrt, math.sin 등을 호출할 수 있게 합니다.
- 결과적으로 표현식 내부에서 사용 가능한 이름은 오직 ‘math’ 뿐이며, 내장 함수가 없어 임의 코드 실행 범위가 크게 줄어듭니다.

동작 요약:
1. 사용자가 입력한 문자열 수식을 사전 치환(sqrt → math.sqrt 등) 후
2. 제한된 환경(전역: { "__builtins__": {}, "math": math }, 지역: 기본 None)에서 평가
3. 수치 계산 결과만 반환

간단 비교 예:
````python
# 위험한 일반 eval (내장 모두 허용)
eval("__import__('os').listdir('.')")  # 디렉터리 정보 노출 가능

# 제한된 eval
safe_globals = {"__builtins__": {}, "math": math}
eval("math.sqrt(16) + math.cos(0)", safe_globals)  # 4.0 + 1.0 => 5.0
eval("__import__('os')", safe_globals)  # NameError 발생 (__import__ 제거됨)
````

주의점:
- 완전한 샌드박스 아님: math 객체를 통해 속성 탐색은 가능하지만 내장 import가 없어 확장 공격은 어렵게 됨.
- 더 강한 격리가 필요하면:  
  - ast.literal_eval (리터럴만)  
  - 전용 수식 파서(numexpr, simpleeval 등)  
  - 화이트리스트 기반 AST 검사
- "__builtins__": {} 대신 None을 쓰기도 하지만 일부 버전/상황에서 기대치 못한 접근이 열릴 수 있어 빈 dict 사용이 일반적입니다.

정리:
이 딕셔너리는 eval 실행 범위를 최소화하여 ‘수학 계산’만 허용하도록 만든 간단한 보안 완화 조치입니다.

### 추가 자료
- [Python 공식 문서 eval](https://docs.python.org/3/library/functions.html#eval)
- [ast.literal_eval 안전한 리터럴 평가](https://docs.python.org/3/library/ast.html#ast.literal_eval)
- [numexpr 빠른 수식 계산](https://numexpr.readthedocs.io/)
- [simpleeval 경량 표현식 평가기](https://github.com/danthedeckie/simpleeval)
```