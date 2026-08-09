"""네 교육용 스크립트가 새 작업 폴더에서 끝까지 실행되는지 검사한다."""

# 미래 Python에서도 현재 방식의 type hint 해석을 유지한다.
from __future__ import annotations

# 새 Python process로 각 스크립트를 실행하기 위해 subprocess를 가져온다.
import subprocess
# 현재 pytest를 실행 중인 Python interpreter 경로를 얻기 위해 sys를 가져온다.
import sys
# 파일 경로를 운영체제와 무관하게 조합하기 위해 Path를 가져온다.
from pathlib import Path

# pytest가 제공하는 임시 폴더 fixture의 자료형을 설명하기 위해 pytest를 가져온다.
import pytest


# 실행할 네 스크립트 파일명을 pytest parameter 목록으로 선언한다.
@pytest.mark.parametrize("script_name", ["01_gaussian_1d.py", "02_gaussian_2d.py", "03_camera_projection.py", "04_mini_splat_renderer.py"])
# 각 스크립트를 독립 process에서 실행하는 test 함수를 정의한다.
def test_script_runs(script_name: str, tmp_path: Path) -> None:
    """각 실습 스크립트가 종료 코드 0으로 완료되는지 검사한다."""
    # __file__은 이 test 파일이고 parents[1]은 학습 폴더의 root 경로이다.
    project_root = Path(__file__).resolve().parents[1]
    # root 아래 scripts 폴더와 현재 parameter 파일명을 결합한다.
    script_path = project_root / "scripts" / script_name
    # check=True는 스크립트가 실패할 때 pytest도 실패시키고 cwd는 출력 위치를 임시 폴더로 격리한다.
    subprocess.run([sys.executable, str(script_path)], cwd=tmp_path, check=True)
    # 각 스크립트가 공통으로 outputs 폴더를 만들었는지 검사한다.
    assert (tmp_path / "outputs").is_dir()
