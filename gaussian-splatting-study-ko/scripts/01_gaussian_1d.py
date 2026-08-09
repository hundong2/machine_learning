"""1차원 Gaussian의 평균과 표준편차가 곡선에 미치는 영향을 그린다."""

# 미래 Python에서도 현재 방식의 type hint 해석을 유지한다.
from __future__ import annotations

# 파일과 폴더 경로를 운영체제에 독립적으로 다루기 위해 Path를 가져온다.
from pathlib import Path

# 수치 배열과 지수함수를 사용하기 위해 NumPy를 np라는 별칭으로 가져온다.
import numpy as np
# 그래프를 그리고 이미지로 저장하기 위해 pyplot을 plt라는 별칭으로 가져온다.
import matplotlib.pyplot as plt


# 여러 입력 위치에서 1차원 Gaussian 값을 계산하는 함수를 정의한다.
def gaussian_1d(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """정규화 상수를 생략한 1차원 Gaussian 값을 반환한다."""
    # 표준편차가 0 이하이면 0으로 나누게 되므로 잘못된 입력을 즉시 거부한다.
    if sigma <= 0.0:
        # 호출자에게 입력 조건을 알려 주는 ValueError 예외를 발생시킨다.
        raise ValueError("sigma는 0보다 커야 합니다.")
    # 각 x를 평균에서 뺀 뒤 표준편차로 나누어 표준화 좌표를 계산한다.
    standardized = (x - mu) / sigma
    # ** 2는 제곱이고 np.exp는 배열의 각 원소에 지수함수를 적용한다.
    values = np.exp(-0.5 * standardized**2)
    # 계산한 Gaussian 배열을 함수를 호출한 곳에 돌려준다.
    return values


# 스크립트 전체 실습을 실행하는 main 함수를 정의한다.
def main() -> None:
    """세 표준편차의 Gaussian 곡선을 비교해 저장한다."""
    # -5부터 5까지를 1000개의 동일 간격 실수로 나눈 x축 배열을 만든다.
    x = np.linspace(-5.0, 5.0, 1000, dtype=np.float64)
    # Gaussian 중심을 0.0으로 정한다.
    mu = 0.0
    # 비교할 세 표준편차를 Python tuple로 저장한다.
    sigma_values = (0.5, 1.0, 2.0)
    # 결과 이미지를 저장할 outputs 폴더 경로 객체를 만든다.
    output_dir = Path("outputs")
    # parents=True는 상위 폴더도 만들고 exist_ok=True는 이미 있어도 오류를 내지 않는다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # 너비 10 inch, 높이 6 inch인 Matplotlib figure와 axes를 만든다.
    figure, axes = plt.subplots(figsize=(10, 6))
    # 각 sigma 값을 하나씩 꺼내 같은 좌표축에 곡선을 그린다.
    for sigma in sigma_values:
        # 현재 sigma로 모든 x 위치의 Gaussian 값을 계산한다.
        y = gaussian_1d(x=x, mu=mu, sigma=sigma)
        # f-string은 sigma 값을 label 문자열 안에 넣고 .1f는 소수 첫째 자리까지 표시한다.
        axes.plot(x, y, label=f"sigma={sigma:.1f}")
    # 그래프의 제목을 설정한다.
    axes.set_title("1D Gaussian: sigma controls spread")
    # 가로축 이름을 x로 설정한다.
    axes.set_xlabel("x")
    # 세로축 이름을 G(x)로 설정한다.
    axes.set_ylabel("G(x)")
    # 곡선 값을 쉽게 비교하도록 격자를 표시한다.
    axes.grid(True, alpha=0.3)
    # 각 곡선의 sigma를 보여 주는 범례를 표시한다.
    axes.legend()
    # 제목과 축 글자가 잘리지 않도록 여백을 자동 조정한다.
    figure.tight_layout()
    # 저장할 전체 파일 경로를 만든다.
    output_path = output_dir / "01_gaussian_1d.png"
    # dpi는 dots per inch이며 150으로 지정해 화면에서 읽기 좋은 해상도로 저장한다.
    figure.savefig(output_path, dpi=150)
    # 메모리를 반환하고 GUI 창이 불필요하게 남지 않도록 figure를 닫는다.
    plt.close(figure)
    # resolve는 상대 경로를 절대 경로로 바꾸며 결과 위치를 사용자에게 출력한다.
    print(f"저장 완료: {output_path.resolve()}")


# 이 파일을 직접 실행했을 때만 main 함수를 호출한다.
if __name__ == "__main__":
    # 위에서 정의한 전체 실습 절차를 시작한다.
    main()
