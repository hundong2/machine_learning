"""회전된 2차원 이방성 Gaussian을 계산하고 등고선으로 그린다."""

# 미래 Python에서도 현재 방식의 type hint 해석을 유지한다.
from __future__ import annotations

# 파일과 폴더 경로를 안전하게 조합하기 위해 Path를 가져온다.
from pathlib import Path

# 배열, 행렬, 지수함수 계산을 위해 NumPy를 np라는 별칭으로 가져온다.
import numpy as np
# 2D Gaussian heatmap과 등고선을 그리기 위해 pyplot을 plt라는 별칭으로 가져온다.
import matplotlib.pyplot as plt


# 각도를 degree에서 radian으로 바꾼 뒤 2×2 회전행렬을 만드는 함수를 정의한다.
def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """반시계 방향 2D 회전행렬을 반환한다."""
    # np.deg2rad는 degree 단위 각도를 삼각함수가 사용하는 radian으로 변환한다.
    angle_radians = np.deg2rad(angle_degrees)
    # 현재 각도의 cosine 값을 계산한다.
    cosine = np.cos(angle_radians)
    # 현재 각도의 sine 값을 계산한다.
    sine = np.sin(angle_radians)
    # 표준 2D 회전행렬을 float64 NumPy 배열로 만든다.
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    # 완성한 회전행렬을 반환한다.
    return rotation


# 두 scale과 회전으로 2×2 covariance matrix를 만드는 함수를 정의한다.
def covariance_2d(scale_x: float, scale_y: float, angle_degrees: float) -> np.ndarray:
    """주축 scale과 회전으로 2D covariance를 만든다."""
    # scale이 0 이하이면 유효한 Gaussian을 만들 수 없으므로 입력을 거부한다.
    if scale_x <= 0.0 or scale_y <= 0.0:
        # 잘못된 scale 조건을 설명하는 ValueError를 발생시킨다.
        raise ValueError("scale_x와 scale_y는 0보다 커야 합니다.")
    # 두 축 분산은 표준편차 역할의 scale을 제곱한 값이다.
    variance = np.diag([scale_x**2, scale_y**2])
    # 지정한 각도에 대응하는 회전행렬을 계산한다.
    rotation = rotation_matrix_2d(angle_degrees)
    # @는 행렬곱이고 .T는 transpose이므로 R S R^T로 covariance를 회전한다.
    covariance = rotation @ variance @ rotation.T
    # 회전된 covariance matrix를 반환한다.
    return covariance


# 모든 픽셀 좌표에서 2D Gaussian 값을 계산하는 함수를 정의한다.
def gaussian_2d(pixel_grid: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """shape가 (H, W, 2)인 좌표 격자에 2D Gaussian을 평가한다."""
    # broadcasting으로 모든 픽셀 좌표에서 shape (2,)인 중심을 뺀다.
    delta = pixel_grid - mean
    # covariance의 역행렬을 계산해 Mahalanobis distance에 사용한다.
    inverse_covariance = np.linalg.inv(covariance)
    # einsum은 각 픽셀의 delta^T inverse_covariance delta를 한 번에 계산한다.
    squared_distance = np.einsum("...i,ij,...j->...", delta, inverse_covariance, delta)
    # 중심에서의 값이 1이 되는 Gaussian weight를 계산한다.
    weight = np.exp(-0.5 * squared_distance)
    # shape가 (H, W)인 weight 배열을 반환한다.
    return weight


# 전체 시각화 실습을 실행하는 main 함수를 정의한다.
def main() -> None:
    """회전된 2D Gaussian을 이미지로 저장한다."""
    # 출력 영상의 높이를 300 pixel로 정한다.
    height = 300
    # 출력 영상의 너비를 400 pixel로 정한다.
    width = 400
    # 0부터 height-1까지의 세로 좌표와 0부터 width-1까지의 가로 좌표를 만든다.
    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width]
    # dstack은 x와 y를 마지막 축에 쌓아 shape (H, W, 2)의 좌표 격자를 만든다.
    pixel_grid = np.dstack((x_coordinates, y_coordinates)).astype(np.float64)
    # Gaussian 중심을 영상 중앙의 (x, y) 순서로 정한다.
    mean = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    # x축 scale 70, y축 scale 25, 반시계 35도 회전 covariance를 만든다.
    covariance = covariance_2d(scale_x=70.0, scale_y=25.0, angle_degrees=35.0)
    # 모든 픽셀에서 2D Gaussian weight를 계산한다.
    weight = gaussian_2d(pixel_grid=pixel_grid, mean=mean, covariance=covariance)
    # 결과 저장 폴더 경로를 만든다.
    output_dir = Path("outputs")
    # 상위 폴더가 없어도 만들고 기존 폴더가 있어도 계속한다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # 너비 10 inch, 높이 7 inch인 figure와 axes를 만든다.
    figure, axes = plt.subplots(figsize=(10, 7))
    # origin="upper"로 영상처럼 y=0이 위쪽에 오도록 heatmap을 그린다.
    image = axes.imshow(weight, cmap="viridis", origin="upper")
    # Gaussian 값 0.1부터 0.9까지의 등고선을 흰색 선으로 추가한다.
    axes.contour(weight, levels=np.linspace(0.1, 0.9, 5), colors="white", linewidths=0.8)
    # 좌표 중심을 빨간 x 표식으로 표시한다.
    axes.scatter(mean[0], mean[1], color="red", marker="x", s=100, label="mean")
    # figure 옆에 Gaussian weight의 색상 범례를 추가한다.
    figure.colorbar(image, ax=axes, label="Gaussian weight")
    # 그래프 제목을 설정한다.
    axes.set_title("2D anisotropic Gaussian")
    # 가로축 이름을 pixel x로 설정한다.
    axes.set_xlabel("pixel x")
    # 세로축 이름을 pixel y로 설정한다.
    axes.set_ylabel("pixel y")
    # 빨간 중심 표식의 범례를 표시한다.
    axes.legend()
    # 그래프 요소가 잘리지 않도록 여백을 조정한다.
    figure.tight_layout()
    # 저장할 결과 파일 경로를 만든다.
    output_path = output_dir / "02_gaussian_2d.png"
    # 150 dpi 해상도의 PNG 이미지로 저장한다.
    figure.savefig(output_path, dpi=150)
    # 더 이상 필요하지 않은 figure를 닫는다.
    plt.close(figure)
    # 사용한 covariance matrix를 확인할 수 있도록 출력한다.
    print("covariance matrix:\n", covariance)
    # 생성된 파일의 절대 경로를 출력한다.
    print(f"저장 완료: {output_path.resolve()}")


# 이 파일이 import가 아니라 직접 실행된 경우에만 아래 블록을 실행한다.
if __name__ == "__main__":
    # 전체 2D Gaussian 실습을 시작한다.
    main()
