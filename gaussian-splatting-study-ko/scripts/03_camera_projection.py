"""3D 세계 좌표를 pinhole camera의 2D pixel 좌표로 투영한다."""

# 미래 Python에서도 현재 방식의 type hint 해석을 유지한다.
from __future__ import annotations

# 출력 파일 경로를 다루기 위해 Path를 가져온다.
from pathlib import Path

# 벡터와 행렬 계산을 위해 NumPy를 np라는 별칭으로 가져온다.
import numpy as np
# 투영 결과를 산점도로 그리기 위해 pyplot을 plt라는 별칭으로 가져온다.
import matplotlib.pyplot as plt


# 세계 좌표 점들을 카메라 좌표로 바꾸는 함수를 정의한다.
def world_to_camera(points_world: np.ndarray, rotation_cw: np.ndarray, translation_cw: np.ndarray) -> np.ndarray:
    """camera-from-world 외부 파라미터를 점들에 적용한다."""
    # points_world @ rotation_cw.T는 각 행 벡터 점에 회전을 적용한다.
    rotated_points = points_world @ rotation_cw.T
    # shape (3,) 이동 벡터는 broadcasting되어 모든 회전된 점에 더해진다.
    points_camera = rotated_points + translation_cw
    # shape (N, 3)인 카메라 좌표 점들을 반환한다.
    return points_camera


# 카메라 좌표 점들을 pixel 좌표로 바꾸는 함수를 정의한다.
def camera_to_pixel(points_camera: np.ndarray, intrinsic_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pinhole projection을 적용하고 pixel 좌표와 유효 mask를 반환한다."""
    # 각 점의 세 번째 좌표인 Z 깊이가 0보다 큰지 검사한다.
    valid_mask = points_camera[:, 2] > 0.0
    # 카메라 앞에 있는 점만 남긴다.
    visible_points = points_camera[valid_mask]
    # 3D 점에 K^T를 곱해 homogeneous image coordinate를 계산한다.
    homogeneous_pixels = visible_points @ intrinsic_matrix.T
    # 앞의 두 좌표를 세 번째 좌표로 나눠 실제 (u, v) pixel 좌표를 얻는다.
    pixels = homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:3]
    # pixel 좌표와 원래 점들 중 유효한 점을 표시하는 boolean mask를 반환한다.
    return pixels, valid_mask


# 카메라 투영 예제를 실행하는 main 함수를 정의한다.
def main() -> None:
    """깊이가 서로 다른 3D 점을 투영해 시각화한다."""
    # 점마다 x, y, z를 한 행에 저장한 shape (7, 3) 세계 좌표 배열을 만든다.
    points_world = np.array([[-1.0, -0.5, 2.0], [0.0, -0.5, 2.0], [1.0, -0.5, 2.0], [-1.0, 0.5, 4.0], [0.0, 0.5, 4.0], [1.0, 0.5, 4.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    # 이 예제에서는 세계축과 카메라축이 같으므로 3×3 단위 회전행렬을 사용한다.
    rotation_cw = np.eye(3, dtype=np.float64)
    # 카메라 위치 이동이 없으므로 영벡터를 사용한다.
    translation_cw = np.zeros(3, dtype=np.float64)
    # 출력 영상 너비를 640 pixel로 정한다.
    image_width = 640
    # 출력 영상 높이를 480 pixel로 정한다.
    image_height = 480
    # x와 y 방향 focal length를 각각 500 pixel로 정한다.
    focal_length = 500.0
    # 주점을 영상 중앙으로 두는 3×3 camera intrinsic matrix K를 만든다.
    intrinsic_matrix = np.array([[focal_length, 0.0, image_width / 2.0], [0.0, focal_length, image_height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # 세계 좌표 점들을 카메라 좌표로 변환한다.
    points_camera = world_to_camera(points_world, rotation_cw, translation_cw)
    # 카메라 앞의 점을 pixel 좌표로 투영하고 유효 mask를 받는다.
    pixels, valid_mask = camera_to_pixel(points_camera, intrinsic_matrix)
    # 카메라 앞에 있는 점들의 깊이를 색상 값으로 사용하기 위해 선택한다.
    visible_depths = points_camera[valid_mask, 2]
    # 출력 디렉터리 경로를 만든다.
    output_dir = Path("outputs")
    # 출력 디렉터리와 필요한 상위 디렉터리를 생성한다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # 640×480 화면 비율과 비슷한 figure와 axes를 만든다.
    figure, axes = plt.subplots(figsize=(8, 6))
    # 투영된 점을 깊이에 따라 다른 색으로 표시한다.
    scatter = axes.scatter(pixels[:, 0], pixels[:, 1], c=visible_depths, cmap="viridis", s=100)
    # 각 점의 깊이를 색으로 읽을 수 있도록 colorbar를 추가한다.
    figure.colorbar(scatter, ax=axes, label="camera depth Z")
    # 화면 좌표의 가로 범위를 0부터 image_width까지로 설정한다.
    axes.set_xlim(0, image_width)
    # 영상 좌표처럼 y=0이 위가 되도록 높이 범위를 역순으로 설정한다.
    axes.set_ylim(image_height, 0)
    # 가로축의 의미를 pixel u로 표시한다.
    axes.set_xlabel("pixel u")
    # 세로축의 의미를 pixel v로 표시한다.
    axes.set_ylabel("pixel v")
    # 그림의 제목을 설정한다.
    axes.set_title("Pinhole camera projection")
    # 좌표를 읽기 쉽게 격자를 표시한다.
    axes.grid(True, alpha=0.3)
    # figure 내부 여백을 자동 조정한다.
    figure.tight_layout()
    # 결과 이미지 파일 경로를 만든다.
    output_path = output_dir / "03_camera_projection.png"
    # figure를 150 dpi PNG 파일로 저장한다.
    figure.savefig(output_path, dpi=150)
    # figure가 차지한 메모리를 반환한다.
    plt.close(figure)
    # 원래 점 중 카메라 앞에 있는 점의 개수를 출력한다.
    print(f"유효한 점: {int(valid_mask.sum())}/{len(points_world)}")
    # 각 유효 3D 점과 대응하는 2D pixel 좌표를 함께 반복한다.
    for point_3d, pixel_2d in zip(points_camera[valid_mask], pixels, strict=True):
        # NumPy 배열을 읽기 좋은 문자열로 바꿔 대응 관계를 출력한다.
        print(f"camera {point_3d} -> pixel {pixel_2d}")
    # 저장된 결과 파일의 절대 경로를 출력한다.
    print(f"저장 완료: {output_path.resolve()}")


# 이 파일을 직접 실행했을 때만 main 함수를 실행한다.
if __name__ == "__main__":
    # 카메라 투영 전체 실습을 시작한다.
    main()
