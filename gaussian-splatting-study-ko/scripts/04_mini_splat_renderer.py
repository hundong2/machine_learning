"""작은 3D Gaussian 집합을 CPU에서 투영하고 alpha compositing한다."""

# 미래 Python에서도 현재 방식의 type hint 해석을 유지한다.
from __future__ import annotations

# 출력 파일 경로와 폴더를 다루기 위해 Path를 가져온다.
from pathlib import Path

# 배열, 투영, Gaussian weight 계산을 위해 NumPy를 np라는 별칭으로 가져온다.
import numpy as np
# RGB와 alpha 결과를 PNG로 저장하기 위해 Pillow의 Image를 가져온다.
from PIL import Image


# 0~1 실수 RGB 배열을 0~255 uint8 영상으로 저장하는 함수를 정의한다.
def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """float RGB 배열을 8-bit PNG 파일로 저장한다."""
    # np.clip은 수치 오차로 0~1 밖에 나간 값을 유효 영상 범위로 제한한다.
    clipped = np.clip(image, 0.0, 1.0)
    # 255를 곱하고 반올림해 8-bit unsigned integer 배열로 변환한다.
    image_uint8 = np.rint(clipped * 255.0).astype(np.uint8)
    # NumPy 배열을 RGB 모드 Pillow 이미지 객체로 만든다.
    pillow_image = Image.fromarray(image_uint8, mode="RGB")
    # Pillow 이미지 객체를 지정한 PNG 경로에 저장한다.
    pillow_image.save(path)


# 0~1 실수 단일 채널 배열을 0~255 grayscale 영상으로 저장하는 함수를 정의한다.
def save_gray_image(path: Path, image: np.ndarray) -> None:
    """float grayscale 배열을 8-bit PNG 파일로 저장한다."""
    # grayscale 값을 0~1 범위로 제한한다.
    clipped = np.clip(image, 0.0, 1.0)
    # grayscale 실수를 0~255 범위 uint8 정수로 변환한다.
    image_uint8 = np.rint(clipped * 255.0).astype(np.uint8)
    # NumPy 배열을 L 모드의 단일 채널 Pillow 이미지로 만든다.
    pillow_image = Image.fromarray(image_uint8, mode="L")
    # 생성한 grayscale 이미지를 지정한 경로에 저장한다.
    pillow_image.save(path)


# 작은 Gaussian 목록을 화면에 렌더링하는 교육용 함수를 정의한다.
def render_gaussians(means_3d: np.ndarray, scales_3d: np.ndarray, colors_rgb: np.ndarray, opacities: np.ndarray, intrinsic_matrix: np.ndarray, image_height: int, image_width: int, background_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """등방성 scale 근사를 사용해 front-to-back으로 Gaussian을 합성한다."""
    # 모든 Gaussian 관련 배열의 첫 축 길이가 같은지 검사한다.
    if not (len(means_3d) == len(scales_3d) == len(colors_rgb) == len(opacities)):
        # Gaussian 속성 개수가 다르면 대응 관계가 없으므로 ValueError를 발생시킨다.
        raise ValueError("모든 Gaussian 속성 배열의 첫 번째 차원 길이가 같아야 합니다.")
    # np.indices는 shape (2, H, W)의 정수 격자를 만들고 각각 y와 x로 나눈다.
    pixel_y, pixel_x = np.indices((image_height, image_width), dtype=np.float64)
    # 출력 RGB를 모두 0인 shape (H, W, 3) 배열로 초기화한다.
    accumulated_rgb = np.zeros((image_height, image_width, 3), dtype=np.float64)
    # 아직 아무것도 가리지 않았으므로 transmittance를 모두 1로 초기화한다.
    transmittance = np.ones((image_height, image_width, 1), dtype=np.float64)
    # 카메라 깊이 z가 작은 Gaussian부터 처리할 수 있도록 index를 오름차순 정렬한다.
    sorted_indices = np.argsort(means_3d[:, 2])
    # 정렬된 index를 하나씩 꺼내 front-to-back 합성을 수행한다.
    for gaussian_index in sorted_indices:
        # 현재 Gaussian의 3D 중심을 가져온다.
        mean_3d = means_3d[gaussian_index]
        # 현재 Gaussian의 z 깊이를 별도 변수로 가져온다.
        depth = mean_3d[2]
        # 카메라 뒤 또는 카메라 평면 위의 Gaussian은 투영할 수 없으므로 건너뛴다.
        if depth <= 0.0:
            # continue는 현재 반복을 끝내고 다음 Gaussian으로 이동한다.
            continue
        # homogeneous 좌표를 만들기 위해 K와 3D 중심을 행렬-벡터 곱한다.
        homogeneous_pixel = intrinsic_matrix @ mean_3d
        # homogeneous x를 세 번째 성분으로 나눠 실제 pixel u를 구한다.
        pixel_u = homogeneous_pixel[0] / homogeneous_pixel[2]
        # homogeneous y를 세 번째 성분으로 나눠 실제 pixel v를 구한다.
        pixel_v = homogeneous_pixel[1] / homogeneous_pixel[2]
        # 세 3D scale의 평균을 교육용 등방성 world scale로 사용한다.
        world_scale = float(np.mean(scales_3d[gaussian_index]))
        # K[0,0]은 x focal length이며 perspective 관계로 world scale을 pixel scale로 바꾼다.
        screen_sigma = intrinsic_matrix[0, 0] * world_scale / depth
        # 수치적으로 너무 작은 Gaussian을 피하도록 pixel sigma의 최솟값을 1로 제한한다.
        screen_sigma = max(screen_sigma, 1.0)
        # 모든 pixel의 x 좌표에서 Gaussian 중심 u를 빼 가로 차이를 구한다.
        delta_x = pixel_x - pixel_u
        # 모든 pixel의 y 좌표에서 Gaussian 중심 v를 빼 세로 차이를 구한다.
        delta_y = pixel_y - pixel_v
        # 등방성 2D Gaussian의 제곱 Mahalanobis distance를 계산한다.
        squared_distance = (delta_x**2 + delta_y**2) / (screen_sigma**2)
        # 거리로부터 각 pixel에서의 Gaussian weight를 계산한다.
        gaussian_weight = np.exp(-0.5 * squared_distance)
        # Gaussian 자체 opacity와 화면 weight를 곱해 pixel별 alpha를 만든다.
        alpha = opacities[gaussian_index] * gaussian_weight
        # 한 Gaussian이 pixel을 완전히 막아 gradient나 곱셈이 불안정해지지 않도록 0.99로 제한한다.
        alpha = np.clip(alpha, 0.0, 0.99)
        # [:, :, None]은 shape (H,W)에 색 채널용 길이 1 축을 추가한다.
        alpha_with_channel = alpha[:, :, None]
        # [None, None, :]은 shape (3,)인 색을 모든 H×W pixel에 broadcasting할 준비를 한다.
        color_with_image_axes = colors_rgb[gaussian_index][None, None, :]
        # 앞을 통과한 빛 × 현재 alpha × 현재 색을 출력 RGB에 더한다.
        accumulated_rgb += transmittance * alpha_with_channel * color_with_image_axes
        # 현재 Gaussian에 막히지 않고 뒤로 진행하는 빛의 비율을 갱신한다.
        transmittance *= 1.0 - alpha_with_channel
    # 모든 Gaussian 뒤에 남은 빛에 배경색을 곱해 최종 RGB를 완성한다.
    accumulated_rgb += transmittance * background_color[None, None, :]
    # 최종 누적 alpha는 1에서 남은 transmittance를 뺀 값이다.
    accumulated_alpha = 1.0 - transmittance[:, :, 0]
    # 완성한 RGB 배열과 alpha 배열을 함께 반환한다.
    return accumulated_rgb, accumulated_alpha


# 예제 장면 생성과 렌더링을 담당하는 main 함수를 정의한다.
def main() -> None:
    """다섯 Gaussian으로 작은 장면을 렌더링한다."""
    # 출력 영상 높이를 480 pixel로 설정한다.
    image_height = 480
    # 출력 영상 너비를 640 pixel로 설정한다.
    image_width = 640
    # x와 y 방향에 공통으로 사용할 focal length를 520 pixel로 설정한다.
    focal_length = 520.0
    # focal length와 영상 중앙의 principal point로 intrinsic matrix K를 만든다.
    intrinsic_matrix = np.array([[focal_length, 0.0, image_width / 2.0], [0.0, focal_length, image_height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    # Gaussian 다섯 개의 중심을 카메라 좌표계 (x,y,z)로 저장한다.
    means_3d = np.array([[-0.7, 0.1, 2.6], [0.0, -0.1, 2.2], [0.7, 0.15, 3.0], [-0.35, -0.55, 3.4], [0.45, -0.5, 3.7]], dtype=np.float64)
    # 각 Gaussian의 세 축 scale을 world unit으로 저장한다.
    scales_3d = np.array([[0.32, 0.32, 0.32], [0.38, 0.38, 0.38], [0.40, 0.40, 0.40], [0.45, 0.45, 0.45], [0.48, 0.48, 0.48]], dtype=np.float64)
    # 각 Gaussian의 RGB 색을 0~1 실수 범위로 저장한다.
    colors_rgb = np.array([[0.95, 0.20, 0.20], [0.20, 0.85, 0.35], [0.20, 0.35, 0.95], [0.95, 0.75, 0.15], [0.75, 0.25, 0.90]], dtype=np.float64)
    # 각 Gaussian의 기본 불투명도를 0~1 범위로 저장한다.
    opacities = np.array([0.82, 0.88, 0.84, 0.72, 0.76], dtype=np.float64)
    # 배경색을 아주 어두운 청회색 RGB로 설정한다.
    background_color = np.array([0.025, 0.035, 0.055], dtype=np.float64)
    # 교육용 renderer를 호출해 RGB와 alpha 영상을 계산한다.
    rendered_rgb, rendered_alpha = render_gaussians(means_3d, scales_3d, colors_rgb, opacities, intrinsic_matrix, image_height, image_width, background_color)
    # 출력 디렉터리 경로 객체를 만든다.
    output_dir = Path("outputs")
    # 출력 디렉터리와 필요한 상위 디렉터리를 생성한다.
    output_dir.mkdir(parents=True, exist_ok=True)
    # RGB 결과 파일의 전체 경로를 만든다.
    rgb_path = output_dir / "04_mini_splat_rgb.png"
    # alpha 결과 파일의 전체 경로를 만든다.
    alpha_path = output_dir / "04_mini_splat_alpha.png"
    # RGB 실수 배열을 8-bit PNG 이미지로 저장한다.
    save_rgb_image(rgb_path, rendered_rgb)
    # alpha 실수 배열을 8-bit grayscale PNG 이미지로 저장한다.
    save_gray_image(alpha_path, rendered_alpha)
    # 결과 RGB 배열의 shape를 출력해 H×W×3 구조를 확인한다.
    print(f"RGB shape: {rendered_rgb.shape}")
    # 결과 alpha 배열의 최솟값과 최댓값을 소수 넷째 자리까지 출력한다.
    print(f"alpha range: {rendered_alpha.min():.4f} ~ {rendered_alpha.max():.4f}")
    # RGB 결과 파일의 절대 경로를 출력한다.
    print(f"RGB 저장 완료: {rgb_path.resolve()}")
    # alpha 결과 파일의 절대 경로를 출력한다.
    print(f"alpha 저장 완료: {alpha_path.resolve()}")


# 이 파일을 직접 실행했을 때만 main 함수를 호출한다.
if __name__ == "__main__":
    # mini splat renderer 전체 예제를 시작한다.
    main()
