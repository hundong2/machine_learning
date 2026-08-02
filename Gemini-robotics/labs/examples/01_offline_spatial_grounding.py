"""API 없이 Gemini 좌표 계약과 평면 calibration을 연습합니다.

실행:
    $env:PYTHONPATH = "src"
    python examples/01_offline_spatial_grounding.py
"""

# 실제 Gemini 응답처럼 Markdown 설명이 섞인 sample 문자열을 준비합니다.
SAMPLE_RESPONSE = """
탐지 결과입니다.
[{"point": [375, 625], "label": "blue block"}]
"""

# 학습용 package에서 안전한 parser와 calibration 클래스를 가져옵니다.
from gemini_robotics_learning.geometry import PlanarCalibration
from gemini_robotics_learning.schemas import parse_point_detections


def main() -> None:
    """정규화 좌표 → pixel → table 좌표의 전체 흐름을 출력합니다."""

    # parser는 JSON을 실행하지 않고 좌표 순서·범위를 검증합니다.
    detections = parse_point_detections(SAMPLE_RESPONSE)
    # 이 예제는 대상 하나를 요청했으므로 정확히 하나인지 확인합니다.
    if len(detections) != 1:
        raise RuntimeError(f"expected one object, got {len(detections)}")
    # 첫 번째 검증 결과를 명확한 이름으로 꺼냅니다.
    detection = detections[0]
    # sample camera의 해상도를 명시합니다.
    image_width = 1280
    image_height = 720
    # [y,x] 정규화 좌표를 pixel (x,y)로 변환합니다.
    pixel = detection.point.to_pixel(width=image_width, height=image_height)

    # 예시 homography는 pixel 하나가 테이블 0.5mm에 해당한다고 가정합니다.
    # 실제 프로젝트에서는 ChArUco/calibration target으로 이 matrix를 구해야 합니다.
    calibration = PlanarCalibration(
        matrix=(
            (0.0005, 0.0, -0.32),
            (0.0, 0.0005, -0.18),
            (0.0, 0.0, 1.0),
        ),
        frame_id="table",
        units="meter",
    )
    # pixel을 table frame의 meter 좌표로 옮깁니다.
    world_x, world_y = calibration.pixel_to_world(pixel)

    # 각 경계의 값과 단위를 출력해 축 뒤집힘을 눈으로 확인합니다.
    print(f"label             : {detection.label}")
    print(f"normalized [y, x] : [{detection.point.y:.1f}, {detection.point.x:.1f}]")
    print(f"pixel (x, y)      : ({pixel.x:.1f}, {pixel.y:.1f})")
    print(f"{calibration.frame_id} (x, y) m : ({world_x:.4f}, {world_y:.4f})")
    # 이 출력은 행동 제안일 뿐 실제 모터 명령이 아님을 명시합니다.
    print("ACTION DISABLED: run safety validation and operator confirmation first.")


# 다른 module에서 import할 때 자동 실행되지 않도록 entry point를 분리합니다.
if __name__ == "__main__":
    main()

