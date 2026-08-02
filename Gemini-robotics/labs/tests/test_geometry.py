"""좌표 계약이 바뀌거나 순서가 뒤집히는 회귀를 잡습니다."""

import unittest

from gemini_robotics_learning.geometry import (
    NormalizedBox,
    NormalizedPoint,
    PixelPoint,
    PlanarCalibration,
)
from gemini_robotics_learning.schemas import ProgressReport, parse_point_detections


class GeometryTest(unittest.TestCase):
    """API 없이 실행 가능한 결정론적 좌표 시험입니다."""

    def test_y_x_order_converts_to_pixel_x_y(self) -> None:
        # [y=250, x=750]을 넣어 축이 서로 다른 값으로 시험합니다.
        point = NormalizedPoint.from_sequence([250, 750])
        # 101×201 영상에서는 마지막 index가 각각 100과 200입니다.
        pixel = point.to_pixel(width=101, height=201)
        # x=750/1000×100이므로 75여야 합니다.
        self.assertAlmostEqual(pixel.x, 75.0)
        # y=250/1000×200이므로 50이어야 합니다.
        self.assertAlmostEqual(pixel.y, 50.0)

    def test_out_of_range_point_is_rejected(self) -> None:
        # 1001은 공식 좌표 범위를 벗어나므로 ValueError가 필요합니다.
        with self.assertRaises(ValueError):
            NormalizedPoint.from_sequence([500, 1001])

    def test_box_center(self) -> None:
        # 공식 y/x/y2/x2 형식으로 box를 만듭니다.
        box = NormalizedBox.from_mapping(
            {"label": "block", "y": 100, "x": 200, "y2": 300, "x2": 600}
        )
        # y 중심은 200, x 중심은 400입니다.
        self.assertEqual(box.center(), NormalizedPoint(y=200.0, x=400.0))

    def test_identity_homography(self) -> None:
        # 단위 matrix는 pixel 좌표를 같은 world 숫자로 돌려줍니다.
        calibration = PlanarCalibration(
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.assertEqual(calibration.pixel_to_world(PixelPoint(12.0, 34.0)), (12.0, 34.0))

    def test_point_response_parser(self) -> None:
        # Markdown 설명이 앞에 있어도 첫 JSON 배열만 안전하게 읽습니다.
        result = parse_point_detections('result: [{"point":[500,250],"label":"cup"}]')
        self.assertEqual(result[0].label, "cup")
        self.assertEqual(result[0].point, NormalizedPoint(y=500.0, x=250.0))

    def test_progress_enum(self) -> None:
        # 공식 다섯 bracket 중 하나만 허용합니다.
        report = ProgressReport.from_text('{"progress_level":"60-80"}')
        self.assertEqual(report.level, "60-80")


if __name__ == "__main__":
    # 파일을 직접 실행할 때도 unittest runner를 시작합니다.
    unittest.main()
