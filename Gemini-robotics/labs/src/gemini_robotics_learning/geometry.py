"""Gemini의 정규화 좌표를 이미지와 로봇 평면 좌표로 변환합니다.

이 모듈은 API를 호출하지 않습니다. 생성형 모델의 문자열 출력을 실제 행동과
분리해, 결정론적인 좌표 계산만 독립적으로 단위 테스트하기 위한 코드입니다.
"""

from __future__ import annotations

# dataclass는 좌표의 필드 이름을 명확히 하고 불변 객체를 만들기 위해 사용합니다.
from dataclasses import dataclass
# isfinite는 NaN과 무한대가 로봇 좌표로 흘러가는 것을 막습니다.
from math import isfinite
from typing import Sequence


# Gemini 공식 spatial guide가 사용하는 정규화 좌표의 최댓값입니다.
NORMALIZED_MAX = 1000.0


def _finite_number(value: object, name: str) -> float:
    """입력 하나가 유한한 실수인지 검사하고 float로 반환합니다."""

    # bool은 int의 하위 타입이지만 좌표 True/False는 의미가 없으므로 거부합니다.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    # 이후 수식을 단순하게 하기 위해 숫자를 float로 통일합니다.
    number = float(value)
    # NaN/Inf는 모든 범위 검사를 우회하거나 제어기를 망가뜨릴 수 있습니다.
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    # 검증을 통과한 값만 호출자에게 돌려줍니다.
    return number


@dataclass(frozen=True)
class PixelPoint:
    """이미지의 픽셀 좌표입니다. 일반 영상 관례대로 x가 먼저입니다."""

    # x는 왼쪽에서 오른쪽으로 증가합니다.
    x: float
    # y는 위에서 아래로 증가합니다.
    y: float


@dataclass(frozen=True)
class NormalizedPoint:
    """Gemini 응답의 [y, x], 0~1000 좌표를 나타냅니다."""

    # 공식 응답 순서가 y 먼저이므로 필드도 y를 먼저 둡니다.
    y: float
    # x는 두 번째 값입니다.
    x: float

    @classmethod
    def from_sequence(cls, values: Sequence[object]) -> "NormalizedPoint":
        """길이 2의 JSON 배열을 범위 검증된 좌표로 변환합니다."""

        # 문자열이나 길이가 다른 배열이 조용히 해석되지 않도록 먼저 검사합니다.
        if isinstance(values, (str, bytes)) or len(values) != 2:
            raise ValueError("point must contain exactly [y, x]")
        # 첫 번째 값을 y로 읽는 것이 핵심 계약입니다.
        y = _finite_number(values[0], "point.y")
        # 두 번째 값을 x로 읽습니다.
        x = _finite_number(values[1], "point.x")
        # 모델이 약속한 0~1000 범위를 벗어나면 clamp하지 않고 거부합니다.
        if not (0.0 <= y <= NORMALIZED_MAX and 0.0 <= x <= NORMALIZED_MAX):
            raise ValueError("point coordinates must be in [0, 1000]")
        # 검증된 불변 좌표 객체를 만듭니다.
        return cls(y=y, x=x)

    def to_pixel(self, width: int, height: int) -> PixelPoint:
        """정규화 좌표를 0-based 픽셀 좌표로 변환합니다."""

        # 너비와 높이가 1 이하면 유효한 2D 영상 좌표계를 만들 수 없습니다.
        if width <= 1 or height <= 1:
            raise ValueError("width and height must be greater than 1")
        # 1000이 마지막 픽셀 width-1에 대응하도록 x를 변환합니다.
        pixel_x = self.x / NORMALIZED_MAX * (width - 1)
        # y도 같은 방식으로 height-1 범위에 매핑합니다.
        pixel_y = self.y / NORMALIZED_MAX * (height - 1)
        # 시각화와 calibration에서 소수 픽셀을 보존하기 위해 반올림하지 않습니다.
        return PixelPoint(x=pixel_x, y=pixel_y)


@dataclass(frozen=True)
class NormalizedBox:
    """0~1000 범위의 [y_min, x_min, y_max, x_max] box입니다."""

    y_min: float
    x_min: float
    y_max: float
    x_max: float
    label: str = ""

    @classmethod
    def from_mapping(cls, item: dict[str, object]) -> "NormalizedBox":
        """공식 예제 형식인 y, x, y2, x2 mapping을 검증합니다."""

        # 누락 필드는 KeyError가 아니라 이해하기 쉬운 ValueError로 바꿉니다.
        required = {"y", "x", "y2", "x2"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"box is missing fields: {sorted(missing)}")
        # 각 숫자에 동일한 finite 검사를 적용합니다.
        y_min = _finite_number(item["y"], "box.y")
        x_min = _finite_number(item["x"], "box.x")
        y_max = _finite_number(item["y2"], "box.y2")
        x_max = _finite_number(item["x2"], "box.x2")
        # 모든 모서리가 정규화 범위 안에 있어야 합니다.
        values = (y_min, x_min, y_max, x_max)
        if any(value < 0.0 or value > NORMALIZED_MAX for value in values):
            raise ValueError("box coordinates must be in [0, 1000]")
        # 왼쪽 위와 오른쪽 아래 순서가 뒤집힌 box는 거부합니다.
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("box min coordinates must be smaller than max coordinates")
        # label은 실행 권한이 아니라 표시용 데이터로만 저장합니다.
        label = str(item.get("label", ""))
        return cls(y_min=y_min, x_min=x_min, y_max=y_max, x_max=x_max, label=label)

    def center(self) -> NormalizedPoint:
        """Box의 중심을 Gemini 순서인 [y, x] point로 반환합니다."""

        # 두 y 모서리의 평균으로 중심 y를 구합니다.
        center_y = (self.y_min + self.y_max) / 2.0
        # 두 x 모서리의 평균으로 중심 x를 구합니다.
        center_x = (self.x_min + self.x_max) / 2.0
        # 이미 범위가 검증된 값이므로 직접 객체를 만듭니다.
        return NormalizedPoint(y=center_y, x=center_x)


@dataclass(frozen=True)
class PlanarCalibration:
    """픽셀을 평면 robot 좌표로 옮기는 3x3 homography입니다."""

    # 행 우선 3x3 matrix를 immutable tuple로 저장해 실행 중 변경을 막습니다.
    matrix: tuple[tuple[float, float, float], ...]
    # 출력 좌표가 속한 frame 이름을 데이터와 함께 보존합니다.
    frame_id: str = "table"
    # 출력 거리 단위를 명시합니다.
    units: str = "meter"

    def __post_init__(self) -> None:
        """Matrix shape와 숫자를 생성 시점에 검증합니다."""

        # homography는 정확히 세 개의 행이어야 합니다.
        if len(self.matrix) != 3 or any(len(row) != 3 for row in self.matrix):
            raise ValueError("homography matrix must be 3x3")
        # 모든 원소가 유한한 실수인지 확인합니다.
        for row_index, row in enumerate(self.matrix):
            for column_index, value in enumerate(row):
                _finite_number(value, f"matrix[{row_index}][{column_index}]")

    def pixel_to_world(self, point: PixelPoint) -> tuple[float, float]:
        """픽셀 point에 projective transform을 적용합니다."""

        # 짧은 이름을 사용해 homography 수식과 코드를 대응시킵니다.
        h = self.matrix
        # 동차좌표의 첫 번째 분자를 계산합니다.
        world_x_h = h[0][0] * point.x + h[0][1] * point.y + h[0][2]
        # 동차좌표의 두 번째 분자를 계산합니다.
        world_y_h = h[1][0] * point.x + h[1][1] * point.y + h[1][2]
        # projective scale w를 계산합니다.
        scale = h[2][0] * point.x + h[2][1] * point.y + h[2][2]
        # scale이 0에 가까우면 점이 무한대로 가므로 행동에 사용할 수 없습니다.
        if abs(scale) < 1e-12:
            raise ValueError("homography mapped the point to infinity")
        # 동차좌표를 실제 평면 좌표로 나눕니다.
        world_x = world_x_h / scale
        world_y = world_y_h / scale
        # 결과도 finite인지 다시 확인해 안전 계층으로 넘깁니다.
        return _finite_number(world_x, "world.x"), _finite_number(world_y, "world.y")

    def normalized_to_world(
        self, point: NormalizedPoint, width: int, height: int
    ) -> tuple[float, float]:
        """Gemini point를 pixel을 거쳐 평면 robot 좌표로 변환합니다."""

        # 먼저 공식 정규화 좌표를 실제 이미지 픽셀로 바꿉니다.
        pixel = point.to_pixel(width=width, height=height)
        # calibration은 pixel 좌표에 적용합니다.
        return self.pixel_to_world(pixel)

