"""Gemini 텍스트 응답을 안전한 Python 데이터로 변환합니다."""

from __future__ import annotations

# JSONDecoder.raw_decode는 응답 앞뒤 설명을 실행하지 않고 JSON만 읽게 해 줍니다.
from json import JSONDecodeError, JSONDecoder
from dataclasses import dataclass

from .geometry import NormalizedBox, NormalizedPoint


def extract_json(text: str) -> object:
    """응답에서 첫 번째 JSON object 또는 array만 파싱합니다.

    `eval`이나 `ast.literal_eval` 대신 표준 JSON parser를 사용합니다. Markdown
    설명이 섞여 있어도 첫 JSON 시작점부터 정확한 한 값만 읽습니다.
    """

    # 빈 응답은 모델 실패로 간주합니다.
    if not text or not text.strip():
        raise ValueError("model response is empty")
    # object와 array 중 먼저 등장하는 위치를 후보로 모읍니다.
    starts = [index for token in ("[", "{") if (index := text.find(token)) >= 0]
    # JSON 시작 문자가 없으면 구조화 응답이 아닙니다.
    if not starts:
        raise ValueError("model response does not contain JSON")
    # 가장 앞의 JSON 시작 위치부터 하나의 값만 decode합니다.
    start = min(starts)
    try:
        value, _end = JSONDecoder().raw_decode(text[start:])
    except JSONDecodeError as error:
        raise ValueError(f"invalid JSON response: {error.msg}") from error
    # 파싱한 데이터는 아직 의미 검증 전이므로 그대로 반환합니다.
    return value


@dataclass(frozen=True)
class PointDetection:
    """검증된 pointing 결과 하나입니다."""

    point: NormalizedPoint
    label: str


def parse_point_detections(text: str, maximum_items: int = 10) -> list[PointDetection]:
    """`[{point:[y,x], label:string}]` 형식의 응답을 검증합니다."""

    # 먼저 문자열을 실행하지 않는 JSON 데이터로 바꿉니다.
    value = extract_json(text)
    # pointing 응답의 최상위는 배열이어야 합니다.
    if not isinstance(value, list):
        raise ValueError("point response must be a JSON array")
    # prompt가 제한한 개수보다 많으면 예상하지 않은 출력을 거부합니다.
    if len(value) > maximum_items:
        raise ValueError(f"point response has more than {maximum_items} items")
    # 검증된 결과를 새 배열에만 추가합니다.
    detections: list[PointDetection] = []
    for index, item in enumerate(value):
        # 각 원소는 point와 label을 가진 object여야 합니다.
        if not isinstance(item, dict):
            raise ValueError(f"item {index} must be an object")
        # point가 빠지면 위치를 사용할 수 없습니다.
        if "point" not in item:
            raise ValueError(f"item {index} is missing point")
        # label은 표시·matching에 필요하므로 문자열만 허용합니다.
        label = item.get("label", "")
        if not isinstance(label, str):
            raise ValueError(f"item {index} label must be a string")
        # 좌표 순서와 범위는 NormalizedPoint가 검증합니다.
        point = NormalizedPoint.from_sequence(item["point"])
        # 검증을 모두 통과한 항목만 결과에 넣습니다.
        detections.append(PointDetection(point=point, label=label.strip()))
    # 빈 배열은 “대상을 찾지 못함”이라는 정상 결과로 허용합니다.
    return detections


def parse_boxes(text: str, maximum_items: int = 20) -> list[NormalizedBox]:
    """공식 y/x/y2/x2 bounding-box 배열을 검증합니다."""

    # JSON parsing은 pointing과 같은 안전 경계를 사용합니다.
    value = extract_json(text)
    if not isinstance(value, list):
        raise ValueError("box response must be a JSON array")
    if len(value) > maximum_items:
        raise ValueError(f"box response has more than {maximum_items} items")
    # object가 아닌 항목을 조용히 버리면 모델 오류를 숨기므로 전부 거부합니다.
    if any(not isinstance(item, dict) for item in value):
        raise ValueError("every box item must be an object")
    # 각 mapping의 범위와 모서리 순서는 NormalizedBox가 검사합니다.
    return [NormalizedBox.from_mapping(item) for item in value]


@dataclass(frozen=True)
class ProgressReport:
    """ER 2의 다섯 단계 영상 진행도 결과입니다."""

    level: str

    @classmethod
    def from_text(cls, text: str) -> "ProgressReport":
        """`{progress_level: ...}` 응답을 허용된 enum으로 제한합니다."""

        value = extract_json(text)
        if not isinstance(value, dict):
            raise ValueError("progress response must be an object")
        level = value.get("progress_level")
        allowed = {"0-20", "20-40", "40-60", "60-80", "80-100"}
        if level not in allowed:
            raise ValueError(f"unsupported progress level: {level!r}")
        return cls(level=str(level))
