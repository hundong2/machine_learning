"""모델과 독립적으로 로봇 tool 인자를 검사하는 안전 계층입니다."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist, isfinite


class ToolRejected(RuntimeError):
    """Tool 요청이 안전 정책을 통과하지 못했음을 나타냅니다."""


@dataclass(frozen=True)
class WorkspaceLimits:
    """Robot base frame의 직육면체 허용 작업 공간입니다."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, x: float, y: float, z: float) -> bool:
        """점이 모든 축의 닫힌 구간 안에 있는지 반환합니다."""

        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class ForbiddenBox:
    """허용 workspace 안에서도 진입하면 안 되는 구역입니다."""

    name: str
    limits: WorkspaceLimits


@dataclass(frozen=True)
class MoveCommand:
    """검증 전의 Cartesian move 제안입니다."""

    x: float
    y: float
    z: float
    speed_m_s: float
    frame_id: str = "table"


@dataclass
class SafetyEnvelope:
    """Workspace, 속도, 이동 거리, 사람 근접 정책을 검사합니다."""

    # 전체 허용 공간은 생성 시 반드시 제공해야 합니다.
    workspace: WorkspaceLimits
    # 한 번의 tool call이 이동할 수 있는 최대 거리입니다.
    max_step_m: float = 0.20
    # 학습용 mock이 허용할 최대 Cartesian 속도입니다.
    max_speed_m_s: float = 0.10
    # 좌표 frame이 섞이는 사고를 막기 위한 허용 frame입니다.
    allowed_frame: str = "table"
    # workspace 내부의 금지 영역 목록입니다.
    forbidden: list[ForbiddenBox] = field(default_factory=list)

    def validate_move(
        self,
        current_xyz: tuple[float, float, float],
        command: MoveCommand,
        *,
        human_present: bool = False,
    ) -> None:
        """Move가 안전 정책을 통과하지 못하면 ToolRejected를 발생시킵니다."""

        # 사람 감지는 모델 추정이 아니라 독립 sensor 입력이라고 가정합니다.
        if human_present:
            raise ToolRejected("human is present in the protected workspace")
        # 서로 다른 좌표 frame의 숫자를 더하는 실수를 막습니다.
        if command.frame_id != self.allowed_frame:
            raise ToolRejected(f"frame {command.frame_id!r} is not allowed")
        # NaN과 Inf는 범위 비교 결과를 오염시키므로 가장 먼저 거부합니다.
        numeric_values = (*current_xyz, command.x, command.y, command.z, command.speed_m_s)
        if not all(isinstance(value, (int, float)) and isfinite(value) for value in numeric_values):
            raise ToolRejected("move contains a non-finite numeric value")
        # 목표점은 전체 workspace 안에 있어야 합니다.
        if not self.workspace.contains(command.x, command.y, command.z):
            raise ToolRejected("target is outside the allowed workspace")
        # 속도는 양수이고 정책의 최대값 이하여야 합니다.
        if not (0.0 < command.speed_m_s <= self.max_speed_m_s):
            raise ToolRejected("speed is outside the allowed range")
        # 큰 점프를 여러 개의 재관찰 단계로 나누도록 한 번의 거리를 제한합니다.
        target = (command.x, command.y, command.z)
        if dist(current_xyz, target) > self.max_step_m:
            raise ToolRejected("move exceeds the maximum distance per step")
        # 허용 workspace 안의 카메라 기둥 같은 금지 영역도 검사합니다.
        for region in self.forbidden:
            if region.limits.contains(command.x, command.y, command.z):
                raise ToolRejected(f"target enters forbidden region: {region.name}")

