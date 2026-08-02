"""실제 모터 없이 tool orchestration과 안전 정책을 연습하는 mock입니다."""

from __future__ import annotations

# dataclass는 robot state와 log record의 구조를 명시합니다.
from dataclasses import dataclass, field
# monotonic은 시스템 시각 변경의 영향을 받지 않는 실행 시간을 기록합니다.
from time import monotonic
from typing import Any

from .safety import MoveCommand, SafetyEnvelope, ToolRejected


@dataclass
class RobotState:
    """Mock robot이 알고 있는 최소 상태입니다."""

    # 시작 pose는 안전한 home 위치라고 가정합니다.
    x: float = 0.0
    y: float = 0.0
    z: float = 0.25
    # True는 gripper가 열린 상태입니다.
    gripper_open: bool = True
    # stop 이후에는 operator reset 전까지 어떤 행동도 허용하지 않습니다.
    stopped: bool = False

    @property
    def xyz(self) -> tuple[float, float, float]:
        """현재 Cartesian 위치를 tuple로 반환합니다."""

        return self.x, self.y, self.z


@dataclass(frozen=True)
class ToolRecord:
    """재현과 감사를 위해 저장하는 tool 실행 기록입니다."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    elapsed_s: float


@dataclass
class MockRobot:
    """SafetyEnvelope를 통과한 명령만 상태에 반영하는 가상 로봇입니다."""

    # safety는 모델이나 agent와 별도 객체로 주입합니다.
    safety: SafetyEnvelope
    # 상태는 테스트마다 새 객체가 만들어지도록 default_factory를 사용합니다.
    state: RobotState = field(default_factory=RobotState)
    # call ID를 저장해 network retry의 중복 실행을 막습니다.
    processed_call_ids: set[str] = field(default_factory=set)
    # 실행 이력은 append-only 형태로 보관합니다.
    records: list[ToolRecord] = field(default_factory=list)

    def move(self, call_id: str, command: MoveCommand, *, human_present: bool = False) -> dict[str, Any]:
        """검증된 Cartesian move를 mock state에 적용합니다."""

        # 같은 call ID가 재전송되면 움직이지 않고 이전 처리 사실을 알립니다.
        if call_id in self.processed_call_ids:
            return {"status": "duplicate_ignored", "call_id": call_id}
        # STOPPED 상태는 모델이 자동으로 해제할 수 없습니다.
        if self.state.stopped:
            raise ToolRejected("robot is stopped; manual reset is required")
        # 실제 위치 변경 전에 독립 안전 정책을 실행합니다.
        self.safety.validate_move(self.state.xyz, command, human_present=human_present)
        # 이 mock에서는 물리 실행이 즉시 성공했다고 가정하고 상태를 갱신합니다.
        self.state.x = command.x
        self.state.y = command.y
        self.state.z = command.z
        # 성공한 call ID만 idempotency set에 넣습니다.
        self.processed_call_ids.add(call_id)
        # 실제 시스템이라면 encoder·pose sensor로 측정한 값을 넣어야 합니다.
        return {
            "status": "success",
            "observed_pose": {"x": self.state.x, "y": self.state.y, "z": self.state.z},
            "frame_id": command.frame_id,
        }

    def set_gripper(self, call_id: str, opened: bool, max_force_n: float) -> dict[str, Any]:
        """제한된 gripper 명령을 mock state에 적용합니다."""

        if call_id in self.processed_call_ids:
            return {"status": "duplicate_ignored", "call_id": call_id}
        if self.state.stopped:
            raise ToolRejected("robot is stopped; manual reset is required")
        # bool 이외의 truthy 문자열이 실수로 실행되는 것을 거부합니다.
        if not isinstance(opened, bool):
            raise ToolRejected("opened must be a boolean")
        # 교육용 gripper의 힘을 0보다 크고 20N 이하로 제한합니다.
        if isinstance(max_force_n, bool) or not isinstance(max_force_n, (int, float)):
            raise ToolRejected("max_force_n must be numeric")
        if not (0.0 < float(max_force_n) <= 20.0):
            raise ToolRejected("gripper force is outside the allowed range")
        # 안전 검증 이후에만 상태를 변경합니다.
        self.state.gripper_open = opened
        self.processed_call_ids.add(call_id)
        return {"status": "success", "gripper_open": self.state.gripper_open}

    def stop(self, reason: str) -> dict[str, Any]:
        """Robot을 latch된 STOPPED 상태로 전환합니다."""

        # stop은 여러 번 호출해도 같은 안전 상태를 유지하도록 idempotent합니다.
        self.state.stopped = True
        # 실제 driver라면 torque/PWM disable과 safety controller 확인이 필요합니다.
        return {"status": "stopped", "reason": reason}


@dataclass
class ToolExecutor:
    """모델이 제안한 tool name과 arguments를 엄격하게 dispatch합니다."""

    robot: MockRobot
    max_steps: int = 12
    deadline_s: float = 10.0

    def execute_plan(
        self,
        plan: list[dict[str, Any]],
        *,
        human_present: bool = False,
    ) -> list[dict[str, Any]]:
        """미리 준비한 tool-call plan을 안전하게 실행합니다."""

        # 빈 plan도 정상적으로 빈 결과를 반환합니다.
        if len(plan) > self.max_steps:
            raise ToolRejected(f"plan exceeds the {self.max_steps}-step budget")
        # monotonic deadline은 wall clock 변경과 무관합니다.
        started_at = monotonic()
        # model에 돌려줄 실제 tool result를 순서대로 저장합니다.
        results: list[dict[str, Any]] = []
        # 각 호출은 독립적으로 schema와 안전 정책을 통과해야 합니다.
        for index, call in enumerate(plan):
            # 전체 agent loop가 deadline을 넘으면 즉시 정지합니다.
            if monotonic() - started_at > self.deadline_s:
                self.robot.stop("plan deadline exceeded")
                raise ToolRejected("plan deadline exceeded")
            # 필수 envelope field를 명시적으로 읽습니다.
            call_id = call.get("id")
            name = call.get("name")
            arguments = call.get("arguments")
            # ID가 없으면 idempotency를 보장할 수 없습니다.
            if not isinstance(call_id, str) or not call_id:
                raise ToolRejected(f"step {index} has no valid call id")
            # tool 이름은 문자열이어야 allowlist와 정확히 비교할 수 있습니다.
            if not isinstance(name, str):
                raise ToolRejected(f"step {index} has no valid tool name")
            # argument는 free-form text가 아니라 object여야 합니다.
            if not isinstance(arguments, dict):
                raise ToolRejected(f"step {index} arguments must be an object")
            # 각 tool 실행 시간을 별도로 기록합니다.
            tool_started_at = monotonic()
            try:
                # allowlist에 있는 move만 명시적 keyword로 변환합니다.
                if name == "move":
                    command = self._move_command(arguments)
                    result = self.robot.move(call_id, command, human_present=human_present)
                # gripper tool도 알려진 두 인자만 전달합니다.
                elif name == "set_gripper":
                    self._require_exact_keys(arguments, {"opened", "max_force_n"})
                    result = self.robot.set_gripper(
                        call_id,
                        opened=arguments["opened"],
                        max_force_n=arguments["max_force_n"],
                    )
                # stop은 모델이 아닌 안전 계층도 호출할 수 있는 fail-safe tool입니다.
                elif name == "stop":
                    self._require_exact_keys(arguments, {"reason"})
                    result = self.robot.stop(reason=str(arguments["reason"]))
                # allowlist 밖 이름은 reflection이나 getattr로 실행하지 않습니다.
                else:
                    raise ToolRejected(f"tool {name!r} is not allowlisted")
            except Exception:
                # 어떤 validation/execution 실패도 mock을 STOPPED 상태로 latch합니다.
                self.robot.stop(f"tool {name!r} failed")
                # 원래 예외를 유지해 테스트와 운영자가 원인을 볼 수 있게 합니다.
                raise
            # 실행된 결과와 시간을 append-only log에 기록합니다.
            record = ToolRecord(
                call_id=call_id,
                name=name,
                arguments=dict(arguments),
                result=dict(result),
                elapsed_s=monotonic() - tool_started_at,
            )
            self.robot.records.append(record)
            # model loop에 되돌릴 실제 결과에도 call ID를 붙입니다.
            results.append({"call_id": call_id, "name": name, "result": result})
        # 모든 단계가 성공한 경우에만 전체 결과를 반환합니다.
        return results

    @staticmethod
    def _require_exact_keys(arguments: dict[str, Any], expected: set[str]) -> None:
        """누락·추가 인자를 모두 거부해 schema drift를 막습니다."""

        actual = set(arguments)
        if actual != expected:
            raise ToolRejected(
                f"arguments must have exactly {sorted(expected)}, got {sorted(actual)}"
            )

    def _move_command(self, arguments: dict[str, Any]) -> MoveCommand:
        """Move tool mapping을 명시적인 command 객체로 변환합니다."""

        # move가 받을 수 있는 필드를 고정합니다.
        expected = {"x", "y", "z", "speed_m_s", "frame_id"}
        self._require_exact_keys(arguments, expected)
        # float 변환은 문자열 숫자를 허용하지 않도록 safety에서 다시 type 검사합니다.
        return MoveCommand(
            x=arguments["x"],
            y=arguments["y"],
            z=arguments["z"],
            speed_m_s=arguments["speed_m_s"],
            frame_id=arguments["frame_id"],
        )

