"""ER 2가 제안할 법한 pick-and-place plan을 mock에서 안전하게 실행합니다.

실행:
    $env:PYTHONPATH = "src"
    python examples/03_safe_mock_orchestrator.py
    python examples/03_safe_mock_orchestrator.py --unsafe-demo
"""

# argparse는 정상 plan과 의도적인 위험 plan을 선택합니다.
import argparse
# json은 실제 tool result와 audit log를 읽기 쉽게 출력합니다.
import json

from gemini_robotics_learning.mock_robot import MockRobot, ToolExecutor
from gemini_robotics_learning.safety import (
    ForbiddenBox,
    SafetyEnvelope,
    ToolRejected,
    WorkspaceLimits,
)


def move(call_id: str, x: float, y: float, z: float) -> dict:
    """반복되는 move tool envelope를 만드는 교육용 helper입니다."""

    # 실제 model function call과 같은 id/name/arguments 구조를 반환합니다.
    return {
        "id": call_id,
        "name": "move",
        "arguments": {
            "x": x,
            "y": y,
            "z": z,
            "speed_m_s": 0.05,
            "frame_id": "table",
        },
    }


def normal_plan() -> list[dict]:
    """Hover → descend → grasp → lift → place 순서의 안전한 sample입니다."""

    return [
        move("01-hover-block", 0.05, 0.08, 0.25),
        move("02-descend-block", 0.05, 0.08, 0.08),
        {"id": "03-grasp", "name": "set_gripper", "arguments": {"opened": False, "max_force_n": 5.0}},
        move("04-lift-block", 0.05, 0.08, 0.25),
        move("05-hover-bowl", -0.10, 0.10, 0.25),
        move("06-descend-bowl", -0.10, 0.10, 0.08),
        {"id": "07-release", "name": "set_gripper", "arguments": {"opened": True, "max_force_n": 5.0}},
        move("08-retreat", -0.10, 0.10, 0.25),
    ]


def build_executor() -> ToolExecutor:
    """Mock와 실제 adapter가 공유할 안전 envelope를 만듭니다."""

    # 전체 table workspace를 meter 단위로 제한합니다.
    workspace = WorkspaceLimits(-0.30, 0.30, -0.30, 0.30, 0.05, 0.40)
    # 카메라 기둥이 있는 위치는 workspace 안이어도 금지합니다.
    camera_post = ForbiddenBox(
        name="camera-post",
        limits=WorkspaceLimits(0.10, 0.18, -0.05, 0.05, 0.05, 0.40),
    )
    # 속도와 한 단계 이동 거리를 별도로 제한합니다.
    safety = SafetyEnvelope(
        workspace=workspace,
        max_step_m=0.30,
        max_speed_m_s=0.10,
        forbidden=[camera_post],
    )
    # Model과 무관한 mock robot을 만든 뒤 step/deadline budget을 적용합니다.
    return ToolExecutor(robot=MockRobot(safety=safety), max_steps=12, deadline_s=5.0)


def main() -> None:
    """선택한 plan을 실행하고 안전 결과를 출력합니다."""

    parser = argparse.ArgumentParser(description="Safe mock tool orchestrator")
    parser.add_argument("--unsafe-demo", action="store_true", help="금지 구역 요청을 의도적으로 시험")
    args = parser.parse_args()
    executor = build_executor()
    # unsafe mode는 카메라 기둥 내부로 이동하는 한 단계 plan입니다.
    plan = [move("unsafe-camera-post", 0.14, 0.0, 0.20)] if args.unsafe_demo else normal_plan()
    try:
        # 이 예제의 human_present는 독립 sensor 결과라고 가정합니다.
        results = executor.execute_plan(plan, human_present=False)
    except ToolRejected as error:
        # 거부는 crash가 아니라 기대한 안전 결과로 설명합니다.
        print(f"PLAN REJECTED: {error}")
        print(f"Robot stopped: {executor.robot.state.stopped}")
        return
    # 실제 tool result만 model에 반환해야 하므로 JSON 형태를 보여 줍니다.
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Final pose: {executor.robot.state.xyz}")
    print(f"Gripper open: {executor.robot.state.gripper_open}")


if __name__ == "__main__":
    main()

