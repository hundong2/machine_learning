"""생성형 tool 제안보다 safety gate가 항상 우선함을 검증합니다."""

import unittest

from gemini_robotics_learning.mock_robot import MockRobot, ToolExecutor
from gemini_robotics_learning.safety import (
    ForbiddenBox,
    SafetyEnvelope,
    ToolRejected,
    WorkspaceLimits,
)


def make_executor() -> ToolExecutor:
    """각 시험이 같은 독립 안전 설정을 사용하게 합니다."""

    workspace = WorkspaceLimits(-0.30, 0.30, -0.30, 0.30, 0.05, 0.40)
    forbidden = ForbiddenBox(
        name="camera-post",
        limits=WorkspaceLimits(0.10, 0.18, -0.05, 0.05, 0.05, 0.40),
    )
    safety = SafetyEnvelope(
        workspace=workspace,
        max_step_m=0.30,
        max_speed_m_s=0.10,
        forbidden=[forbidden],
    )
    return ToolExecutor(robot=MockRobot(safety=safety))


class SafetyTest(unittest.TestCase):
    def test_safe_move_is_executed(self) -> None:
        executor = make_executor()
        plan = [
            {
                "id": "call-1",
                "name": "move",
                "arguments": {
                    "x": 0.05,
                    "y": 0.05,
                    "z": 0.20,
                    "speed_m_s": 0.05,
                    "frame_id": "table",
                },
            }
        ]
        result = executor.execute_plan(plan)
        self.assertEqual(result[0]["result"]["status"], "success")

    def test_forbidden_region_stops_robot(self) -> None:
        executor = make_executor()
        plan = [
            {
                "id": "call-2",
                "name": "move",
                "arguments": {
                    "x": 0.14,
                    "y": 0.0,
                    "z": 0.20,
                    "speed_m_s": 0.05,
                    "frame_id": "table",
                },
            }
        ]
        with self.assertRaises(ToolRejected):
            executor.execute_plan(plan)
        self.assertTrue(executor.robot.state.stopped)

    def test_unknown_tool_is_rejected(self) -> None:
        executor = make_executor()
        with self.assertRaises(ToolRejected):
            executor.execute_plan([{"id": "x", "name": "run_shell", "arguments": {}}])

    def test_duplicate_call_is_not_executed_twice(self) -> None:
        executor = make_executor()
        call = {
            "id": "same-id",
            "name": "set_gripper",
            "arguments": {"opened": False, "max_force_n": 5.0},
        }
        first = executor.execute_plan([call])
        second = executor.execute_plan([call])
        self.assertEqual(first[0]["result"]["status"], "success")
        self.assertEqual(second[0]["result"]["status"], "duplicate_ignored")

    def test_human_presence_rejects_move(self) -> None:
        executor = make_executor()
        plan = [
            {
                "id": "human-test",
                "name": "move",
                "arguments": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.20,
                    "speed_m_s": 0.05,
                    "frame_id": "table",
                },
            }
        ]
        with self.assertRaises(ToolRejected):
            executor.execute_plan(plan, human_present=True)


if __name__ == "__main__":
    unittest.main()

