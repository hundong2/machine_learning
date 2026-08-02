"""ER 2 streaming agent의 concurrency 경계를 API 없이 시뮬레이션합니다.

공식 Live API 연결 코드는 preview SDK와 함께 바뀔 수 있습니다. 이 파일은 오래된
frame을 쌓지 않는 latest-frame slot, heartbeat, blocking tool 경계를 학습합니다.
"""

import asyncio
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class Frame:
    """카메라 frame의 내용 대신 sequence와 freshness만 모의합니다."""

    sequence: int
    captured_at: float
    jpeg: bytes


class LatestFrameSlot:
    """소비가 느릴 때 과거 frame 대신 최신 frame 하나만 보존합니다."""

    def __init__(self) -> None:
        # None은 아직 카메라 frame이 없음을 뜻합니다.
        self._latest: Frame | None = None
        # Event는 새 frame이 들어왔음을 consumer에게 알립니다.
        self._updated = asyncio.Event()

    def put(self, frame: Frame) -> None:
        """이전 미처리 frame을 최신 frame으로 교체합니다."""

        # Queue append가 아니라 대입이므로 메모리가 무한히 증가하지 않습니다.
        self._latest = frame
        # 기다리는 heartbeat를 깨웁니다.
        self._updated.set()

    async def get_latest(self) -> Frame:
        """최소 한 frame을 기다린 뒤 현재 최신 값을 반환합니다."""

        await self._updated.wait()
        # Event를 지워 다음 update를 구분합니다.
        self._updated.clear()
        # Event가 set됐다면 latest는 존재하지만 type checker를 위해 확인합니다.
        if self._latest is None:
            raise RuntimeError("frame event set without a frame")
        return self._latest


async def fake_camera(slot: LatestFrameSlot, stop: asyncio.Event) -> None:
    """10 FPS 카메라를 모의하고 slot에는 최신 값만 넣습니다."""

    sequence = 0
    while not stop.is_set():
        slot.put(Frame(sequence=sequence, captured_at=monotonic(), jpeg=b"fake-jpeg"))
        sequence += 1
        await asyncio.sleep(0.1)


async def heartbeat(slot: LatestFrameSlot, stop: asyncio.Event) -> None:
    """1 FPS 제한의 semantic heartbeat를 모의합니다."""

    while not stop.is_set():
        frame = await slot.get_latest()
        age_ms = (monotonic() - frame.captured_at) * 1000.0
        # Production에서는 age가 threshold를 넘으면 API로 보내지 않습니다.
        if age_ms > 500.0:
            print(f"DROP stale frame {frame.sequence}: {age_ms:.1f} ms")
        else:
            # 실제 adapter는 JPEG와 짧은 판단 prompt를 Live API에 전송합니다.
            print(f"HEARTBEAT frame={frame.sequence} age_ms={age_ms:.1f} action=ack")
        # 공식 endpoint의 이미지 입력 제한에 맞춰 약 1초 간격을 둡니다.
        await asyncio.sleep(1.0)


async def blocking_tool(name: str, duration_s: float) -> dict[str, str]:
    """물리 행동이 끝날 때까지 다음 tool을 선택하지 않는 경계를 모의합니다."""

    # Tool 이름은 실제 구현에서 allowlist를 통과한 값이어야 합니다.
    print(f"TOOL START {name}")
    # 비동기 sleep은 receive/안전 task가 계속 실행되도록 합니다.
    await asyncio.sleep(duration_s)
    print(f"TOOL END   {name}")
    # 실제 구현은 sensor로 확인한 status를 반환해야 합니다.
    return {"status": "success"}


async def main() -> None:
    """카메라·heartbeat·tool task를 3초간 함께 실행합니다."""

    slot = LatestFrameSlot()
    stop = asyncio.Event()
    camera_task = asyncio.create_task(fake_camera(slot, stop))
    heartbeat_task = asyncio.create_task(heartbeat(slot, stop))
    # Tool 실행 중에도 카메라와 heartbeat task가 살아 있음을 확인합니다.
    await blocking_tool("navigate_to_named_waypoint", duration_s=2.2)
    stop.set()
    # 대기 중인 task를 명시적으로 취소해 process 종료를 결정적으로 만듭니다.
    camera_task.cancel()
    heartbeat_task.cancel()
    # 취소 예외는 종료 과정의 정상 신호이므로 모아 처리합니다.
    await asyncio.gather(camera_task, heartbeat_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())

