"""YOLO26 Depth 실습 환경을 점검한다."""

from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("[실패] torch가 없습니다. pip install -r requirements.txt를 실행하세요.") from exc

    try:
        import ultralytics
    except ImportError as exc:
        raise SystemExit("[실패] ultralytics가 없습니다. pip install -r requirements.txt를 실행하세요.") from exc

    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        print("권장 인자: --device 0")
    else:
        print("권장 인자: --device cpu (GPU보다 느릴 수 있습니다.)")

    version = tuple(int(part) for part in ultralytics.__version__.split(".")[:3])
    if version < (8, 4, 115):
        print("[주의] YOLO26 Depth 지원을 위해 ultralytics>=8.4.115로 업그레이드하세요.")
    else:
        print("[확인] YOLO26 Depth 실습 최소 버전을 충족합니다.")


if __name__ == "__main__":
    main()
