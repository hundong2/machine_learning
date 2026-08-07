"""YOLO26 Depth 추론 후 원시 배열과 세 종류의 시각화를 저장한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colorize_depth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="https://ultralytics.com/images/bus.jpg")
    parser.add_argument("--model", default="yolo26n-depth.pt")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="cpu", help="예: cpu, 0, cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--metric-max", type=float, default=20.0, help="metric 색상 맵의 최대 거리(m)")
    parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"))
    return parser.parse_args()


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"이미지 저장 실패: {path}")


def main() -> None:
    args = parse_args()
    if args.metric_max <= 0:
        raise SystemExit("--metric-max는 0보다 커야 합니다.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    result = model.predict(source=args.source, imgsz=args.imgsz, device=args.device)[0]

    if result.depth is None:
        raise RuntimeError("깊이 결과가 없습니다. '-depth.pt' 모델인지 확인하세요.")

    depth = result.depth.data.detach().cpu().numpy().astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise RuntimeError("유효한 양의 깊이값이 없습니다.")

    np.save(args.output_dir / "depth_raw.npy", depth)
    write_image(
        args.output_dir / "depth_colored.png",
        colorize_depth(depth, cmap="spectral", mode="disparity"),
    )
    write_image(
        args.output_dir / "depth_metric.png",
        colorize_depth(
            depth,
            vmin=0.0,
            vmax=args.metric_max,
            cmap="inferno",
            mode="metric",
        ),
    )
    result.save(filename=str(args.output_dir / "depth_overlay.png"))

    values = depth[valid]
    print(f"depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"valid pixels: {valid.mean():.2%}")
    print(
        "depth meters - "
        f"min={values.min():.3f}, median={np.median(values):.3f}, "
        f"mean={values.mean():.3f}, max={values.max():.3f}"
    )

    height, width = depth.shape
    x, y = args.point if args.point else (width // 2, height // 2)
    if not (0 <= x < width and 0 <= y < height):
        raise SystemExit(f"좌표 ({x}, {y})가 이미지 범위 0<=X<{width}, 0<=Y<{height} 밖입니다.")
    print(f"pixel ({x}, {y}): {depth[y, x]:.3f} m")
    print(f"saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
