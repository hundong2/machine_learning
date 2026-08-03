"""Ultralytics 깊이 데이터셋의 이미지-깊이 짝과 배열 품질을 검사한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def expected_depth_path(image_path: Path, root: Path) -> Path:
    relative = image_path.relative_to(root / "images")
    return (root / "depth" / relative).with_suffix(".npy")


def check_split(root: Path, split: str) -> tuple[int, int]:
    image_dir = root / "images" / split
    depth_dir = root / "depth" / split
    if not image_dir.is_dir() or not depth_dir.is_dir():
        print(f"[오류] {split}: {image_dir} 또는 {depth_dir}가 없습니다.")
        return 0, 1

    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    expected = {expected_depth_path(path, root).resolve() for path in images}
    errors = 0

    for image_path in images:
        depth_path = expected_depth_path(image_path, root)
        if not depth_path.is_file():
            print(f"[오류] 깊이 파일 없음: {depth_path}")
            errors += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[오류] 이미지 읽기 실패: {image_path}")
            errors += 1
            continue

        try:
            depth = np.load(depth_path, allow_pickle=False)
        except Exception as exc:
            print(f"[오류] 깊이 읽기 실패: {depth_path} ({exc})")
            errors += 1
            continue

        if depth.ndim != 2 or depth.shape != image.shape[:2]:
            print(
                f"[오류] 크기 불일치: {image_path.name} image={image.shape[:2]}, "
                f"depth={depth.shape}"
            )
            errors += 1
        if depth.dtype != np.float32:
            print(f"[주의] {depth_path.name}: dtype={depth.dtype}, 권장=float32")
        finite_positive = np.isfinite(depth) & (depth > 0)
        if not finite_positive.any():
            print(f"[오류] 유효한 양의 깊이 없음: {depth_path}")
            errors += 1
        elif finite_positive.mean() < 0.1:
            print(f"[주의] {depth_path.name}: 유효값 비율 {finite_positive.mean():.1%}")

    depth_files = {path.resolve() for path in depth_dir.rglob("*.npy")}
    for orphan in sorted(depth_files - expected):
        print(f"[주의] 대응 이미지가 없는 깊이 파일: {orphan}")

    print(f"[{split}] 이미지 {len(images)}개, 오류 {errors}개")
    return len(images), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="images/와 depth/를 포함한 루트")
    args = parser.parse_args()
    root = args.root.resolve()

    total_images = 0
    total_errors = 0
    for split in ("train", "val"):
        images, errors = check_split(root, split)
        total_images += images
        total_errors += errors

    print(f"총 이미지 {total_images}개, 오류 {total_errors}개")
    if total_errors:
        raise SystemExit(1)
    print("[통과] 기본 데이터셋 구조와 이미지-깊이 짝이 올바릅니다.")


if __name__ == "__main__":
    main()
