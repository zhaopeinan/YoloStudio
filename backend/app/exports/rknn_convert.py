"""Standalone ONNX → RKNN converter.

Can be imported in-process or executed by a dedicated Python that has
rknn-toolkit2 installed (for example a conda env).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_calibration_images(roots: list[Path], limit: int = 100) -> list[Path]:
    images: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
                if len(images) >= limit:
                    return images
    return images


def convert_onnx_to_rknn(
    onnx_path: Path,
    output_path: Path,
    dataset_txt: Path | None = None,
    *,
    target_platform: str = "rk3588",
    do_quantization: bool = True,
    verbose: bool = False,
) -> Path:
    try:
        from rknn.api import RKNN
    except Exception as exc:  # pragma: no cover - exercised on conversion hosts
        raise RuntimeError(
            f"rknn-toolkit2 is not available: {exc.__class__.__name__}: {exc}"
        ) from exc

    if not onnx_path.is_file():
        raise RuntimeError(f"ONNX model was not found: {onnx_path}")

    if do_quantization and (dataset_txt is None or not dataset_txt.is_file()):
        raise RuntimeError("INT8 quantization requires a calibration dataset.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=verbose)
    try:
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=target_platform,
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        ret = rknn.build(
            do_quantization=do_quantization,
            dataset=str(dataset_txt) if do_quantization else None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()

    if not output_path.is_file():
        raise RuntimeError(f"RKNN export did not create: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO ONNX to RKNN")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    convert_onnx_to_rknn(
        args.onnx,
        args.output,
        args.dataset,
        target_platform=args.target_platform,
        do_quantization=not args.no_quantization,
        verbose=args.verbose,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
