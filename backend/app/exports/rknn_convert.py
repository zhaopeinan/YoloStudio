"""Standalone ONNX → RKNN converter.

Can be imported in-process or executed by a dedicated Python that has
rknn-toolkit2 installed (for example a conda env).

YOLOv8 packs pixel-scale box coords and [0,1] class scores into one
`output0` tensor. Full INT8 per-tensor quantization then rounds every
score below ~1.0 to zero. Default mode keeps that output in float16 via
hybrid quantization (falls back to full FP16 if hybrid is unavailable).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
QUANT_MODES = {"hybrid", "fp16", "int8"}


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


def _quant_parameter_names(cfg_text: str) -> set[str]:
    names: set[str] = set()
    in_params = False
    for line in cfg_text.splitlines():
        if re.match(r"^quantize_parameters:\s*$", line):
            in_params = True
            continue
        if not in_params:
            continue
        if re.match(r"^[A-Za-z_]", line) and not line.startswith(" "):
            break
        match = re.match(r"^    ([^:]+):\s*$", line)
        if match:
            names.add(match.group(1).strip())
    return names


def _patch_quantization_cfg(cfg_path: Path) -> list[str]:
    """Force YOLO detection outputs to float16 in hybrid quantization cfg.

    Only names that already exist under quantize_parameters are written;
    inventing layer names makes hybrid_quantization_step2 fail.
    """
    text = cfg_path.read_text(encoding="utf-8")
    available = _quant_parameter_names(text)
    preferred = [
        "output0",
        "output0-rs",
        "output0_int8",
        "/model.22/Concat_output_0",
        "/model.22/Concat_1_output_0-rs",
        "/model.22/Concat_1_output_0_sw",
        "/model.22/Sigmoid_output_0_rs",
        "/model.22/Sigmoid_output_0",
    ]
    patched_layers = [name for name in preferred if name in available]
    if not patched_layers:
        patched_layers = sorted(
            name
            for name in available
            if "output0" in name or "Sigmoid" in name or name.endswith("/Concat_output_0")
        )
    if not patched_layers:
        raise RuntimeError(
            "Could not find YOLO output layers to keep in float16 for hybrid quantization"
        )

    custom_block_lines = ["custom_quantize_layers:"]
    for name in patched_layers:
        custom_block_lines.append(f"  {name}: float16")
    custom_block = "\n".join(custom_block_lines) + "\n"

    # Drop any previous custom_quantize_layers block (including `{}`).
    text = re.sub(
        r"^custom_quantize_layers:(?:\s*\{\})?(?:\n(?:[ \t].*)?)*\n?",
        "",
        text,
        count=1,
        flags=re.M,
    )
    text = custom_block + text.lstrip("\n")
    cfg_path.write_text(text, encoding="utf-8")
    return patched_layers


def _build_fp16(rknn, onnx_path: Path, output_path: Path, *, target_platform: str) -> None:
    ret = rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
        optimization_level=3,
        float_dtype="float16",
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"rknn.build(fp16) failed with code {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")


def _build_int8(
    rknn,
    onnx_path: Path,
    output_path: Path,
    dataset_txt: Path,
    *,
    target_platform: str,
) -> None:
    ret = rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
        optimization_level=3,
        quantized_algorithm="normal",
        quantized_method="channel",
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    ret = rknn.build(do_quantization=True, dataset=str(dataset_txt))
    if ret != 0:
        raise RuntimeError(f"rknn.build(int8) failed with code {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")


def _build_hybrid(
    rknn,
    onnx_path: Path,
    output_path: Path,
    dataset_txt: Path,
    work_dir: Path,
    *,
    target_platform: str,
) -> list[str]:
    ret = rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=target_platform,
        optimization_level=3,
        quantized_algorithm="normal",
        quantized_method="channel",
        float_dtype="float16",
    )
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code {ret}")

    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

    # hybrid_quantization_step1 writes files into the current working directory.
    ret = rknn.hybrid_quantization_step1(dataset=str(dataset_txt))
    if ret != 0:
        raise RuntimeError(f"rknn.hybrid_quantization_step1 failed with code {ret}")

    model_stem = onnx_path.stem
    candidates = {
        "model": work_dir / f"{model_stem}.model",
        "data": work_dir / f"{model_stem}.data",
        "cfg": work_dir / f"{model_stem}.quantization.cfg",
    }
    # Toolkit may also emit generic names; accept first existing set.
    if not candidates["cfg"].is_file():
        cfgs = sorted(work_dir.glob("*.quantization.cfg"))
        if not cfgs:
            raise RuntimeError("hybrid step1 did not produce a .quantization.cfg")
        stem = cfgs[0].name.replace(".quantization.cfg", "")
        candidates = {
            "model": work_dir / f"{stem}.model",
            "data": work_dir / f"{stem}.data",
            "cfg": cfgs[0],
        }

    for key, path in candidates.items():
        if not path.is_file():
            raise RuntimeError(f"hybrid step1 missing {key} file: {path}")

    patched = _patch_quantization_cfg(candidates["cfg"])

    ret = rknn.hybrid_quantization_step2(
        model_input=str(candidates["model"]),
        data_input=str(candidates["data"]),
        model_quantization_cfg=str(candidates["cfg"]),
    )
    if ret != 0:
        raise RuntimeError(f"rknn.hybrid_quantization_step2 failed with code {ret}")

    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    return patched


def convert_onnx_to_rknn(
    onnx_path: Path,
    output_path: Path,
    dataset_txt: Path | None = None,
    *,
    target_platform: str = "rk3588",
    quant_mode: str = "fp16",
    do_quantization: bool | None = None,
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

    # Back-compat for older callers that only knew do_quantization.
    if do_quantization is False:
        quant_mode = "fp16"
    elif do_quantization is True:
        quant_mode = "int8"

    quant_mode = (quant_mode or "fp16").lower()
    if quant_mode not in QUANT_MODES:
        raise RuntimeError(f"Unsupported quant mode: {quant_mode}")

    if quant_mode in {"hybrid", "int8"} and (dataset_txt is None or not dataset_txt.is_file()):
        raise RuntimeError(f"{quant_mode} quantization requires a calibration dataset.txt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent
    previous_cwd = Path.cwd()

    rknn = RKNN(verbose=verbose)
    try:
        # Hybrid/step1 artifacts are written relative to CWD.
        import os

        os.chdir(work_dir)

        if quant_mode == "fp16":
            _build_fp16(rknn, onnx_path, output_path, target_platform=target_platform)
        elif quant_mode == "int8":
            assert dataset_txt is not None
            _build_int8(
                rknn,
                onnx_path,
                output_path,
                dataset_txt,
                target_platform=target_platform,
            )
        else:
            assert dataset_txt is not None
            try:
                _build_hybrid(
                    rknn,
                    onnx_path,
                    output_path,
                    dataset_txt,
                    work_dir,
                    target_platform=target_platform,
                )
            except Exception as hybrid_exc:
                # FP16 is the known-good fallback when hybrid tooling/layout changes.
                print(
                    f"hybrid quantization failed ({hybrid_exc}); falling back to fp16",
                    file=sys.stderr,
                )
                rknn.release()
                rknn = RKNN(verbose=verbose)
                _build_fp16(rknn, onnx_path, output_path, target_platform=target_platform)
    finally:
        try:
            rknn.release()
        except Exception:
            pass
        try:
            import os

            os.chdir(previous_cwd)
        except Exception:
            pass

    if not output_path.is_file():
        raise RuntimeError(f"RKNN export did not create: {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert YOLO ONNX to RKNN")
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument(
        "--quant-mode",
        choices=sorted(QUANT_MODES),
        default="fp16",
        help="fp16=full float16 (default, safe for YOLOv8); hybrid=INT8 body + FP16 outputs; int8=full INT8",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Deprecated alias for --quant-mode fp16",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    quant_mode = "fp16" if args.no_quantization else args.quant_mode
    convert_onnx_to_rknn(
        args.onnx,
        args.output,
        args.dataset,
        target_platform=args.target_platform,
        quant_mode=quant_mode,
        verbose=args.verbose,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
