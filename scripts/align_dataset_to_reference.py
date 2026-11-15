#!/usr/bin/env python3
"""把一份抽帧数据集按画面内容对齐到另一份更早的数据集，并回收其中的遥测字段。

适用场景：原始视频在剪辑后重新抽帧，但剪辑版丢失了 DJI 遥测字幕（重编码会去掉
mov_text 字幕流），于是新数据集的 meta 里没有 t/z。若之前那份数据集仍在，可以用
本脚本把新帧逐帧对回旧帧，再把旧 meta 的 t / z / GPS / 相机参数写回。

对齐方法：
  * 描述子 = 单帧 48x27 灰度归一化指纹，串联前后各 radius 帧（慢速航拍里相邻秒
    画面极像，单帧指纹分不开，串联后判别力明显提升）；
  * 逐帧取参考库最优匹配后，先做中值滤波去掉孤立误匹配，再取「只增不减」的前缀
    最大值——剪辑只删内容，参考帧序号不可能回退；
  * 写回时记录 aligned_from（来源帧文件名）与 aligned_distance（指纹距离，越小
    越可信），便于事后判断哪些帧的遥测是可靠的。

用法：

    # 只打印对齐报告（分段偏移、每段置信度）
    python3 scripts/align_dataset_to_reference.py video_frames video_frames_old

    # 指定分组配对（默认按同名分组配对）
    python3 scripts/align_dataset_to_reference.py video_frames video_frames_old \
        --group 20261113_1739=20261113_1739

    # 把遥测字段写回新数据集的 meta.jsonl
    python3 scripts/align_dataset_to_reference.py video_frames video_frames_old --write

注意：新旧视频的抽样网格未必落在同一秒上，恢复出的 t / z 精度约 ±1 秒。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

TELEMETRY_KEYS = ("t", "z", "lat", "lon", "sats", "dist", "hs", "vs",
                  "iso", "ss", "fnum", "zoom", "ev")
SIGNATURE_SIZE = (48, 27)


def signature(image: Image.Image) -> np.ndarray:
    small = image.convert("L").resize(SIGNATURE_SIZE, Image.BILINEAR)
    array = np.asarray(small, dtype=np.float32)
    return (array - array.mean()) / (array.std() + 1e-6)


def context(signatures: np.ndarray, radius: int) -> np.ndarray:
    """把每帧及前后各 radius 帧的指纹串联成上下文描述子（边缘重复端点）。"""
    count = len(signatures)
    parts = [signatures[np.clip(np.arange(count) + delta, 0, count - 1)]
             for delta in range(-radius, radius + 1)]
    return np.concatenate(parts, axis=1)


def load_group(root: Path, group: str) -> tuple[list[dict], np.ndarray]:
    directory = root / "images" / group
    meta_path = directory / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    signatures = []
    for row in rows:
        with Image.open(directory / row["file"]) as image:
            signatures.append(signature(image))
    return rows, np.stack(signatures)


def monotonic_indices(indices: np.ndarray, window: int = 5) -> np.ndarray:
    """中值滤波去掉孤立误匹配，再用前缀最大值保证参考序号单调不减。"""
    pad = window // 2
    padded = np.pad(indices, pad, mode="edge")
    filtered = np.array([int(np.median(padded[i:i + window])) for i in range(len(indices))])
    return np.maximum.accumulate(filtered)


def split_segments(indices: np.ndarray, gap: int = 3) -> list[tuple[int, int]]:
    """参考序号跳变超过 gap 视为剪辑点，返回 [(起始帧, 结束帧)]。"""
    segments, start = [], 0
    for i in range(1, len(indices) + 1):
        if i == len(indices) or indices[i] - indices[i - 1] > gap:
            segments.append((start, i - 1))
            start = i
    return segments


def align_group(new_root: Path, ref_root: Path, new_group: str, ref_group: str,
                radius: int, write: bool) -> None:
    new_rows, new_signatures = load_group(new_root, new_group)
    ref_rows, ref_signatures = load_group(ref_root, ref_group)
    ref_t = np.array([row.get("t_rel") or 0.0 for row in ref_rows])
    new_t = np.array([row.get("t_rel") or 0.0 for row in new_rows])

    distance = np.abs(
        context(new_signatures, radius)[:, None] - context(ref_signatures, radius)[None]
    ).mean(axis=(2, 3))
    raw = distance.argmin(axis=1)
    final = monotonic_indices(raw)
    final_distance = distance[np.arange(len(new_rows)), final]

    print(f"\n=== {new_group} ({len(new_rows)} 帧) 对齐到 {ref_group} ({len(ref_rows)} 帧) ===")
    print("逐帧最优偏移(前 5 种):",
          Counter((np.round(ref_t[raw] - new_t).astype(int)).tolist()).most_common(5))
    for start, end in split_segments(final):
        segment_distance = final_distance[start:end + 1]
        offset = float(np.median(ref_t[final[start:end + 1]] - new_t[start:end + 1]))
        print(f"  帧 {start:4d}-{end:4d} (新 t {new_t[start]:6.1f}-{new_t[end]:6.1f}s) "
              f"偏移 {offset:+7.1f}s -> 参考 t {new_t[start] + offset:6.1f}-{new_t[end] + offset:6.1f}s "
              f"距离中位 {np.median(segment_distance):.3f} P90 {np.percentile(segment_distance, 90):.3f}")
    print(f"整体距离: 中位 {np.median(final_distance):.4f} P90 {np.percentile(final_distance, 90):.4f} "
          f"最大 {final_distance.max():.4f}")

    if not write:
        return

    directory = new_root / "images" / new_group
    lines = []
    filled = 0
    for index, row in enumerate(new_rows):
        source = ref_rows[final[index]]
        for key in TELEMETRY_KEYS:
            if source.get(key) is not None:
                row[key] = source[key]
                filled += 1
        row["aligned_from"] = source.get("file")
        row["aligned_distance"] = round(float(final_distance[index]), 4)
        lines.append(json.dumps(row, ensure_ascii=False))
    (directory / "meta.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    altitudes = [row["z"] for row in new_rows if row.get("z") is not None]
    if altitudes:
        print(f"已写回 {len(lines)} 帧（{filled} 个字段），"
              f"高度 {min(altitudes):.1f}~{max(altitudes):.1f} m")
    else:
        print(f"已写回 {len(lines)} 帧（参考库没有高度字段）")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把抽帧数据集按画面内容对齐到更早的数据集并回收遥测字段",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="新数据集根目录（其下 images/<group>/）")
    parser.add_argument("reference", type=Path, help="参考数据集根目录")
    parser.add_argument("--group", action="append", default=None,
                        help="分组配对，形如 新组=参考组；可重复，默认按同名分组配对")
    parser.add_argument("--radius", type=int, default=1, help="上下文描述子的前后帧数")
    parser.add_argument("--write", action="store_true", help="把遥测字段写回新数据集的 meta.jsonl")
    args = parser.parse_args()

    new_root = args.dataset.expanduser().resolve()
    ref_root = args.reference.expanduser().resolve()
    if args.group:
        pairs = []
        for item in args.group:
            new_group, _, ref_group = item.partition("=")
            pairs.append((new_group, ref_group or new_group))
    else:
        new_groups = sorted(p.name for p in (new_root / "images").iterdir() if p.is_dir())
        pairs = [(name, name) for name in new_groups]

    if not pairs:
        print("错误: 没有可对齐的分组", file=sys.stderr)
        return 2
    for new_group, ref_group in pairs:
        align_group(new_root, ref_root, new_group, ref_group, args.radius, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
