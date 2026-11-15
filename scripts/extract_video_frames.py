#!/usr/bin/env python3
"""按固定时间间隔从视频抽帧，生成可导入 YoloStudio 的待标注图片数据集。

特点：
  * 输出目录布局与 YoloStudio「项目与数据」导入器约定一致：

        <out-root>/images/<group>/<视频名>_f<源帧号>.jpg
        <out-root>/images/<group>/meta.jsonl

    导入后 group 名进入图片 platform 字段；meta.jsonl 中 t（epoch 秒）与
    z（相对起飞点高度，米）分别进入 timestamp / altitude 字段。
  * 输入视频若带 DJI 遥测字幕流（mov_text SRT），按帧时间对齐解析
    GPS / 高度 / 水平速度 / 垂直速度 / 相机参数写入 meta.jsonl。
  * 抽帧按源视频真实时间轴逐帧取样（ffmpeg select + showinfo），文件名带
    源帧号，可精确回溯到原视频。
  * 同一 group 目录已存在时默认拒绝覆盖，避免不同密度混抽。

用法示例：

    # 抽 1.MP4 的前 331 秒，每秒 1 帧，输出到 video_frames/
    python3 scripts/extract_video_frames.py raw_video/1.MP4 \
        --out-root video_frames --fps 1 --end 331

    # 只打印抽帧计划（探测视频与遥测，不写文件）
    python3 scripts/extract_video_frames.py raw_video/1.MP4 \
        --out-root video_frames --fps 1 --end 331 --dry-run

已抽取的数据记录（2026-11-15，每 1 秒 1 帧）：

    raw_video/1.MP4 -> video_frames/images/20261113_1739/   331 张（取 [0, 331)s）
    raw_video/2.MP4 -> video_frames/images/20261114_1033/   616 张（取 [0, 616)s）

    两段视频的尾部均为落地后的地面镜头，不取；其余全程为航拍画面。
    video_frames/ 是导入 YoloStudio 的位置数据，已在 .gitignore 中忽略。
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SHANGHAI = timezone(timedelta(hours=8))

PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
SRT_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
TELEMETRY_RE = re.compile(
    r"F/(?P<fnum>[0-9.]+),\s*SS\s*(?P<ss>[0-9.]+),\s*ISO\s*(?P<iso>\d+),\s*"
    r"EV\s*(?P<ev>-?[0-9.]+),\s*DZOOM\s*(?P<zoom>[0-9.]+),\s*"
    r"GPS\s*\((?P<lon>-?[0-9.]+),\s*(?P<lat>-?[0-9.]+),\s*(?P<sats>\d+)\)"
    r"(?:,\s*D\s*(?P<dist>[0-9.]+)m)?"
    r"(?:,\s*H\s*(?P<h>-?[0-9.]+)m)?"
    r"(?:,\s*H\.S\s*(?P<hs>-?[0-9.]+)m/s)?"
    r"(?:,\s*V\.S\s*(?P<vs>-?[0-9.]+)m/s)?"
)


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ffprobe_json(video: Path, *args: str) -> dict:
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", *args, str(video)])
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def probe_video(video: Path) -> dict:
    data = ffprobe_json(
        video,
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name",
        "-show_entries", "format=duration,size:format_tags=creation_time,location",
    )
    stream = data["streams"][0]
    fmt = data["format"]
    tags = fmt.get("tags", {})

    def parse_rate(value: str) -> float:
        num, _, den = value.partition("/")
        return float(num) / float(den) if den else float(num)

    rate = parse_rate(stream.get("avg_frame_rate") or stream["r_frame_rate"])
    duration = float(fmt.get("duration") or stream.get("duration") or 0.0)
    creation = tags.get("creation_time")
    creation_dt = None
    if creation:
        try:
            creation_dt = datetime.fromisoformat(creation.replace("Z", "+00:00"))
        except ValueError:
            creation_dt = None
    return {
        "path": video,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": rate,
        "duration": duration,
        "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "creation_dt": creation_dt,
        "location": tags.get("location"),
        "codec": stream.get("codec_name"),
    }


def extract_telemetry_rows(video: Path, tmp_dir: Path) -> list[tuple[float, dict]]:
    """抽取 DJI 遥测字幕（若有），返回按起始秒排序的 (秒, 遥测字典) 列表。"""
    srt_path = tmp_dir / "telemetry.srt"
    proc = run([
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
        "-map", "0:s:0", "-c:s", "srt", str(srt_path), "-y",
    ])
    if proc.returncode != 0 or not srt_path.exists() or srt_path.stat().st_size == 0:
        return []

    rows: list[tuple[float, dict]] = []
    text = srt_path.read_text(encoding="utf-8", errors="replace")
    for block in re.split(r"\n\s*\n", text):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        timecode = next((line for line in lines if "-->" in line), None)
        if timecode is None:
            continue
        match = SRT_TIME_RE.match(timecode.strip())
        if not match:
            continue
        hours, minutes, seconds, millis = (int(part) for part in match.groups())
        start = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        payload = " ".join(lines[lines.index(timecode) + 1:])
        telemetry = TELEMETRY_RE.search(payload)
        if not telemetry:
            continue
        parsed = {key: value for key, value in telemetry.groupdict().items() if value is not None}
        numeric = {}
        for key, value in parsed.items():
            numeric[key] = int(value) if key in {"iso", "sats"} else float(value)
        rows.append((start, numeric))
    rows.sort(key=lambda item: item[0])
    return rows


def telemetry_at(rows: list[tuple[float, dict]], t_rel: float) -> dict:
    if not rows:
        return {}
    starts = [row[0] for row in rows]
    index = bisect.bisect_right(starts, t_rel + 1e-6) - 1
    if index < 0:
        index = 0
    return rows[index][1]


def select_frames(video: Path, tmp_dir: Path, stem: str, gap: float, end: float | None,
                  quality: int) -> tuple[list[float], int]:
    """用 select+showinfo 按源时间轴抽帧，返回 (每个输出帧的源 pts 秒列表, 输出帧数)。

    输出文件为 <tmp>/<stem>_%06d.jpg（从 1 开始编号）。
    """
    select_expr = f"isnan(prev_selected_t)+gte(t-prev_selected_t,{gap:g})"
    pattern = str(tmp_dir / f"{stem}_%06d.jpg")
    cmd = [
        "ffmpeg", "-v", "info", "-nostdin", "-stats",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"select='{select_expr}',showinfo",
        "-fps_mode", "passthrough",
        "-q:v", str(quality),
    ]
    if end is not None:
        cmd += ["-t", f"{end:g}"]
    cmd += [pattern, "-y"]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败:\n{proc.stderr[-2000:]}")

    times: list[float] = []
    for line in proc.stderr.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = PTS_RE.search(line)
        if match is None:
            raise RuntimeError(f"showinfo 行解析失败: {line.strip()}")
        times.append(float(match.group(1)))

    written = sorted(tmp_dir.glob(f"{stem}_*.jpg"))
    if len(written) > len(times):
        raise RuntimeError(
            f"帧数不一致: ffmpeg 写出 {len(written)} 个文件, showinfo 记录 {len(times)} 帧"
        )
    # -t 截止处的尾帧可能先被 showinfo 记录但未写出：按实际文件数截断
    times = times[: len(written)]
    return times, len(written)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="从视频按固定时间间隔抽帧，生成可导入 YoloStudio 的待标注数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="输入视频路径")
    parser.add_argument("--out-root", type=Path, default=Path("video_frames"),
                        help="数据集根目录（其下自动创建 images/<group>/）")
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧频率（帧/秒）")
    parser.add_argument("--start", type=float, default=0.0, help="起始秒（输入侧定位，可能±数帧误差）")
    parser.add_argument("--end", type=float, default=None, help="结束秒（不含；默认整段）")
    parser.add_argument("--quality", type=int, default=2, help="JPEG 质量（ffmpeg -q:v，数值越小越清晰）")
    parser.add_argument("--group", default=None, help="图片分组名（默认按拍摄时间自动命名）")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已存在的分组目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写文件")
    args = parser.parse_args()

    video: Path = args.video.expanduser().resolve()
    if not video.exists():
        print(f"错误: 视频不存在: {video}", file=sys.stderr)
        return 2
    if args.fps <= 0:
        print("错误: --fps 必须为正数", file=sys.stderr)
        return 2

    info = probe_video(video)
    gap = 1.0 / args.fps
    start = max(0.0, args.start)
    end = args.end if args.end is not None else info["duration"]
    if start >= end:
        print(f"错误: 起始时间 {start}s 不小于结束时间 {end}s", file=sys.stderr)
        return 2

    created = info["creation_dt"].astimezone(SHANGHAI) if info["creation_dt"] else None
    group = args.group or (created.strftime("%Y%m%d_%H%M") if created else video.stem)

    tmp_root = Path(tempfile.mkdtemp(prefix="extract_frames_"))
    try:
        telemetry_rows = extract_telemetry_rows(video, tmp_root)
        out_dir = args.out_root.expanduser().resolve() / "images" / group

        print(f"视频      : {video.name}  {info['width']}x{info['height']} "
              f"{info['fps']:.3f}fps  时长 {info['duration']:.1f}s")
        print(f"取帧区间  : [{start:g}, {end:g}) s  每 {gap:g}s 一帧")
        print(f"分组/目录 : {group} -> {out_dir}")
        print(f"遥测字幕  : {len(telemetry_rows)} 条" if telemetry_rows else "遥测字幕  : 无（meta 只写时间）")
        if created:
            print(f"拍摄时间  : {created:%Y-%m-%d %H:%M:%S} (UTC+8)")
        if info["location"]:
            print(f"拍摄位置  : {info['location']}")
        expected = int((end - start) * args.fps) + 1
        print(f"预计帧数  : 约 {expected} 张")

        if args.dry_run:
            print("dry-run：未写任何文件")
            return 0

        if out_dir.exists() and any(out_dir.iterdir()):
            if not args.overwrite:
                print(f"错误: 分组目录已存在且非空: {out_dir}\n     如需重抽请加 --overwrite 或换 --group",
                      file=sys.stderr)
                return 3

        pts_list, frame_count = select_frames(video, tmp_root, video.stem, gap, end, args.quality)
        if frame_count == 0:
            print("错误: 没有抽到任何帧", file=sys.stderr)
            return 1

        if out_dir.exists() and args.overwrite:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_lines: list[str] = []
        min_alt: float | None = None
        max_alt: float | None = None
        missing_telemetry = 0
        for index, pts in enumerate(pts_list):
            t_rel = start + pts
            frame_index = round(t_rel * info["fps"])
            new_name = f"{video.stem}_f{frame_index:06d}.jpg"
            shutil.move(str(tmp_root / f"{video.stem}_{index + 1:06d}.jpg"), str(out_dir / new_name))

            telemetry = telemetry_at(telemetry_rows, t_rel)
            if not telemetry:
                missing_telemetry += 1
            row: dict = {
                "file": new_name,
                "frame": frame_index,
                "t": round((info["creation_dt"].timestamp() + t_rel), 3) if info["creation_dt"] else None,
                "t_rel": round(t_rel, 3),
            }
            if telemetry:
                if "h" in telemetry:
                    row["z"] = telemetry["h"]
                    min_alt = telemetry["h"] if min_alt is None else min(min_alt, telemetry["h"])
                    max_alt = telemetry["h"] if max_alt is None else max(max_alt, telemetry["h"])
                for key in ("lat", "lon", "sats", "dist", "hs", "vs", "iso", "ss", "fnum", "zoom", "ev"):
                    if key in telemetry:
                        row[key] = telemetry[key]
            meta_lines.append(json.dumps(row, ensure_ascii=False))

        (out_dir / "meta.jsonl").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

        total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.jpg"))
        print(f"完成      : 抽出 {frame_count} 张 -> {out_dir}")
        print(f"体积      : {total_bytes / 1024 / 1024:.1f} MB（平均 {total_bytes / frame_count / 1024:.0f} KB/张）")
        if min_alt is not None:
            print(f"高度范围  : {min_alt:.1f} ~ {max_alt:.1f} m")
        if missing_telemetry:
            print(f"提示      : {missing_telemetry} 帧没有匹配到遥测行（meta 中 z/t 缺省）")
        return 0
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
