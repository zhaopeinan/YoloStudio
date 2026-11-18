from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.core.settings import Settings


PROGRESS_RE = re.compile(
    r"(?P<bytes>[\d,]+)\s+(?P<percent>\d+)%\s+(?P<speed>[\d.]+)(?P<unit>[kKmMgGtT]?B)/s"
)
ACTIVE_SYNC_PROCESSES: dict[int, subprocess.Popen[str]] = {}


@dataclass
class RsyncProgress:
    bytes_done: int
    percent: float | None
    speed_bps: float


def parse_rsync_progress_line(line: str) -> RsyncProgress | None:
    match = PROGRESS_RE.search(line.replace("\r", " ").strip())
    if not match:
        return None
    bytes_done = int(match.group("bytes").replace(",", ""))
    percent = float(match.group("percent"))
    speed_value = float(match.group("speed"))
    unit = match.group("unit").upper()
    multipliers = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }
    speed_bps = speed_value * multipliers.get(unit, 1)
    return RsyncProgress(bytes_done=bytes_done, percent=percent, speed_bps=speed_bps)


def build_rsync_command(
    settings: Settings,
    source: Path,
    remote_destination: str,
    *,
    is_directory: bool,
) -> list[str]:
    source_arg = str(source if not is_directory else f"{source}/")
    destination = f"{settings.remote_ssh_host}:{remote_destination}"
    if is_directory and not destination.endswith("/"):
        destination = f"{destination}/"
    return [
        "rsync",
        "-a",
        "--info=progress2",
        "-e",
        (
            f"ssh -o BatchMode=yes -o ConnectTimeout={settings.remote_connect_timeout} "
            "-o StrictHostKeyChecking=accept-new"
        ),
        source_arg,
        destination,
    ]


def rsync_path(
    settings: Settings,
    source: Path,
    remote_destination: str,
    *,
    is_directory: bool,
    job_id: int | None = None,
    on_progress: Callable[[RsyncProgress], None] | None = None,
    timeout: int | None = None,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"rsync source missing: {source}")

    command = build_rsync_command(
        settings,
        source,
        remote_destination,
        is_directory=is_directory,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if job_id is not None:
        ACTIVE_SYNC_PROCESSES[job_id] = process

    buffer = ""
    deadline = time.monotonic() + timeout if timeout else None
    try:
        assert process.stdout is not None
        while True:
            if deadline is not None and time.monotonic() > deadline:
                process.kill()
                raise TimeoutError("rsync timed out")
            chunk = process.stdout.read(256)
            if not chunk:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            buffer += chunk
            while True:
                if "\r" in buffer:
                    line, buffer = buffer.split("\r", 1)
                elif "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                else:
                    break
                progress = parse_rsync_progress_line(line)
                if progress and on_progress:
                    on_progress(progress)
        if buffer.strip():
            progress = parse_rsync_progress_line(buffer)
            if progress and on_progress:
                on_progress(progress)
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"rsync failed with exit code {returncode}")
    finally:
        if job_id is not None:
            ACTIVE_SYNC_PROCESSES.pop(job_id, None)


def cancel_active_rsync(job_id: int) -> bool:
    process = ACTIVE_SYNC_PROCESSES.get(job_id)
    if process is None or process.poll() is not None:
        return False
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    return True


def rsync_pull(
    settings: Settings,
    remote_source: str,
    local_destination: Path,
    *,
    is_directory: bool = True,
) -> None:
    local_destination.mkdir(parents=True, exist_ok=True)
    source = f"{settings.remote_ssh_host}:{remote_source}"
    if is_directory and not source.endswith("/"):
        source = f"{source}/"
    destination = str(local_destination if not is_directory else f"{local_destination}/")
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            (
                f"ssh -o BatchMode=yes -o ConnectTimeout={settings.remote_connect_timeout} "
                "-o StrictHostKeyChecking=accept-new"
            ),
            source,
            destination,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "rsync pull failed").strip()
        raise RuntimeError(detail)
