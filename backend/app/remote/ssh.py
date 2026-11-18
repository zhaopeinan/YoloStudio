from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from app.core.settings import Settings


@dataclass
class SSHResult:
    returncode: int
    stdout: str
    stderr: str


def build_ssh_base(settings: Settings) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={settings.remote_connect_timeout}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        settings.remote_ssh_host,
    ]


def run_ssh(
    settings: Settings,
    remote_command: str,
    *,
    timeout: int | None = 60,
    check: bool = False,
) -> SSHResult:
    completed = subprocess.run(
        [*build_ssh_base(settings), remote_command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    result = SSHResult(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "ssh failed").strip()
        raise RuntimeError(detail)
    return result


def quote_remote(path: str | object) -> str:
    return shlex.quote(str(path))


def ensure_remote_dirs(settings: Settings, *paths: object) -> None:
    if not paths:
        return
    joined = " ".join(quote_remote(path) for path in paths)
    run_ssh(settings, f"mkdir -p {joined}", check=True, timeout=settings.remote_connect_timeout + 30)
