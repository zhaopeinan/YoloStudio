from __future__ import annotations

from app.core.settings import Settings
from app.remote.ssh import quote_remote, run_ssh


def probe_remote_host(settings: Settings) -> dict:
    """Return connectivity + environment status for the configured remote host."""
    host_id = settings.remote_host_id
    label = settings.remote_host_label
    if not settings.remote_enabled:
        return {
            "id": host_id,
            "label": label,
            "enabled": False,
            "online": False,
            "ssh_host": settings.remote_ssh_host,
            "remote_root": str(settings.remote_root),
            "python": settings.remote_python,
            "ultralytics_ok": False,
            "cuda_available": False,
            "gpu_count": 0,
            "error": "远程训练已禁用（YOLOSTUDIO_REMOTE_ENABLED=false）",
        }

    try:
        hello = run_ssh(
            settings,
            "hostname && whoami && pwd",
            timeout=settings.remote_connect_timeout + 10,
            check=True,
        )
    except Exception as exc:
        return {
            "id": host_id,
            "label": label,
            "enabled": True,
            "online": False,
            "ssh_host": settings.remote_ssh_host,
            "remote_root": str(settings.remote_root),
            "python": settings.remote_python,
            "ultralytics_ok": False,
            "cuda_available": False,
            "gpu_count": 0,
            "error": f"SSH 连接失败: {exc}",
            "hostname": None,
        }

    hostname = (hello.stdout.splitlines() or [""])[0].strip() or None
    python = quote_remote(settings.remote_python)
    remote_root = quote_remote(settings.remote_root)
    env_script = (
        f"{python} - <<'PY'\n"
        "import json, shutil, subprocess\n"
        "payload = {"
        "'ultralytics_ok': False, 'cuda_available': False, 'gpu_count': 0, "
        "'gpus': [], 'torch': None, 'error': None"
        "}\n"
        "try:\n"
        "    import torch\n"
        "    payload['torch'] = getattr(torch, '__version__', None)\n"
        "    payload['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    count = int(torch.cuda.device_count()) if payload['cuda_available'] else 0\n"
        "    payload['gpu_count'] = count\n"
        "    payload['gpus'] = ["
        "{'index': i, 'name': torch.cuda.get_device_name(i)} for i in range(count)"
        "]\n"
        "except Exception as exc:\n"
        "    payload['error'] = f'torch: {exc.__class__.__name__}: {exc}'\n"
        "if not payload['gpus'] and shutil.which('nvidia-smi'):\n"
        "    try:\n"
        "        completed = subprocess.run(\n"
        "            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],\n"
        "            capture_output=True, text=True, check=False, timeout=10,\n"
        "        )\n"
        "        gpus = []\n"
        "        for line in completed.stdout.splitlines():\n"
        "            parts = [p.strip() for p in line.split(',', 1)]\n"
        "            if len(parts) == 2 and parts[0].isdigit():\n"
        "                gpus.append({'index': int(parts[0]), 'name': parts[1]})\n"
        "        if gpus:\n"
        "            payload['gpus'] = gpus\n"
        "            payload['gpu_count'] = len(gpus)\n"
        "            payload['cuda_available'] = True\n"
        "    except Exception:\n"
        "        pass\n"
        "try:\n"
        "    import ultralytics\n"
        "    payload['ultralytics_ok'] = True\n"
        "    payload['ultralytics'] = getattr(ultralytics, '__version__', None)\n"
        "except Exception as exc:\n"
        "    payload['ultralytics_ok'] = False\n"
        "    extra = f'ultralytics: {exc.__class__.__name__}: {exc}'\n"
        "    payload['error'] = (payload['error'] + '; ' + extra) if payload['error'] else extra\n"
        "print(json.dumps(payload))\n"
        "PY\n"
        f"mkdir -p {remote_root} && df -h {remote_root} | tail -n 1"
    )
    try:
        env = run_ssh(settings, env_script, timeout=60, check=True)
    except Exception as exc:
        return {
            "id": host_id,
            "label": label,
            "enabled": True,
            "online": True,
            "ssh_host": settings.remote_ssh_host,
            "remote_root": str(settings.remote_root),
            "python": settings.remote_python,
            "ultralytics_ok": False,
            "cuda_available": False,
            "gpu_count": 0,
            "error": f"远端环境探测失败: {exc}",
            "hostname": hostname,
        }

    import json

    lines = [line.strip() for line in env.stdout.splitlines() if line.strip()]
    payload = {
        "ultralytics_ok": False,
        "cuda_available": False,
        "gpu_count": 0,
        "error": None,
    }
    disk_line = None
    for line in lines:
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
        else:
            disk_line = line

    gpus = payload.get("gpus") if isinstance(payload.get("gpus"), list) else []
    normalized_gpus = []
    for item in gpus:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        normalized_gpus.append(
            {
                "index": index,
                "name": str(item.get("name") or f"GPU {index}"),
            }
        )
    gpu_count = int(payload.get("gpu_count") or len(normalized_gpus) or 0)

    return {
        "id": host_id,
        "label": label,
        "enabled": True,
        "online": True,
        "ssh_host": settings.remote_ssh_host,
        "remote_root": str(settings.remote_root),
        "python": settings.remote_python,
        "ultralytics_ok": bool(payload.get("ultralytics_ok")),
        "cuda_available": bool(payload.get("cuda_available")),
        "gpu_count": gpu_count,
        "gpus": normalized_gpus,
        "torch_version": payload.get("torch"),
        "ultralytics_version": payload.get("ultralytics"),
        "disk": disk_line,
        "error": payload.get("error"),
        "hostname": hostname,
    }
