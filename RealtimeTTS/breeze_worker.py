"""Own a packaged Breeze worker for the lifetime of the compatible API."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import socket
import subprocess
import time

import requests


@contextmanager
def local_worker(model: Path, voice_dir: Path, *, timeout: float = 180,
                 chunk_first: int = 4, chunk_max: int = 6):
    if not 1 <= chunk_first <= chunk_max <= 64:
        raise ValueError("Breeze chunks must satisfy 1 <= first <= max <= 64")
    model = model.expanduser().absolute()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        raise ValueError("--model must name a Breeze TTS 2 Q8_0 GGUF file")
    try:
        from realtimetts_breeze_native import executable_path
    except ImportError as exc:
        raise RuntimeError(
            "--model needs the optional realtimetts-breeze-native wheel; "
            "see docs/engines/breeze.md, or connect with --worker-url"
        ) from exc
    # Select both ports together; never reuse or terminate an existing worker.
    with ExitStack() as reserved:
        ports = []
        for _ in range(2):
            sock = reserved.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    http_port, ws_port = ports
    base_url, ws_url = f"http://127.0.0.1:{http_port}", f"ws://127.0.0.1:{ws_port}"
    voice_dir = voice_dir.expanduser().resolve()
    voice_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(executable_path()), str(model), "--host", "127.0.0.1",
         "--port", str(http_port), "--ws-port", str(ws_port),
         "--chunk-first", str(chunk_first), "--chunk-max", str(chunk_max), "--voices-dir", str(voice_dir)],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + timeout
        with requests.Session() as session:
            session.trust_env = False
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Breeze worker exited with code {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Breeze worker did not become ready; see its startup output")
                try:
                    response = session.get(base_url + "/health", timeout=1)
                    response.raise_for_status()
                except requests.RequestException:
                    time.sleep(0.2)
                    continue
                health = response.json()
                if (health.get("native_checkpoints") is not True
                        or health.get("sample_rate") != 24000
                        or health.get("ws_port") != ws_port):
                    raise RuntimeError("Breeze worker has incompatible native controls or audio format")
                if not str(health.get("backend", "")).lower().startswith("vulkan"):
                    raise RuntimeError("Breeze could not select a Vulkan GPU; check the GPU driver. "
                                       "Use --worker-url for an explicitly configured CPU worker.")
                if process.poll() is not None:
                    raise RuntimeError("Breeze worker exited during startup")
                break
        yield base_url, ws_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
