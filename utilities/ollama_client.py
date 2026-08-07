import shutil
import subprocess
import tempfile
import time
import os

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:31b"
STARTUP_TIMEOUT_SEC = 60
SERVE_LOG_PATH = os.path.join(tempfile.gettempdir(), "hypothesis_extractor_ollama_serve.log")


def _tail(path, n_lines=20):
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:]).strip()
    except OSError:
        return "(no log available)"


class OllamaClient:
    """Talks to a local Ollama server, starting it (and pulling the model) if needed.

    Requires the `ollama` binary to already be installed (https://ollama.com/download)
    — that's a one-time manual step this can't safely do from Python. Everything after
    that (starting the daemon, pulling model weights) happens automatically on first use.
    """

    def __init__(self, model=DEFAULT_MODEL, host=None, auto_start=True):
        self.name = f"ollama/{model}"
        self.model = model
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.auto_start = auto_start
        self._ready = False
        self._n_calls = 0

    def _log_gpu_usage(self):
        try:
            resp = requests.get(f"{self.host}/api/ps", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            m = next((m for m in models if m.get("name") == self.model), None)
            if m is None:
                print(f"[ollama] GPU check: '{self.model}' not listed as loaded by /api/ps")
                return
            size = m.get("size", 0)
            size_vram = m.get("size_vram", 0)
            pct = 100 * size_vram / size if size else 0
            print(f"[ollama] GPU check: {self.model} using {size_vram / 1e9:.1f}GB/{size / 1e9:.1f}GB "
                  f"on GPU ({pct:.0f}%)")
        except requests.exceptions.RequestException as e:
            print(f"[ollama] GPU check failed: {e}")

    def _get_tags(self):
        """Returns the list of locally-available models, or None if the server isn't reachable."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=2)
            resp.raise_for_status()
            return resp.json().get("models", [])
        except requests.exceptions.RequestException:
            return None

    def ensure_ready(self):
        if self._ready or not self.auto_start:
            self._ready = True
            return

        t0 = time.monotonic()
        tags = self._get_tags()
        if tags is None:
            if shutil.which("ollama") is None:
                raise RuntimeError(
                    f"Ollama is not installed and no server is reachable at {self.host}. "
                    "Install it once from https://ollama.com/download, then re-run."
                )
            print(f"[ollama] No server reachable at {self.host} — starting `ollama serve` "
                  f"locally (logging to {SERVE_LOG_PATH})...")
            with open(SERVE_LOG_PATH, "w") as log_f:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=log_f, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            for _ in range(STARTUP_TIMEOUT_SEC):
                tags = self._get_tags()
                if tags is not None:
                    break
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Ollama server did not come up at {self.host} within {STARTUP_TIMEOUT_SEC}s. "
                    f"Last output from `ollama serve` ({SERVE_LOG_PATH}):\n{_tail(SERVE_LOG_PATH)}"
                )
            print(f"[ollama] Server is up ({time.monotonic() - t0:.1f}s).")

        have = {m.get("name", "") for m in tags}
        if self.model not in have:
            print(f"[ollama] Model '{self.model}' not found locally (have: {sorted(have) or 'none'}) "
                  f"— pulling now (one-time, can take a while for a ~20GB model)...")
            t_pull = time.monotonic()
            subprocess.run(["ollama", "pull", self.model], check=True)
            print(f"[ollama] Pulled {self.model} ({time.monotonic() - t_pull:.1f}s).")
        else:
            print(f"[ollama] Model '{self.model}' already available at {self.host} "
                  f"(readiness check took {time.monotonic() - t0:.1f}s).")

        self._ready = True

    def query(self, prompt, temperature=0.0, max_tokens=8192):
        self.ensure_ready()  # no-op after the first successful call
        t0 = time.monotonic()
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",  # keep weights resident between calls in a long autorate loop
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=1800,
        )
        response.raise_for_status()
        print(f"[ollama] query took {time.monotonic() - t0:.1f}s "
              f"({len(prompt)} prompt chars).")
        self._n_calls += 1
        if self._n_calls == 1 or self._n_calls % 50 == 0:
            self._log_gpu_usage()
        return response.json()["response"]
