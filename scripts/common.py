"""Shared paths, config loading, JSONL helpers and llama-server management."""

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUTS = ROOT / "inputs"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
TF = ROOT / "tf"
SCORES = ROOT / "scores"
FIGURES = ROOT / "figures"

GEN_EXPS = ["E1", "E3", "E4", "E4b", "E5", "E6"]


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def model_cfg(cfg, model):
    return cfg["models"][model]


def variant_cfg(cfg, model, variant):
    for v in model_cfg(cfg, model)["variants"]:
        if v["name"] == variant:
            return v
    raise KeyError(f"unknown variant {variant} for {model}")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dir(cfg):
    return (ROOT / cfg["llama_cpp"]["build"]).resolve()


def bin_path(cfg, name):
    return build_dir(cfg) / "bin" / name


def backend_env(cfg, device=None):
    """Environment for running llama.cpp binaries: oneAPI vars for SYCL, plus the device pin."""
    env = dict(os.environ)
    lc = cfg["llama_cpp"]
    if lc["backend"] == "sycl":
        out = subprocess.run(
            ["bash", "-c", f"source {shlex.quote(lc['oneapi_setvars'])} >/dev/null 2>&1; env -0"],
            capture_output=True, check=True,
        ).stdout
        for item in out.split(b"\0"):
            if b"=" in item:
                k, v = item.split(b"=", 1)
                env[k.decode()] = v.decode()
        env["ONEAPI_DEVICE_SELECTOR"] = device or lc["device"]
    return env


def model_n_params(gguf_path):
    """Total parameter count, summed over every tensor in a GGUF file."""
    import sys
    sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))
    from gguf import GGUFReader
    reader = GGUFReader(str(gguf_path))
    return int(sum(int(t.n_elements) for t in reader.tensors))


class Server:
    """One llama-server per model file, started warm and kept for every experiment."""

    def __init__(self, cfg, model_file, device=None, log_path=None):
        self.cfg = cfg
        self.port = cfg["server"]["port"]
        self.url = f"http://127.0.0.1:{self.port}"
        self.cmd = [str(bin_path(cfg, "llama-server")), "-m", str(model_file),
                    "--port", str(self.port), *cfg["server"]["args"]]
        self.env = backend_env(cfg, device)
        self.log_path = log_path
        self.proc = None
        self.peak_rss_mib = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self, timeout=600):
        log = open(self.log_path, "a") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}; see {self.log_path}")
            try:
                if requests.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server did not become healthy")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.peak_rss_mib = read_peak_rss_mib(self.proc.pid)
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def completion(self, prompt, n_predict):
        body = {"prompt": prompt, "n_predict": n_predict, **self.cfg["request"]}
        r = requests.post(self.url + "/completion", json=body, timeout=3600)
        r.raise_for_status()
        d = r.json()
        finish = {"limit": "length"}.get(d.get("stop_type"), "stop")
        return body, {"output": d["content"], "tokens_in": d.get("tokens_evaluated"),
                      "tokens_out": d.get("tokens_predicted"), "finish_reason": finish,
                      "timings": d.get("timings")}

    def chat(self, user_message, n_predict):
        body = {"messages": [{"role": "user", "content": user_message}], "max_tokens": n_predict,
                "chat_template_kwargs": self.cfg["chat_template_kwargs"], **self.cfg["request"]}
        r = requests.post(self.url + "/v1/chat/completions", json=body, timeout=3600)
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        return body, {"output": ch["message"].get("content") or "",
                      "reasoning": ch["message"].get("reasoning_content") or "",
                      "tokens_in": d["usage"]["prompt_tokens"], "tokens_out": d["usage"]["completion_tokens"],
                      "finish_reason": ch["finish_reason"], "timings": d.get("timings")}

    def run(self, inp):
        if inp["endpoint"] == "completion":
            return self.completion(inp["prompt"], inp["n_predict"])
        return self.chat(inp["prompt"], inp["n_predict"])


def read_peak_rss_mib(pid):
    """Peak resident host memory of a process (VmHWM), in MiB."""
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


class ResourceSampler:
    """Samples GPU memory, memory-bandwidth utilization and power once a second with xpu-smi.

    The device index is the N in level_zero:N; both enumerate the B60s in PCI order.
    """

    def __init__(self, device, csv_path):
        self.index = device.split(":")[-1]
        self.csv_path = csv_path
        self.proc = None

    def baseline_mib(self):
        rows = self._parse(subprocess.run(
            ["xpu-smi", "dump", "--device", self.index, "--metrics", "m", "--interval", "1", "--number", "1"],
            capture_output=True, text=True).stdout)
        return rows[0].get("memory.used (MiB)") if rows else None

    def __enter__(self):
        self.baseline = self.baseline_mib()
        self.proc = subprocess.Popen(
            ["xpu-smi", "dump", "--device", self.index, "--metrics", "u,m,p", "--interval", "1"],
            stdout=open(self.csv_path, "w"), stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        drop_timestamp_column(self.csv_path)

    @staticmethod
    def _parse(text):
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return []
        head = [h.strip() for h in lines[0].split(",")]
        rows = []
        for l in lines[1:]:
            vals = [v.strip() for v in l.split(",")]
            row = {}
            for h, v in zip(head, vals):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = None
            rows.append(row)
        return rows

    def summary(self):
        rows = self._parse(open(self.csv_path).read())
        def col(name):
            return [r[name] for r in rows if r.get(name) is not None]
        used, power, bw = col("memory.used (MiB)"), col("power.draw (W)"), col("utilization.memory (%)")
        return {
            "samples": len(rows),
            "vram_baseline_mib": self.baseline,
            "vram_peak_mib": max(used) if used else None,
            "vram_delta_mib": (max(used) - self.baseline) if used and self.baseline is not None else None,
            "power_mean_w": sum(power) / len(power) if power else None,
            "power_peak_w": max(power) if power else None,
            "mem_bw_util_mean_pct": sum(bw) / len(bw) if bw else None,
        }


def drop_timestamp_column(csv_path):
    """Remove xpu-smi's wall-clock Timestamp column, so published samples carry no time of day."""
    lines = open(csv_path).read().splitlines()
    if not lines or not lines[0].startswith("Timestamp"):
        return
    with open(csv_path, "w") as f:
        for line in lines:
            f.write(line.split(",", 1)[1].lstrip() + "\n" if "," in line else line + "\n")
