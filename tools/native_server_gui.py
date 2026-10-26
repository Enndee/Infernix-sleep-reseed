"""Windows tray control panel for the isolated native Infernix server on port 8011."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import hashlib
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk


def is_frozen() -> bool:
    """True when this panel runs from a PyInstaller bundle instead of a checkout."""
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> Path:
    """Bundled data files: the tools directory from source, the bundle directory when frozen."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def resolve_repo_root() -> Path:
    """Locate the checkout this panel drives, for both source and frozen launches.

    A frozen panel cannot derive its checkout from ``__file__``, so it reads the
    ``INFERNIX_REPO`` environment variable first and then a ``repo-root.txt``
    file next to the executable. Source launches keep the checkout-relative rule.
    """
    override = os.environ.get("INFERNIX_REPO", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        pointer = Path(sys.executable).resolve().parent / "repo-root.txt"
        try:
            recorded = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded:
            return Path(recorded).expanduser().resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def ensure_console_streams() -> None:
    """Give a windowed frozen panel working stdout/stderr; ``INFERNIX_PANEL_LOG`` keeps a log.

    PyInstaller ``--noconsole`` builds have no console handles, so ``sys.stdout``
    and ``sys.stderr`` are ``None`` and a plain ``print`` would raise there.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    log_path = os.environ.get("INFERNIX_PANEL_LOG", "").strip()
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if log_path:
            try:
                stream = open(log_path, "a", encoding="utf-8", buffering=1)
            except OSError:
                stream = None
        if stream is None:
            stream = open(os.devnull, "w", encoding="utf-8")
        setattr(sys, name, stream)


def self_launch_command(*flags: str) -> str:
    """Sign-in command that reopens this panel, from a checkout or a frozen bundle."""
    if is_frozen():
        parts = [f'"{Path(sys.executable).resolve()}"']
    else:
        pythonw = Path(sys.executable)
        if pythonw.name.lower() == "python.exe":
            candidate = pythonw.with_name("pythonw.exe")
            if candidate.is_file():
                pythonw = candidate
        parts = [f'"{pythonw}"', f'"{Path(__file__).resolve()}"']
    return " ".join(parts + list(flags))


REPO = resolve_repo_root()
RUN_DIR = REPO / "build-baseline" / "native-run"
GUI_SETTINGS = RUN_DIR / "gui-settings.json"
RUNNER_CONFIG = RUN_DIR / "config.json"
PID_FILE = RUN_DIR / "infernix-serve.pid"
STDOUT_LOG = RUN_DIR / "infernix-serve.stdout.log"
STDERR_LOG = RUN_DIR / "infernix-serve.stderr.log"
SERVER_EXE = REPO / "build-baseline" / "apps" / "Release" / "infernix-serve.exe"
MODEL = Path(
    r"F:\Models\Qwen3.8-Flash-Next-Uncensored-Infernix"
    r"\Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix-00001-of-00003.infernix"
)
NGRAM = Path(r"F:\Models\Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram")
TEMPLATE = Path(r"F:\Models\infernix\infernix-2026.10.18-windows-x64\chat_templates\qwen3_8.jinja")
EXPERT_STATE = RUN_DIR / "native-build.expert-state"
HOST = "127.0.0.1"
PORT = 8011
BASE_URL = f"http://{HOST}:{PORT}"
DEFAULT_MODEL_ID = "Qwen3.8-Flash-Next-Uncensored-NVFP4-Infernix"
KV_DTYPES = ("int8", "bf16", "fp8", "nvfp4", "k8v4", "vq2", "k4v2")
# Single dark palette for ttk styles and the plain Tk text widgets.
THEME_BG = "#1f2124"
THEME_PANEL = "#2a2d31"
THEME_FIELD = "#17191c"
THEME_TEXT = "#e3e5e8"
THEME_MUTED = "#98a0a8"
THEME_ACCENT = "#5b9cf6"
THEME_BORDER = "#3d4248"
THEME_SELECT = "#2f5fa8"
MAX_LOG_CHARS = 400_000
MAX_HISTORY_ROWS = 300
STARTUP_TIMEOUT_SECONDS = 900
APP_NAME = "InfernixNativeServerGUI"
WINDOW_CLASS = APP_NAME + "_" + hashlib.sha256(str(REPO).lower().encode("utf-8")).hexdigest()[:16]
QWEN_AUTO_LOAD_APP_NAME = "InfernixExperimentalQwenAutoLoad"
QWEN_ICON = bundle_dir() / "assets" / "qwen.ico"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 0x102

_REQUEST_RE = re.compile(
    r"req#(?P<id>\d+)\s+done\s+\|\s*(?P<kind>[^|]+)\|.*?"
    r"\bprompt\s+(?P<prompt>[\d,]+)\s*\|\s*output\s+(?P<output>[\d,]+).*?"
    r"\bTTFT\s+(?P<ttft>[\d.]+\s*(?:ms|s)).*?\btotal\s+(?P<total>[\d.]+\s*s).*?"
    r"\bprefill\s+(?P<prefill>[\d.]+)\s+tok/s\s*\|\s*"
    r"decode\s+(?P<decode>[\d.]+)\s+tok/s"
)
_LOG_TIME_RE = re.compile(r"^(?P<stamp>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)")


class SettingsError(ValueError):
    """A displayed server setting is not safe or supported."""


def default_settings() -> dict[str, Any]:
    env: dict[str, Any] = {}
    try:
        existing = json.loads(RUNNER_CONFIG.read_text(encoding="utf-8"))
        env = existing.get("env", {})
    except (OSError, ValueError, TypeError):
        pass
    return {
        "max_context": 262144,
        "vram_headroom_mib": 12288,
        "ram_headroom_mib": 24576,
        "kv_dtype": "int8",
        "max_concurrency": 1,
        "max_thinking_tokens": 512,
        "sleep_after_seconds": int(env.get("INFERNIX_SLEEP_AFTER_SECONDS", 15)),
        "sleep_keep_frames": int(env.get("INFERNIX_SLEEP_KEEP_FRAMES", 0)),
        "sleep_scratch": str(env.get("INFERNIX_SLEEP_SCRATCH", "1")) == "1",
        "sleep_weights": str(env.get("INFERNIX_SLEEP_WEIGHTS", "1")) == "1",
        "sleep_kv": str(env.get("INFERNIX_SLEEP_KV", "1")) == "1",
        "launch_at_login": False,
        "auto_load_qwen_at_login": False,
    }


def load_settings() -> dict[str, Any]:
    settings = default_settings()
    try:
        saved = json.loads(GUI_SETTINGS.read_text(encoding="utf-8"))
        if isinstance(saved, dict):
            settings.update(saved)
    except (OSError, ValueError):
        pass
    return validate_settings(settings)


def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate settings before saving or starting the server."""
    try:
        result = {
            "max_context": int(settings["max_context"]),
            "vram_headroom_mib": int(settings["vram_headroom_mib"]),
            "ram_headroom_mib": int(settings["ram_headroom_mib"]),
            "kv_dtype": str(settings["kv_dtype"]),
            "max_concurrency": int(settings.get("max_concurrency", 1)),
            "max_thinking_tokens": int(settings.get("max_thinking_tokens", 512)),
            "sleep_after_seconds": int(settings["sleep_after_seconds"]),
            "sleep_keep_frames": int(settings["sleep_keep_frames"]),
            "sleep_scratch": bool(settings["sleep_scratch"]),
            "sleep_weights": bool(settings["sleep_weights"]),
            "sleep_kv": bool(settings.get("sleep_kv", True)),
            "launch_at_login": bool(settings.get("launch_at_login", False)),
            "auto_load_qwen_at_login": bool(settings.get("auto_load_qwen_at_login", False)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SettingsError("Enter whole numbers for context, headroom, concurrency, thinking, and sleep settings.") from exc
    checks = (
        (1024 <= result["max_context"] <= 262144, "Context size must be between 1,024 and 262,144 tokens."),
        (0 <= result["vram_headroom_mib"] <= 32768, "VRAM headroom must be between 0 and 32,768 MiB."),
        (0 <= result["ram_headroom_mib"] <= 262144, "RAM headroom must be between 0 and 262,144 MiB."),
        (result["kv_dtype"] in KV_DTYPES, f"KV type must be one of: {', '.join(KV_DTYPES)}."),
        (1 <= result["max_concurrency"] <= 8, "Maximum concurrency must be between 1 and 8 sequences."),
        (0 <= result["max_thinking_tokens"] <= 4294967295,
         "Maximum thinking tokens must be 0 (no explicit cap) or between 1 and 4,294,967,295."),
        (0 <= result["sleep_after_seconds"] <= 86400, "Idle sleep must be between 0 and 86,400 seconds."),
        (0 <= result["sleep_keep_frames"] <= 1048576, "Kept expert frames must be between 0 and 1,048,576."),
    )
    for valid, explanation in checks:
        if not valid:
            raise SettingsError(explanation)
    if result["sleep_after_seconds"] == 0 and (result["sleep_scratch"] or result["sleep_kv"]):
        raise SettingsError("Set a nonzero idle sleep delay before enabling scratch or KV VMM sleep.")
    return result


def registered_login_options() -> tuple[bool, bool]:
    """Return the independent tray-panel and experimental-Qwen sign-in options."""
    if os.name != "nt":
        return False, False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
            try:
                panel, _ = winreg.QueryValueEx(key, APP_NAME)
            except FileNotFoundError:
                panel = ""
            try:
                qwen, _ = winreg.QueryValueEx(key, QWEN_AUTO_LOAD_APP_NAME)
            except FileNotFoundError:
                qwen = ""
    except FileNotFoundError:
        return False, False
    return bool(panel), bool(qwen)


def build_server_arguments(settings: dict[str, Any]) -> list[str]:
    values = validate_settings(settings)
    arguments = [
        str(SERVER_EXE), str(MODEL), "--host", HOST, "--port", str(PORT),
        "--max-context", str(values["max_context"]), "--kv-dtype", values["kv_dtype"],
        "--max-concurrency", str(values["max_concurrency"]), "--ram-headroom-mib", str(values["ram_headroom_mib"]),
        "--vram-headroom-mib", str(values["vram_headroom_mib"]), "--expert-pin-mib", "auto",
        "--ngram-volume", str(NGRAM), "--ngram-residency", "volume",
        "--chat-template", str(TEMPLATE), "--expert-state", str(EXPERT_STATE),
        "--spec", "mtp", "--draft-tokens", "auto", "--lm-head-draft",
        "--vision", "--vision-offload", "on",
    ]
    if values["max_thinking_tokens"] > 0:
        arguments.extend(("--default-thinking-budget", str(values["max_thinking_tokens"])))
    return arguments


def write_startup_configuration(settings: dict[str, Any]) -> None:
    values = validate_settings(settings)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(RUNNER_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            config = {}
    except (OSError, ValueError):
        config = {}
    env = config.get("env", {})
    if not isinstance(env, dict):
        env = {}
    env.update({
        "INFERNIX_SLEEP_AFTER_SECONDS": str(values["sleep_after_seconds"]),
        "INFERNIX_SLEEP_KEEP_FRAMES": str(values["sleep_keep_frames"]),
        "INFERNIX_SLEEP_SCRATCH": "1" if values["sleep_scratch"] else "0",
        "INFERNIX_SLEEP_WEIGHTS": "1" if values["sleep_weights"] else "0",
        "INFERNIX_SLEEP_KV": "1" if values["sleep_kv"] else "0",
    })
    config["env"] = env
    temporary = RUNNER_CONFIG.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, RUNNER_CONFIG)
    temporary = GUI_SETTINGS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, GUI_SETTINGS)


def parse_server_request(line: str) -> dict[str, Any] | None:
    """Parse Infernix's one-line completion summary without exposing prompt text."""
    match = _REQUEST_RE.search(line)
    if not match:
        return None
    result: dict[str, Any] = match.groupdict()
    for name in ("prompt", "output"):
        result[name] = int(result[name].replace(",", ""))
    result["prefill"] = float(result["prefill"])
    result["decode"] = float(result["decode"])
    stamp = _LOG_TIME_RE.match(line)
    result["time"] = stamp.group("stamp")[11:19] if stamp else datetime.now().strftime("%H:%M:%S")
    result["line"] = line.rstrip("\r\n")
    return result


def process_command_line(pid: int) -> dict[str, Any] | None:
    """Read command-line identity for one PID; callers still verify its image path."""
    powershell = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    if not Path(powershell).is_file():
        return None
    script = (
        f"$p = Get-CimInstance -ClassName Win32_Process -Filter 'ProcessId = {int(pid)}'; "
        "if ($null -ne $p) { $p | Select-Object ProcessId,ExecutablePath,CommandLine "
        "| ConvertTo-Json -Compress }"
    )
    try:
        result = subprocess.run(
            [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5, check=False,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode or not result.stdout.strip():
            return None
        info = json.loads(result.stdout)
        return info if isinstance(info, dict) else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


CREATE_NO_WINDOW = 0x08000000


def _kernel32() -> Any:
    if os.name != "nt":
        raise OSError("Server process control is supported only on Windows.")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p,
                                                 ctypes.POINTER(ctypes.c_ulong)]
    kernel.QueryFullProcessImageNameW.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


def is_experimental_command(command: str) -> bool:
    """Parse Windows quoting; reject duplicate ports and production-port overrides."""
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel = _kernel32()
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    pointer = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not pointer:
        return False
    try:
        arguments = [pointer[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(pointer)
    ports = []
    for index, argument in enumerate(arguments):
        if argument == "--port":
            ports.append(arguments[index + 1] if index + 1 < len(arguments) else "")
        elif argument.startswith("--port="):
            ports.append(argument.partition("=")[2])
    return ports == [str(PORT)]


def inspect_owned_server(pid: int, require_port: bool = True) -> tuple[bool, str]:
    """Verify PID, image, and (by default) exact experimental --port before control."""
    kernel = _kernel32()
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_wchar_p,
                                                   ctypes.POINTER(ctypes.c_ulong)]
    kernel.QueryFullProcessImageNameW.restype = ctypes.c_int
    handle = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False, "Recorded server process is not running or cannot be inspected."
    try:
        image = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(len(image))
        if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            return False, "Windows could not verify the recorded process image."
        if os.path.normcase(os.path.abspath(image.value)) != os.path.normcase(os.path.abspath(SERVER_EXE)):
            return False, "Recorded PID belongs to a different executable; it will not be stopped."
        info = process_command_line(pid)
        if not info or not isinstance(info.get("CommandLine"), str):
            return False, "Could not verify the process command line; it will not be stopped."
        command = info["CommandLine"]
        if require_port and not is_experimental_command(command):
            return False, "Recorded executable is not verified on experimental port 8011."
        return True, command
    finally:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


def stop_owned_server() -> tuple[bool, str]:
    """Terminate only the PID recorded for this checkout after strict identity checks."""
    try:
        pid = int(PID_FILE.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return False, "No valid experimental-server PID is recorded."
    kernel = _kernel32()
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | SYNCHRONIZE, False, pid)
    if not handle:
        return False, "Recorded server process is no longer running."
    try:
        image = ctypes.create_unicode_buffer(32768)
        size = ctypes.c_ulong(len(image))
        if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            return False, "Windows could not verify the recorded process image; refusing to stop it."
        if os.path.normcase(os.path.abspath(image.value)) != os.path.normcase(os.path.abspath(SERVER_EXE)):
            return False, "Recorded PID belongs to a different executable; refusing to stop it."
        info = process_command_line(pid)
        command = info.get("CommandLine", "") if info else ""
        if not isinstance(command, str) or not is_experimental_command(command):
            return False, "PID is not verified as this checkout's port-8011 server; refusing to stop it."
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess.restype = ctypes.c_int
        if not kernel.TerminateProcess(handle, 1):
            return False, f"Windows could not stop verified server PID {pid}."
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        outcome = kernel.WaitForSingleObject(handle, 20_000)
        if outcome == WAIT_TIMEOUT:
            return False, f"Verified server PID {pid} did not exit within 20 seconds."
        if outcome != WAIT_OBJECT_0:
            return False, f"Windows could not confirm exit of verified server PID {pid}."
        try:
            PID_FILE.unlink()
        except OSError:
            pass
        return True, f"Stopped verified experimental server PID {pid} on port 8011."
    finally:
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)


def _port_is_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((HOST, PORT)) == 0


def fetch_models(timeout: float = 1.5) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=timeout) as response:
            body = json.loads(response.read(64_000).decode("utf-8"))
        return body["data"][0] if body.get("data") else None
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError):
        return None


def launch_server(settings: dict[str, Any]) -> subprocess.Popen[bytes]:
    """Launch the experimental binary on 8011; never invoke the production supervisor."""
    if os.name != "nt":
        raise OSError("The experimental native server GUI launcher is Windows-only.")
    if fetch_models() is not None:
        raise RuntimeError("Port 8011 already serves a model. The GUI will not start a duplicate server.")
    if _port_is_open():
        raise RuntimeError("Port 8011 is already occupied by an unverified process; refusing to start.")
    missing = [path for path in (SERVER_EXE, MODEL, NGRAM, TEMPLATE) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required server asset is missing:\n" + "\n".join(map(str, missing)))

    write_startup_configuration(settings)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
    with STDOUT_LOG.open("a", encoding="utf-8", errors="replace") as stdout, STDERR_LOG.open(
        "a", encoding="utf-8", errors="replace"
    ) as stderr:
        try:
            process = subprocess.Popen(
                build_server_arguments(settings), cwd=str(SERVER_EXE.parent), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=flags, close_fds=True,
                env={**os.environ, **_child_environment(settings)},
            )
        except OSError:
            process = subprocess.Popen(
                build_server_arguments(settings), cwd=str(SERVER_EXE.parent), stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True,
                env={**os.environ, **_child_environment(settings)},
            )
    PID_FILE.write_text(f"{process.pid}\n", encoding="ascii")
    return process


def _child_environment(settings: dict[str, Any]) -> dict[str, str]:
    values = validate_settings(settings)
    return {
        "INFERNIX_SLEEP_AFTER_SECONDS": str(values["sleep_after_seconds"]),
        "INFERNIX_SLEEP_KEEP_FRAMES": str(values["sleep_keep_frames"]),
        "INFERNIX_SLEEP_SCRATCH": "1" if values["sleep_scratch"] else "0",
        "INFERNIX_SLEEP_WEIGHTS": "1" if values["sleep_weights"] else "0",
        "INFERNIX_SLEEP_KV": "1" if values["sleep_kv"] else "0",
    }


def parse_streaming_completion(response: Any, on_text: Any) -> dict[str, Any]:
    """Consume an OpenAI SSE response and measure output-token rate after first text."""
    started = time.perf_counter()
    first_text_at: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    for raw in response:
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta", {})
            text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(text, str) and text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                on_text(text)
    finished = time.perf_counter()
    count = usage.get("completion_tokens")
    elapsed = max(0.001, finished - first_text_at) if first_text_at is not None else 0.0
    return {
        "completion_tokens": count if isinstance(count, int) else None,
        "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
        "generation_seconds": elapsed if first_text_at is not None else None,
        "tokens_per_second": count / elapsed if isinstance(count, int) and elapsed > 0 else None,
        "total_seconds": finished - started,
        "finish_reason": finish_reason,
    }


def send_prompt(prompt: str, max_tokens: int, on_text: Any) -> dict[str, Any]:
    model = fetch_models(timeout=5)
    if model is None:
        raise RuntimeError("Experimental server is not ready on port 8011.")
    request_body = json.dumps({
        "model": model.get("id", DEFAULT_MODEL_ID),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=request_body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return parse_streaming_completion(response, on_text)


def set_startup_registration(enabled: bool, auto_load_qwen: bool = False) -> None:
    """Update only this GUI's and experimental Qwen's per-user sign-in entries."""
    if os.name != "nt":
        return
    import winreg

    run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, run_key) as key:
        if enabled:
            command = self_launch_command("--hidden")
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, command)
        else:
            try:
                winreg.DeleteValue(key, APP_NAME)
            except FileNotFoundError:
                pass
        if auto_load_qwen:
            command = self_launch_command("--hidden", "--auto-load-qwen")
            winreg.SetValueEx(key, QWEN_AUTO_LOAD_APP_NAME, 0, winreg.REG_SZ, command)
        else:
            try:
                winreg.DeleteValue(key, QWEN_AUTO_LOAD_APP_NAME)
            except FileNotFoundError:
                pass


class SingleInstance:
    """One panel per checkout and interactive session, including both Run entries."""

    def __init__(self) -> None:
        self.kernel = _kernel32()
        self.kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel.CreateMutexW.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int
        self.handle = self.kernel.CreateMutexW(None, False, "Local\\" + WINDOW_CLASS)
        error = ctypes.get_last_error()
        if not self.handle:
            raise ctypes.WinError(error)
        self.primary = error != 183  # ERROR_ALREADY_EXISTS; keep the handle until exit.

    def notify(self, show: bool, auto_load: bool) -> bool:
        user = ctypes.WinDLL("user32", use_last_error=True)
        user.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user.FindWindowW.restype = ctypes.c_void_p
        user.PostMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
        user.PostMessageW.restype = ctypes.c_int
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            hwnd = user.FindWindowW(WINDOW_CLASS, None)
            if hwnd:
                return bool(user.PostMessageW(hwnd, TrayIcon.WM_INSTANCE, int(show), int(auto_load)))
            time.sleep(0.1)
        return False

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class TrayIcon:
    """Own a hidden top-level Win32 window; never subclass Tk's window procedure."""

    WM_TRAY = 0x8000 + 41
    WM_INSTANCE = 0x8000 + 42
    WM_LBUTTONUP = 0x0202
    WM_LBUTTONDBLCLK = 0x0203
    WM_RBUTTONUP = 0x0205
    WM_CONTEXTMENU = 0x007B
    WM_DESTROY = 0x0002
    WM_NULL = 0
    NIN_SELECT = 0x0400
    NIN_KEYSELECT = 0x0401
    NIM_ADD, NIM_DELETE, NIM_SETFOCUS, NIM_SETVERSION = 0, 2, 3, 4
    NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_SHOWTIP = 1, 2, 4, 0x80
    NOTIFYICON_VERSION_4 = 4
    MENU_OPEN = 41001
    MENU_START = 41002
    MENU_PANEL_STARTUP = 41003
    MENU_AUTO_LOAD = 41004
    MENU_FULL_UNLOAD = 41005
    MENU_EXIT = 41006

    class WindowClass(ctypes.Structure):
        _fields_ = [
            ("style", ctypes.c_uint), ("lpfnWndProc", ctypes.c_void_p),
            ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
            ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
            ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
            ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p),
        ]

    class NotifyIconData(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong), ("hWnd", ctypes.c_void_p), ("uID", ctypes.c_uint),
            ("uFlags", ctypes.c_uint), ("uCallbackMessage", ctypes.c_uint), ("hIcon", ctypes.c_void_p),
            ("szTip", ctypes.c_wchar * 128), ("dwState", ctypes.c_ulong), ("dwStateMask", ctypes.c_ulong),
            ("szInfo", ctypes.c_wchar * 256), ("uTimeoutOrVersion", ctypes.c_uint),
            ("szInfoTitle", ctypes.c_wchar * 64), ("dwInfoFlags", ctypes.c_ulong),
            ("guidItem", ctypes.c_byte * 16), ("hBalloonIcon", ctypes.c_void_p),
        ]

    def __init__(self, root: tk.Tk, on_open: Any, on_start: Any, on_full_unload: Any,
                 on_exit: Any, on_auto_load_changed: Any, auto_load_enabled: bool,
                 panel_startup_enabled: bool, on_panel_startup_changed: Any,
                 on_auto_load: Any = None) -> None:
        self.root = root
        self.callbacks = {
            self.MENU_OPEN: on_open, self.MENU_START: on_start,
            self.MENU_FULL_UNLOAD: on_full_unload, self.MENU_EXIT: on_exit,
        }
        self.on_auto_load_changed = on_auto_load_changed
        self.on_panel_startup_changed = on_panel_startup_changed
        self.on_auto_load = on_auto_load
        self.auto_load_enabled = auto_load_enabled
        self.panel_startup_enabled = panel_startup_enabled
        self.available = os.name == "nt"
        self.added = False
        self.error = ""
        self.hwnd = None
        self._class_registered = False
        self._icon_handle = None
        self._owns_icon = False
        self._removed = False
        self._wndproc = None
        self._pending: queue.Queue[Any] = queue.Queue()
        self._drain_after = None
        if not self.available:
            return
        try:
            self._initialize()
            self._drain_after = self.root.after(25, self._drain_pending)
        except (OSError, AttributeError, ctypes.ArgumentError) as exc:
            self.error = str(exc)
            self.remove()

    def _initialize(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        kernel = _kernel32()
        kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel.GetModuleHandleW.restype = ctypes.c_void_p
        self.instance = kernel.GetModuleHandleW(None)
        if not self.instance:
            raise ctypes.WinError(ctypes.get_last_error())
        user = self.user32
        # WPARAM/LPARAM/LRESULT and every handle must stay pointer-sized on Win64.
        signatures = {
            "RegisterClassW": ([ctypes.POINTER(self.WindowClass)], ctypes.c_ushort),
            "UnregisterClassW": ([ctypes.c_wchar_p, ctypes.c_void_p], ctypes.c_int),
            "CreateWindowExW": ([ctypes.c_ulong, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p),
            "DefWindowProcW": ([ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t], ctypes.c_ssize_t),
            "DestroyWindow": ([ctypes.c_void_p], ctypes.c_int),
            "LoadImageW": ([ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_int,
                            ctypes.c_int, ctypes.c_uint], ctypes.c_void_p),
            "LoadIconW": ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_void_p),
            "DestroyIcon": ([ctypes.c_void_p], ctypes.c_int),
            "CreatePopupMenu": ([], ctypes.c_void_p),
            "AppendMenuW": ([ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_wchar_p], ctypes.c_int),
            "TrackPopupMenu": ([ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p], ctypes.c_uint),
            "DestroyMenu": ([ctypes.c_void_p], ctypes.c_int),
            "GetCursorPos": ([ctypes.POINTER(ctypes.wintypes.POINT)], ctypes.c_int),
            "SetForegroundWindow": ([ctypes.c_void_p], ctypes.c_int),
            "PostMessageW": ([ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t], ctypes.c_int),
            "RegisterWindowMessageW": ([ctypes.c_wchar_p], ctypes.c_uint),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(user, name)
            function.argtypes = arguments
            function.restype = result
        self.shell32.Shell_NotifyIconW.argtypes = [ctypes.c_ulong, ctypes.POINTER(self.NotifyIconData)]
        self.shell32.Shell_NotifyIconW.restype = ctypes.c_int
        self._taskbar_created = user.RegisterWindowMessageW("TaskbarCreated")
        self._class_name = WINDOW_CLASS
        self._wndproc = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t
        )(self._window_proc)
        window_class = self.WindowClass()
        window_class.lpfnWndProc = ctypes.cast(self._wndproc, ctypes.c_void_p)
        window_class.hInstance = self.instance
        window_class.lpszClassName = self._class_name
        if not user.RegisterClassW(ctypes.byref(window_class)):
            raise ctypes.WinError(ctypes.get_last_error())
        self._class_registered = True
        # A top-level hidden window receives Explorer's TaskbarCreated broadcast.
        self.hwnd = user.CreateWindowExW(
            0, self._class_name, self._class_name, 0, 0, 0, 0, 0, None, None, self.instance, None,
        )
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        if QWEN_ICON.is_file():
            self._icon_handle = user.LoadImageW(None, str(QWEN_ICON), 1, 0, 0, 0x0010 | 0x0040)
            self._owns_icon = bool(self._icon_handle)
        if not self._icon_handle:
            self._icon_handle = user.LoadIconW(None, ctypes.c_void_p(32512))
        if not self._icon_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self._icon_data = self.NotifyIconData()
        self._icon_data.cbSize = ctypes.sizeof(self.NotifyIconData)
        self._icon_data.hWnd = self.hwnd
        self._icon_data.uID = 1
        self._icon_data.uFlags = self.NIF_MESSAGE | self.NIF_ICON | self.NIF_TIP | self.NIF_SHOWTIP
        self._icon_data.uCallbackMessage = self.WM_TRAY
        self._icon_data.hIcon = self._icon_handle
        self._icon_data.szTip = "Qwen — Infernix experimental server (port 8011)"
        self._add_icon()

    def _add_icon(self) -> None:
        self._icon_data.uTimeoutOrVersion = 0
        self.added = bool(self.shell32.Shell_NotifyIconW(self.NIM_ADD, ctypes.byref(self._icon_data)))
        if self.added:
            self._icon_data.uTimeoutOrVersion = self.NOTIFYICON_VERSION_4
            self.shell32.Shell_NotifyIconW(self.NIM_SETVERSION, ctypes.byref(self._icon_data))
        else:
            self.error = "Windows did not add the notification-area icon."

    def _schedule(self, callback: Any) -> None:
        if not self._removed:
            # This runs inside a ctypes WNDPROC entered by Tcl's message pump.
            # Even root.after() re-enters _tkinter here and can fatally corrupt
            # its saved thread state. Only Python queue operations are safe.
            self._pending.put(callback)

    def _drain_pending(self) -> None:
        try:
            while not self._removed:
                callback = self._pending.get_nowait()
                callback()
        except queue.Empty:
            pass
        finally:
            if not self._removed:
                self._drain_after = self.root.after(25, self._drain_pending)

    def _window_proc(self, hwnd: Any, message: int, wparam: int, lparam: int) -> int:
        try:
            if message == self.WM_TRAY:
                event = lparam & 0xFFFF
                if event in (self.WM_LBUTTONUP, self.WM_LBUTTONDBLCLK, self.NIN_SELECT, self.NIN_KEYSELECT):
                    self._schedule(self.callbacks[self.MENU_OPEN])
                elif event in (self.WM_RBUTTONUP, self.WM_CONTEXTMENU):
                    self._schedule(self._show_menu)
                return 0
            if message == self.WM_INSTANCE:
                if wparam:
                    self._schedule(self.callbacks[self.MENU_OPEN])
                if lparam and self.on_auto_load is not None:
                    self._schedule(self.on_auto_load)
                return 0
            if message == self._taskbar_created:
                self.added = False
                self._schedule(self._add_icon)
                return 0
            if message == self.WM_DESTROY:
                return 0  # No PostQuitMessage: Tk owns the thread's message loop.
        except Exception as exc:
            self.error = str(exc)  # Never let a Python exception escape a native callback.
        return int(self.user32.DefWindowProcW(hwnd, message, wparam, lparam))

    def _dispatch_menu(self, selected: int) -> None:
        if selected == self.MENU_AUTO_LOAD:
            self._schedule(lambda: self.on_auto_load_changed(not self.auto_load_enabled))
        elif selected == self.MENU_PANEL_STARTUP:
            self._schedule(lambda: self.on_panel_startup_changed(not self.panel_startup_enabled))
        elif selected in self.callbacks:
            self._schedule(self.callbacks[selected])

    def _show_menu(self) -> None:
        point = ctypes.wintypes.POINT()
        if not self.user32.GetCursorPos(ctypes.byref(point)):
            return
        menu = self.user32.CreatePopupMenu()
        if not menu:
            return
        selected = 0
        try:
            for command, label, checked in (
                (self.MENU_OPEN, "Open Qwen control panel", False),
                (self.MENU_START, "Start experimental Qwen", False),
                (self.MENU_PANEL_STARTUP, "Start control panel with Windows", self.panel_startup_enabled),
                (self.MENU_AUTO_LOAD, "Automatically load Qwen with Windows", self.auto_load_enabled),
                (0, None, False),
                (self.MENU_FULL_UNLOAD, "Full unload Qwen (release VRAM)", False),
                (self.MENU_EXIT, "Exit control panel (server stays running)", False),
            ):
                if not self.user32.AppendMenuW(menu, 0x0800 if label is None else (0x0008 if checked else 0), command, label):
                    raise ctypes.WinError(ctypes.get_last_error())
            self.user32.SetForegroundWindow(self.hwnd)
            selected = self.user32.TrackPopupMenu(menu, 0x0100 | 0x0002, point.x, point.y, 0, self.hwnd, None)
        finally:
            self.user32.DestroyMenu(menu)
            self.user32.PostMessageW(self.hwnd, self.WM_NULL, 0, 0)
            if self.added:
                self.shell32.Shell_NotifyIconW(self.NIM_SETFOCUS, ctypes.byref(self._icon_data))
        self._dispatch_menu(selected)

    def remove(self) -> None:
        self._removed = True
        if self._drain_after is not None:
            try:
                self.root.after_cancel(self._drain_after)
            except tk.TclError:
                pass
            self._drain_after = None
        if self.available and self.added:
            self.shell32.Shell_NotifyIconW(self.NIM_DELETE, ctypes.byref(self._icon_data))
            self.added = False
        if self.available and self.hwnd:
            self.user32.DestroyWindow(self.hwnd)
            self.hwnd = None
        if self.available and self._class_registered:
            self.user32.UnregisterClassW(self._class_name, self.instance)
            self._class_registered = False
        if self.available and self._icon_handle and self._owns_icon:
            self.user32.DestroyIcon(self._icon_handle)
        self._icon_handle = None


class NativeServerGUI:
    def __init__(self, start_hidden: bool = False, no_server_actions: bool = False) -> None:
        self.no_server_actions = no_server_actions
        self.root = tk.Tk()
        self.root.title("Qwen — Infernix Experimental Server")
        self.root.geometry("1100x790")
        self.root.minsize(860, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.hide_to_tray)
        self.settings = load_settings()
        registered_panel, registered_auto_load = registered_login_options()
        self.settings["launch_at_login"] = registered_panel
        self.settings["auto_load_qwen_at_login"] = registered_auto_load
        self.startup_var = tk.BooleanVar(value=registered_panel)
        self.auto_load_var = tk.BooleanVar(value=registered_auto_load)
        if QWEN_ICON.is_file():
            try:
                self.root.iconbitmap(default=str(QWEN_ICON))
            except tk.TclError:
                pass  # A missing/corrupt optional image must not prevent opening the panel.
        self.messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.closed = False
        self.log_offsets: dict[Path, int] = {}
        self._log_partial: dict[Path, str] = {}
        self.completed_rows: dict[str, dict[str, Any]] = {}
        self.total_output_tokens = 0
        self.total_decode_seconds = 0.0
        self.last_models: dict[str, Any] | None = None
        self.last_server_state = "Checking experimental server…"
        self._build_ui()
        self._load_recent_logs()
        self.tray = self._create_tray()
        self._drain_after = self.root.after(200, self._drain_messages)
        self._logs_after = self.root.after(500, self._poll_logs)
        if not self.no_server_actions:
            self._monitor_after = self.root.after(100, self._monitor_tick)
        if start_hidden and self.tray.added:
            self.root.withdraw()

    def _apply_dark_theme(self) -> None:
        """Apply one dark palette to ttk widgets and the plain Tk text surfaces."""
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            # The Windows "vista" theme paints its own light surfaces and ignores colors.
            style.theme_use("clam")
        self.root.configure(bg=THEME_BG)
        self.root.option_add("*TCombobox*Listbox.background", THEME_PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", THEME_TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", THEME_SELECT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure(
            ".", background=THEME_BG, foreground=THEME_TEXT, fieldbackground=THEME_FIELD,
            bordercolor=THEME_BORDER, lightcolor=THEME_PANEL, darkcolor=THEME_BG,
            troughcolor=THEME_BG, focuscolor=THEME_ACCENT,
            selectbackground=THEME_SELECT, selectforeground="#ffffff",
        )
        style.configure("TFrame", background=THEME_BG)
        style.configure("TLabel", background=THEME_BG, foreground=THEME_TEXT)
        style.configure("Footer.TLabel", background=THEME_BG, foreground=THEME_MUTED)
        style.configure("TLabelframe", background=THEME_BG, bordercolor=THEME_BORDER)
        style.configure("TLabelframe.Label", background=THEME_BG, foreground=THEME_ACCENT)
        style.configure("TButton", background=THEME_PANEL, foreground=THEME_TEXT,
                        bordercolor=THEME_BORDER, focuscolor=THEME_ACCENT, padding=(10, 4))
        style.map("TButton",
                  background=[("disabled", THEME_BG), ("pressed", THEME_SELECT), ("active", "#363a3f")],
                  foreground=[("disabled", THEME_MUTED)])
        style.configure("TNotebook", background=THEME_BG, bordercolor=THEME_BORDER, tabmargins=(2, 4, 2, 0))
        style.configure("TNotebook.Tab", background=THEME_PANEL, foreground=THEME_MUTED, padding=(14, 6))
        style.map("TNotebook.Tab", background=[("selected", THEME_BG)],
                  foreground=[("selected", "#ffffff")])
        style.configure("TCheckbutton", background=THEME_BG, foreground=THEME_TEXT,
                        indicatorbackground=THEME_FIELD, indicatorforeground=THEME_ACCENT)
        style.map("TCheckbutton", background=[("active", THEME_BG)],
                  indicatorbackground=[("selected", THEME_ACCENT), ("active", THEME_FIELD)])
        style.configure("TSpinbox", background=THEME_PANEL, foreground=THEME_TEXT,
                        fieldbackground=THEME_FIELD, arrowcolor=THEME_TEXT, bordercolor=THEME_BORDER)
        style.map("TSpinbox", fieldbackground=[("readonly", THEME_FIELD)],
                  arrowcolor=[("disabled", THEME_MUTED)])
        style.configure("TCombobox", background=THEME_PANEL, foreground=THEME_TEXT,
                        fieldbackground=THEME_FIELD, arrowcolor=THEME_TEXT, bordercolor=THEME_BORDER)
        style.map("TCombobox",
                  fieldbackground=[("readonly", THEME_FIELD)],
                  foreground=[("readonly", THEME_TEXT)],
                  selectbackground=[("readonly", THEME_FIELD)],
                  selectforeground=[("readonly", THEME_TEXT)],
                  arrowcolor=[("disabled", THEME_MUTED)])
        style.configure("Treeview", background=THEME_FIELD, fieldbackground=THEME_FIELD,
                        foreground=THEME_TEXT, bordercolor=THEME_BORDER,
                        lightcolor=THEME_BG, darkcolor=THEME_BG)
        style.map("Treeview", background=[("selected", THEME_SELECT)],
                  foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", background=THEME_PANEL, foreground=THEME_TEXT,
                        bordercolor=THEME_BORDER, relief="flat", padding=(6, 4))
        style.map("Treeview.Heading", background=[("active", "#363a3f")])
        for orientation in ("Vertical", "Horizontal"):
            style.configure(f"{orientation}.TScrollbar", background=THEME_PANEL, troughcolor=THEME_BG,
                            bordercolor=THEME_BORDER, arrowcolor=THEME_TEXT,
                            darkcolor=THEME_PANEL, lightcolor=THEME_PANEL)
            style.map(f"{orientation}.TScrollbar",
                      background=[("active", "#3f444a"), ("pressed", THEME_SELECT)])

    def _apply_dark_titlebar(self) -> None:
        """Ask DWM for a dark caption bar; ignored on Windows builds without the attribute."""
        if os.name != "nt":
            return
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            dwmapi = ctypes.WinDLL("dwmapi")
            user32.GetParent.argtypes = [ctypes.c_void_p]
            user32.GetParent.restype = ctypes.c_void_p
            self.root.update_idletasks()
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()
            dwmapi.DwmSetWindowAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                                     ctypes.c_void_p, ctypes.c_uint]
            dwmapi.DwmSetWindowAttribute.restype = ctypes.c_long
            enabled = ctypes.c_int(1)
            # 20 is DWMWA_USE_IMMERSIVE_DARK_MODE; 19 is the pre-20H1 attribute number.
            for attribute in (20, 19):
                if dwmapi.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(enabled),
                                                ctypes.sizeof(enabled)) == 0:
                    break
        except (OSError, AttributeError, TypeError, ctypes.ArgumentError):
            pass

    def _build_ui(self) -> None:
        self._apply_dark_theme()

        header = ttk.Frame(self.root, padding=(14, 12, 14, 8))
        header.pack(fill="x")
        self.status_dot = tk.Label(header, text="●", fg="#e0a33a", bg=THEME_BG, font=("Segoe UI", 15))
        self.status_dot.pack(side="left", padx=(0, 7))
        self.status_var = tk.StringVar(value=self.last_server_state)
        ttk.Label(header, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(side="left")
        self.rate_var = tk.StringVar(value="Avg decode —")
        ttk.Label(header, textvariable=self.rate_var, font=("Segoe UI", 10)).pack(side="left", padx=(18, 0))
        ttk.Label(header, text=f"Experimental API  {BASE_URL}").pack(side="left", padx=(18, 0))
        self.start_button = ttk.Button(header, text="Start", command=self.start_server)
        self.start_button.pack(side="right", padx=(6, 0))
        self.stop_button = ttk.Button(header, text="Stop", command=self.stop_server, state="disabled")
        self.stop_button.pack(side="right", padx=(6, 0))
        self.restart_button = ttk.Button(header, text="Restart with settings", command=self.restart_server, state="disabled")
        self.restart_button.pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Open API", command=lambda: webbrowser.open(BASE_URL + "/v1/models")).pack(
            side="right", padx=(6, 0)
        )

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.activity_tab = ttk.Frame(self.notebook, padding=10)
        self.prompt_tab = ttk.Frame(self.notebook, padding=12)
        self.settings_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.activity_tab, text="Activity & logs")
        self.notebook.add(self.prompt_tab, text="Send a prompt")
        self.notebook.add(self.settings_tab, text="Settings")
        self._build_activity_tab()
        self._build_prompt_tab()
        self._build_settings_tab()
        ttk.Label(
            self.root,
            text="Experimental instance only · fixed loopback port 8011 · closing this window hides it in the notification area",
            anchor="w", padding=(14, 4), style="Footer.TLabel",
        ).pack(fill="x")
        self._apply_dark_titlebar()

    def _build_activity_tab(self) -> None:
        summary = ttk.LabelFrame(self.activity_tab, text="Completed requests", padding=8)
        summary.pack(fill="both", expand=True)
        columns = ("time", "source", "prompt", "in", "out", "prefill", "decode")
        self.history = ttk.Treeview(summary, columns=columns, show="headings", height=10)
        headings = {
            "time": ("Time", 78), "source": ("Source / request ID", 245), "prompt": ("Prompt preview / route", 310),
            "in": ("Input", 70), "out": ("Output", 70), "prefill": ("Prefill tok/s", 90),
            "decode": ("Decode tok/s", 95),
        }
        for column, (label, width) in headings.items():
            self.history.heading(column, text=label)
            self.history.column(column, width=width, anchor="w" if column in ("source", "prompt") else "center")
        self.history.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(summary, orient="vertical", command=self.history.yview)
        self.history.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")

        log_frame = ttk.LabelFrame(self.activity_tab, text="Native server log", padding=6)
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        # Word wrap keeps long engine lines readable without a horizontal scrollbar.
        self.log_text = tk.Text(
            log_frame, height=17, state="disabled", font=("Cascadia Mono", 9),
            wrap="word", bg=THEME_FIELD, fg=THEME_TEXT, insertbackground=THEME_TEXT,
            selectbackground=THEME_SELECT, selectforeground="#ffffff",
            relief="flat", borderwidth=0, padx=8, pady=6,
            highlightthickness=1, highlightbackground=THEME_BORDER, highlightcolor=THEME_ACCENT,
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        yscroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=yscroll.set)

    def _build_prompt_tab(self) -> None:
        ttk.Label(
            self.prompt_tab,
            text="Send a single-turn chat completion to the experimental server. Prompt text and streamed output stay in this GUI; the server's request row supplies its decode-rate measurement.",
            wraplength=970, justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(self.prompt_tab, text="Prompt").pack(anchor="w")
        self.prompt_input = tk.Text(
            self.prompt_tab, height=7, wrap="word", font=("Segoe UI", 10),
            bg=THEME_FIELD, fg=THEME_TEXT, insertbackground=THEME_TEXT,
            selectbackground=THEME_SELECT, selectforeground="#ffffff",
            relief="flat", borderwidth=0, padx=8, pady=6,
            highlightthickness=1, highlightbackground=THEME_BORDER, highlightcolor=THEME_ACCENT,
        )
        self.prompt_input.pack(fill="x", pady=(4, 8))
        actions = ttk.Frame(self.prompt_tab)
        actions.pack(fill="x")
        ttk.Label(actions, text="Maximum output tokens:").pack(side="left")
        self.max_tokens_var = tk.IntVar(value=512)
        ttk.Spinbox(actions, from_=1, to=8192, width=9, textvariable=self.max_tokens_var).pack(side="left", padx=7)
        self.send_button = ttk.Button(actions, text="Send prompt", command=self.submit_prompt)
        self.send_button.pack(side="left")
        self.prompt_rate_var = tk.StringVar(value="Observed output rate: —")
        ttk.Label(actions, textvariable=self.prompt_rate_var, font=("Segoe UI", 10, "bold")).pack(side="right")
        ttk.Label(self.prompt_tab, text="Streaming response").pack(anchor="w", pady=(12, 3))
        self.prompt_output = tk.Text(
            self.prompt_tab, height=20, wrap="word", state="disabled", font=("Segoe UI", 10),
            bg=THEME_FIELD, fg=THEME_TEXT, insertbackground=THEME_TEXT,
            selectbackground=THEME_SELECT, selectforeground="#ffffff",
            relief="flat", borderwidth=0, padx=8, pady=6,
            highlightthickness=1, highlightbackground=THEME_BORDER, highlightcolor=THEME_ACCENT,
        )
        self.prompt_output.pack(fill="both", expand=True)

    def _build_settings_tab(self) -> None:
        form = ttk.Frame(self.settings_tab)
        form.pack(anchor="nw", fill="x")
        self.setting_vars: dict[str, tk.Variable] = {}
        integer_settings = [
            ("max_context", "Context size (tokens)", 0, 262144),
            ("vram_headroom_mib", "VRAM headroom target (MiB)", 0, 12288),
            ("ram_headroom_mib", "Host RAM headroom (MiB)", 0, 24576),
            ("max_concurrency", "Maximum concurrent sequences (1-8)", 1, 8),
            ("max_thinking_tokens", "Default thinking-token cap (0 = automatic)", 0, 4294967295),
            ("sleep_after_seconds", "Idle sleep after (seconds; 0 = disabled)", 0, 86400),
            ("sleep_keep_frames", "Expert frames to keep while asleep", 0, 1048576),
        ]
        for row, (key, label, minimum, maximum) in enumerate(integer_settings):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=6)
            variable = tk.StringVar(value=str(self.settings[key]))
            self.setting_vars[key] = variable
            ttk.Spinbox(form, from_=minimum, to=maximum, increment=1, width=16, textvariable=variable).grid(
                row=row, column=1, sticky="w", pady=6
            )
        row = len(integer_settings)
        ttk.Label(form, text="KV cache type").grid(row=row, column=0, sticky="w", padx=(0, 16), pady=6)
        self.setting_vars["kv_dtype"] = tk.StringVar(value=self.settings["kv_dtype"])
        ttk.Combobox(form, textvariable=self.setting_vars["kv_dtype"], values=KV_DTYPES, state="readonly", width=13).grid(
            row=row, column=1, sticky="w", pady=6
        )
        row += 1
        for key, label in (("sleep_scratch", "Release disposable scratch/workspace backing"),
                           ("sleep_weights", "Release resident-weight backing (uses ~5.75 GiB pinned RAM)"),
                           ("sleep_kv", "Offload KV cache to RAM while asleep (preserves cached context)")):
            self.setting_vars[key] = tk.BooleanVar(value=self.settings[key])
            ttk.Checkbutton(form, text=label, variable=self.setting_vars[key]).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=4
            )
            row += 1
        self.setting_vars["launch_at_login"] = self.startup_var
        ttk.Checkbutton(form, text="Start control panel at Windows sign-in (tray-only)",
                        variable=self.setting_vars["launch_at_login"], command=self._toggle_startup).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )
        row += 1
        self.setting_vars["auto_load_qwen_at_login"] = self.auto_load_var
        ttk.Checkbutton(
            form,
            text="Automatically load Qwen with Windows startup",
            variable=self.auto_load_var,
            command=self._toggle_auto_load,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        row += 1
        buttons = ttk.Frame(form)
        buttons.grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 6))
        ttk.Button(buttons, text="Save settings", command=self.save_settings).pack(side="left")
        ttk.Button(buttons, text="Restore tested defaults", command=self.restore_defaults).pack(side="left", padx=7)
        self.settings_note = tk.StringVar(
            value="Server flags apply on the next start. Higher concurrency allows more simultaneous sequences and may increase resource use. Thinking cap applies to thinking-enabled requests unless a request supplies its own budget. Sleep controls affect the experimental build only."
        )
        ttk.Label(form, textvariable=self.settings_note, wraplength=760, justify="left").grid(
            row=row + 1, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

    def _create_tray(self) -> TrayIcon:
        tray = TrayIcon(
            self.root, self.show_window, self.start_server, self.full_unload_server,
            self.exit_gui, self._apply_auto_load_setting, bool(self.auto_load_var.get()),
            bool(self.startup_var.get()), self._apply_startup_setting, self.auto_load_server,
        )
        if not tray.added:
            self.status_var.set(f"Tray icon unavailable; panel stays accessible. {tray.error}")
        return tray

    def _apply_startup_setting(self, enabled: bool) -> None:
        self.startup_var.set(enabled)
        self._toggle_startup()

    def _load_recent_logs(self) -> None:
        for path in (STDOUT_LOG, STDERR_LOG):
            try:
                size = path.stat().st_size
                with path.open("rb") as stream:
                    stream.seek(max(0, size - 1_000_000))
                    data = stream.read().decode("utf-8", "replace")
                self.log_offsets[path] = size
                lines = data.splitlines()
                if lines and not data.endswith(("\n", "\r")):
                    # Hold an unterminated final line back so the first poll can complete it.
                    self._log_partial[path] = lines.pop()
                if len(lines) > 500:
                    lines = lines[-500:]
                self._append_log("".join(f"{line}\n" for line in lines))
                if path == STDERR_LOG:
                    for line in data.splitlines():
                        record = parse_server_request(line)
                        if record:
                            self._record_server_request(record)
            except OSError:
                self.log_offsets[path] = 0
                self._log_partial.pop(path, None)

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        current = int(self.log_text.index("end-1c" ).split(".")[0])
        if current > 5000:
            self.log_text.delete("1.0", f"{current - 4000}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _record_server_request(self, record: dict[str, Any]) -> None:
        fingerprint = hashlib.sha256(record["line"].encode("utf-8", "replace")).hexdigest()[:16]
        item = f"server-{fingerprint}"
        if item in self.completed_rows:
            return
        self.completed_rows[item] = record
        if record["output"] > 0 and record["decode"] > 0:
            self.total_output_tokens += record["output"]
            self.total_decode_seconds += record["output"] / record["decode"]
            average = self.total_output_tokens / self.total_decode_seconds
            self.rate_var.set(f"Avg decode  {average:.1f} tok/s  ·  {len(self.completed_rows)} logged requests")
        values = (record["time"], f"Server req#{record['id']}", record["kind"].strip(),
                  f"{record['prompt']:,}", f"{record['output']:,}", f"{record['prefill']:.1f}", f"{record['decode']:.1f}")
        self._insert_history(item, values)

    def _insert_history(self, item: str, values: tuple[str, ...]) -> None:
        if self.history.exists(item):
            return
        self.history.insert("", 0, iid=item, values=values)
        children = self.history.get_children()
        for old in children[MAX_HISTORY_ROWS:]:
            self.history.delete(old)

    def _poll_logs(self) -> None:
        if self.closed:
            return
        for path in (STDOUT_LOG, STDERR_LOG):
            try:
                size = path.stat().st_size
                offset = self.log_offsets.get(path, 0)
                if size < offset:
                    offset = 0
                    self._log_partial.pop(path, None)
                if size == offset:
                    continue
                with path.open("rb") as stream:
                    stream.seek(offset)
                    data = stream.read(1_000_000).decode("utf-8", "replace")
                    self.log_offsets[path] = stream.tell()
                buffered = self._log_partial.get(path, "") + data
                lines = buffered.splitlines()
                if lines and not buffered.endswith(("\n", "\r")):
                    # Keep a half-written trailing line until the next poll completes it.
                    self._log_partial[path] = lines.pop()
                else:
                    self._log_partial[path] = ""
                self._append_log("".join(f"{line}\n" for line in lines))
                if path == STDERR_LOG:
                    for line in lines:
                        record = parse_server_request(line)
                        if record:
                            self._record_server_request(record)
            except OSError:
                continue
        self._logs_after = self.root.after(500, self._poll_logs)

    def _start_monitor(self) -> None:
        if self.closed or self.monitoring or self.no_server_actions:
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, name="infernix-status", daemon=True)
        self.monitor_thread.start()

    def _monitor_tick(self) -> None:
        if self.closed:
            return
        self._start_monitor()
        self._monitor_after = self.root.after(2500, self._monitor_tick)

    def _monitor_worker(self) -> None:
        pid: int | None = None
        try:
            raw = PID_FILE.read_text(encoding="ascii").strip()
            pid = int(raw)
        except (OSError, ValueError):
            pass
        owned = False
        detail = ""
        if pid is not None:
            try:
                owned, detail = inspect_owned_server(pid)
            except OSError as exc:
                detail = str(exc)
        models = fetch_models()
        if models is not None and not owned:
            state = ("Server responding on port 8011 (ownership unverified); stop disabled", "#c58a19", None, models)
        elif models is not None:
            state = (f"Running · PID {pid} · {models.get('id', DEFAULT_MODEL_ID)}", "#22863a", pid, models)
        elif pid is not None and owned:
            state = (f"Starting · PID {pid} · waiting for model readiness", "#c58a19", pid, None)
        elif pid is not None and detail and "different executable" in detail:
            state = ("Recorded PID is not this experimental server; controls disabled", "#c53030", None, None)
        elif _port_is_open():
            state = ("Port 8011 is occupied by an unverified process; start/stop disabled", "#c53030", None, None)
        else:
            state = ("Stopped · experimental server is not listening", "#6b7280", None, None)
        self.messages.put(("status", state))
        self.messages.put(("monitor-complete", None))

    def _drain_messages(self) -> None:
        if self.closed:
            return
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "status":
                    self._apply_status(payload)
                elif kind == "monitor-complete":
                    self.monitoring = False
                elif kind == "busy":
                    self.busy = payload
                    self._update_buttons()
                elif kind == "message":
                    self.settings_note.set(payload)
                elif kind == "prompt-token":
                    self._append_prompt_output(payload)
                elif kind == "prompt-finished":
                    self._prompt_finished(payload)
                elif kind == "server-log":
                    self._append_log(payload + "\n")
                elif kind == "monitor-now":
                    self._start_monitor()
        except queue.Empty:
            pass
        self._drain_after = self.root.after(150, self._drain_messages)

    def _apply_status(self, state: tuple[str, str, int | None, dict[str, Any] | None]) -> None:
        text, color, pid, models = state
        self.last_server_state = text
        self.last_models = models
        self.status_var.set(text)
        self.status_dot.configure(fg=color)
        owned_running = pid is not None
        unverified_ready = models is not None and not owned_running
        self.start_button.configure(state="disabled" if owned_running or unverified_ready or self.busy else "normal")
        self.stop_button.configure(state="normal" if owned_running and not self.busy else "disabled")
        self.restart_button.configure(state="normal" if owned_running and not self.busy else "disabled")

    def _update_buttons(self) -> None:
        if self.last_server_state.startswith("Running") or self.last_server_state.startswith("Starting"):
            self.stop_button.configure(state="disabled" if self.busy else "normal")
            self.restart_button.configure(state="disabled" if self.busy else "normal")
        else:
            self.stop_button.configure(state="disabled")
            self.restart_button.configure(state="disabled")
        if self.busy or "ownership unverified" in self.last_server_state or "unverified process" in self.last_server_state:
            self.start_button.configure(state="disabled")

    def _run_background(self, name: str, action: Any) -> None:
        if self.busy or self.no_server_actions:
            return
        self.busy = True
        self._update_buttons()

        def run() -> None:
            try:
                action()
            except Exception as exc:  # report worker failures in the UI
                self.messages.put(("message", f"{name} failed: {exc}"))
                self.messages.put(("server-log", f"[{name}] {exc}"))
            finally:
                self.messages.put(("busy", False))
                self.messages.put(("monitor-now", None))

        threading.Thread(target=run, name=f"infernix-{name.lower().replace(' ', '-')}", daemon=True).start()

    def auto_load_server(self) -> None:
        """Coalesce simultaneous sign-in launches; never duplicate an existing listener."""
        if self.no_server_actions or self.busy:
            return
        try:
            pid = int(PID_FILE.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            pid = None
        if pid is not None and inspect_owned_server(pid)[0]:
            return
        if not _port_is_open():
            self.start_server()

    def start_server(self) -> None:
        if self.no_server_actions:
            return
        try:
            settings = self._read_form_settings()
        except SettingsError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            self.notebook.select(self.settings_tab)
            return

        def start() -> None:
            self.messages.put(("message", "Starting the experimental server on 127.0.0.1:8011…"))
            process = launch_server(settings)
            self.messages.put(("server-log", f"Started experimental infernix-serve PID {process.pid} on port 8011"))
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server PID {process.pid} exited with code {process.returncode}. Check stderr logs.")
                model = fetch_models(timeout=2)
                if model is not None:
                    self.messages.put(("message", f"Server ready: {model.get('id', DEFAULT_MODEL_ID)}"))
                    return
                time.sleep(2)
            raise TimeoutError("Server did not become ready within 15 minutes; inspect native stderr logs.")

        self._run_background("Start", start)

    def stop_server(self) -> None:
        if self.no_server_actions:
            return
        if not messagebox.askyesno(
            "Stop experimental server?",
            "This stops only the verified experimental PID on port 8011. Any in-flight generation will be interrupted.",
            parent=self.root,
        ):
            return
        self._stop_verified_server("Stop")

    def full_unload_server(self) -> None:
        if self.no_server_actions:
            return
        if not messagebox.askyesno(
            "Full unload Qwen?",
            "This exits only the verified experimental Qwen process on port 8011 and releases its process-owned VRAM.\n\n"
            "Active generations and in-memory conversation state will be lost.",
            icon="warning", parent=self.root,
        ):
            return
        self._stop_verified_server("Full unload")

    def _stop_verified_server(self, action: str) -> None:
        def stop() -> None:
            ok, detail = stop_owned_server()
            self.messages.put(("message", detail))
            if not ok:
                raise RuntimeError(detail)
            self.messages.put(("server-log", detail))
        self._run_background(action, stop)

    def restart_server(self) -> None:
        if self.no_server_actions:
            return
        try:
            settings = self._read_form_settings()
        except SettingsError as exc:
            messagebox.showerror("Invalid settings", str(exc), parent=self.root)
            self.notebook.select(self.settings_tab)
            return
        if not messagebox.askyesno(
            "Restart experimental server?",
            "The current server will be stopped and restarted on port 8011. In-flight generations and conversation state will be lost.",
            parent=self.root,
        ):
            return

        def restart() -> None:
            ok, detail = stop_owned_server()
            if not ok:
                raise RuntimeError(detail)
            self.messages.put(("message", detail + " Starting with saved settings…"))
            process = launch_server(settings)
            deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}. Check stderr logs.")
                if fetch_models(timeout=2) is not None:
                    self.messages.put(("message", "Restart complete; server is ready on port 8011."))
                    return
                time.sleep(2)
            raise TimeoutError("Restarted server did not become ready within 15 minutes.")

        self._run_background("Restart", restart)

    def reload_recent_activity(self) -> None:
        self._load_recent_logs()
        self.messages.put(("message", "Activity and server logs refreshed."))

    def open_logs_folder(self) -> None:
        try:
            os.startfile(RUN_DIR)
        except OSError as exc:
            messagebox.showerror("Could not open logs folder", str(exc), parent=self.root)

    def submit_prompt(self) -> None:
        if self.no_server_actions:
            return
        prompt = self.prompt_input.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showinfo("Prompt required", "Enter a prompt before sending it.", parent=self.root)
            return
        try:
            max_tokens = int(self.max_tokens_var.get())
            if not 1 <= max_tokens <= 8192:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid output limit", "Maximum output tokens must be from 1 to 8,192.", parent=self.root)
            return
        if self.last_models is None:
            messagebox.showerror("Server not ready", "Start the experimental server on port 8011 first.", parent=self.root)
            return
        self.notebook.select(self.prompt_tab)
        self._set_prompt_output("")
        self.prompt_rate_var.set("Generating…")
        self.send_button.configure(state="disabled")
        prompt_id = f"gui-{int(time.time() * 1000)}"
        preview = " ".join(prompt.split())[:180]
        self._insert_history(prompt_id, (datetime.now().strftime("%H:%M:%S"), "GUI prompt", preview,
                                         "—", "…", "—", "…"))

        def worker() -> None:
            try:
                result = send_prompt(prompt, max_tokens, lambda text: self.messages.put(("prompt-token", text)))
                self.messages.put(("prompt-finished", (prompt_id, result, preview)))
            except Exception as exc:
                self.messages.put(("prompt-finished", (prompt_id, {"error": str(exc)}, preview)))

        threading.Thread(target=worker, name="infernix-prompt", daemon=True).start()

    def _prompt_finished(self, outcome: tuple[str, dict[str, Any], str]) -> None:
        prompt_id, result, preview = outcome
        self.send_button.configure(state="normal")
        if "error" in result:
            self.prompt_rate_var.set("Prompt failed")
            self._append_prompt_output(f"\n\n[Request failed: {result['error']}]\n")
            self._append_log(f"[GUI prompt] Request failed: {result['error']}\n")
            return
        rate = result.get("tokens_per_second")
        tokens = result.get("completion_tokens")
        input_count = result.get("prompt_tokens")
        seconds = result.get("generation_seconds")
        self.prompt_rate_var.set(
            f"Observed output rate: {rate:.1f} tok/s" if isinstance(rate, (int, float))
            else "Observed output rate: usage unavailable"
        )
        item = (datetime.now().strftime("%H:%M:%S"), "GUI prompt", preview,
                f"{input_count:,}" if isinstance(input_count, int) else "—",
                f"{tokens:,}" if isinstance(tokens, int) else "—", "—", f"{rate:.1f}" if rate else "—")
        if self.history.exists(prompt_id):
            self.history.item(prompt_id, values=item)
        self._append_log(
            f"[GUI prompt] {tokens if tokens is not None else 'unknown'} output tokens; "
            f"{seconds:.2f}s generation; "
            f"{rate:.1f} observed output tok/s\n" if rate is not None and seconds is not None
            else "[GUI prompt] Completed; server did not return token usage.\n"
        )

    def _append_prompt_output(self, text: str) -> None:
        self.prompt_output.configure(state="normal")
        self.prompt_output.insert("end", text)
        self.prompt_output.see("end")
        self.prompt_output.configure(state="disabled")

    def _set_prompt_output(self, text: str) -> None:
        self.prompt_output.configure(state="normal")
        self.prompt_output.delete("1.0", "end")
        if text:
            self.prompt_output.insert("end", text)
        self.prompt_output.configure(state="disabled")

    def _read_form_settings(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(self.settings)
        for key, variable in self.setting_vars.items():
            data[key] = variable.get()
        return validate_settings(data)

    def save_settings(self) -> None:
        try:
            settings = self._read_form_settings()
            set_startup_registration(settings["launch_at_login"], settings["auto_load_qwen_at_login"])
            write_startup_configuration(settings)
        except (SettingsError, OSError, PermissionError) as exc:
            messagebox.showerror("Could not save settings", str(exc), parent=self.root)
            return
        self.settings = settings
        self.tray.panel_startup_enabled = settings["launch_at_login"]
        self.tray.auto_load_enabled = settings["auto_load_qwen_at_login"]
        self.settings_note.set(
            "Settings saved. CLI flags apply on next start/restart; Qwen auto-load takes effect at next sign-in."
        )

    def _toggle_startup(self) -> None:
        try:
            set_startup_registration(
                bool(self.startup_var.get()), bool(self.auto_load_var.get()),
            )
            self.settings["launch_at_login"] = bool(self.startup_var.get())
            self.tray.panel_startup_enabled = bool(self.startup_var.get())
        except OSError as exc:
            self.startup_var.set(bool(self.settings["launch_at_login"]))
            messagebox.showerror("Windows sign-in setting failed", str(exc), parent=self.root)

    def _apply_auto_load_setting(self, enabled: bool) -> None:
        self.auto_load_var.set(enabled)
        self._toggle_auto_load()

    def _toggle_auto_load(self) -> None:
        try:
            enabled = bool(self.auto_load_var.get())
            set_startup_registration(bool(self.startup_var.get()), enabled)
            self.settings["auto_load_qwen_at_login"] = enabled
            self.tray.auto_load_enabled = enabled
            self.settings_note.set(
                "Qwen will load into VRAM when you next sign in. Uncheck this option to disable auto-load."
                if enabled else "Qwen auto-load at Windows sign-in is disabled."
            )
        except OSError as exc:
            self.auto_load_var.set(bool(self.settings.get("auto_load_qwen_at_login", False)))
            messagebox.showerror("Windows sign-in setting failed", str(exc), parent=self.root)

    def restore_defaults(self) -> None:
        defaults = default_settings()
        defaults["launch_at_login"] = self.settings.get("launch_at_login", False)
        defaults["auto_load_qwen_at_login"] = self.settings.get("auto_load_qwen_at_login", False)
        for key, variable in self.setting_vars.items():
            variable.set(defaults[key])

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.state("normal")
        self.root.lift()
        self.root.focus_force()

    def hide_to_tray(self) -> None:
        if getattr(self.tray, "added", False):
            self.root.withdraw()
        else:
            self.root.iconify()

    def exit_gui(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.monitoring = False
        for timer_name in ("_monitor_after", "_drain_after", "_logs_after"):
            timer_id = getattr(self, timer_name, None)
            if timer_id is not None:
                try:
                    self.root.after_cancel(timer_id)
                except tk.TclError:
                    pass
        try:
            self.tray.remove()
        except (AttributeError, OSError):
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(argv: list[str] | None = None) -> int:
    global WINDOW_CLASS
    ensure_console_streams()
    if os.name != "nt":
        print("The native notification-area control panel is Windows-only.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", action="store_true", help="Start minimized to the Windows notification area.")
    parser.add_argument("--auto-load-qwen", action="store_true",
                        help="Load the experimental Qwen server after the tray panel starts.")
    parser.add_argument("--smoke-test-ui", action="store_true", help="Create the tray window briefly and exit without server actions.")
    options = parser.parse_args(argv)
    instance = None
    app = None
    try:
        if options.smoke_test_ui:
            # Isolate validation from a real panel; never forward auto-load to it.
            WINDOW_CLASS += f"_smoke_{os.getpid()}"
        else:
            instance = SingleInstance()
            if not instance.primary:
                if not instance.notify(show=not options.hidden, auto_load=options.auto_load_qwen):
                    raise RuntimeError("The existing Qwen panel did not respond within 10 seconds.")
                return 0
        app = NativeServerGUI(
            start_hidden=options.hidden or options.smoke_test_ui,
            no_server_actions=options.smoke_test_ui,
        )
        if options.smoke_test_ui:
            print(f"GUI window initialized; notification-area icon added={getattr(app.tray, 'added', False)}", flush=True)
            app.root.after(1200, app.exit_gui)
        elif options.auto_load_qwen:
            app.root.after(750, app.auto_load_server)
        app.run()
        return 0
    except Exception as exc:
        print(f"Infernix GUI startup failed: {exc}", file=sys.stderr, flush=True)
        if not options.smoke_test_ui:
            messagebox.showerror("Infernix GUI startup failed", str(exc))
        return 1
    finally:
        if app is not None and not app.closed:
            app.exit_gui()
        if instance is not None:
            instance.close()


if __name__ == "__main__":
    raise SystemExit(main())