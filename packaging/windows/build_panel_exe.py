"""Build the Windows tray control panel as a standalone executable.

The panel is plain Tkinter plus Win32 ``ctypes`` calls, so PyInstaller can ship it
without the development Python environment. The executable still drives one local
checkout: it reads ``repo-root.txt`` next to itself (or ``INFERNIX_REPO``) to find
``build-baseline\\native-run`` and ``build-baseline\\apps\\Release\\infernix-serve.exe``.

Usage, from the checkout root:

    & 'F:\\Models\\infernix-env\\python.exe' packaging\\windows\\build_panel_exe.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
GUI_SCRIPT = REPO / "tools" / "native_server_gui.py"
ASSETS = REPO / "tools" / "assets"
ICON = ASSETS / "qwen.ico"
SERVER_EXE = REPO / "build-baseline" / "apps" / "Release" / "infernix-serve.exe"
APP_NAME = "InfernixPanel"
DEFAULT_OUTPUT = REPO / "dist" / f"{APP_NAME}-windows-x64"

README_TEXT = """\
Infernix tray control panel (standalone Windows build)
=====================================================

InfernixPanel.exe is a frozen copy of tools/native_server_gui.py. It needs no
Python installation, but it is NOT a portable server: it drives one local
Infernix checkout.

Requirements
------------
1. A checkout of experimental/Infernix on this machine.
2. A built server inside that checkout:
   build-baseline\\apps\\Release\\infernix-serve.exe
3. The Qwen NVFP4 artifact and n-gram file the panel references in its own
   module constants (F:\\Models\\... by default).
4. The NVIDIA driver plus the CUDA runtime the server was built against.

Pointing the panel at a checkout
--------------------------------
The panel resolves its checkout in this order:
  1. the INFERNIX_REPO environment variable,
  2. the path recorded in repo-root.txt next to InfernixPanel.exe,
  3. the directory holding InfernixPanel.exe.
After moving either the checkout or this folder, edit repo-root.txt (one path
per file, absolute).

Behaviour
---------
* Listens for the panel itself? No. It talks only to 127.0.0.1:8011 and refuses
  to stop a server it cannot verify by PID, executable image and exact port.
* Settings and the native-run PID/latest logs stay inside the checkout under
  build-baseline\\native-run, so the frozen panel and the source panel share the
  same state. Remove any running panel before rebuilding.
* Closing the window hides the panel in the notification area; Exit removes the
  panel and tray icon only and leaves the server running.
* Start-with-Windows entries are per user and, in this frozen build, point
  directly at InfernixPanel.exe instead of pythonw.exe plus a script path.

Diagnostics
-----------
A windowed build has no console, so set INFERNIX_PANEL_LOG to a file path before
launching to capture the panel's stdout/stderr. Example:

    set INFERNIX_PANEL_LOG=%TEMP%\\infernix-panel.log
    InfernixPanel.exe --smoke-test-ui

Validation shortcuts
--------------------
    InfernixPanel.exe --smoke-test-ui    create the tray window, then exit
    InfernixPanel.exe --hidden           start in the notification area
    InfernixPanel.exe --auto-load-qwen   also request a Qwen load after start

SHA256SUMS.txt in this folder covers every shipped file except itself.
"""


def find_tk_runtime() -> list[Path]:
    """Locate the Tcl/Tk runtime DLLs the interpreter's ``_tkinter.pyd`` loads.

    A conda ``_tkinter.pyd`` resolves ``tcl86t.dll``/``tk86t.dll`` from
    ``Library\\bin``, a directory PyInstaller's binary analysis does not search,
    so the resolved ``_tkinter`` import fails inside the bundle unless these
    DLLs are added explicitly.
    """
    roots = [Path(sys.base_prefix), Path(sys.prefix), Path(sys.executable).parent]
    search: list[Path] = []
    for root in roots:
        search += [root, root / "DLLs", root / "Library" / "bin", root / "bin"]
    found: list[Path] = []
    for name in ("tcl86t.dll", "tk86t.dll", "tcl86.dll", "tk86.dll"):
        for folder in search:
            candidate = folder / name
            if candidate.is_file():
                found.append(candidate)
                break
    return found


def build_arguments(output: Path, console: bool, onefile: bool) -> list[str]:
    """PyInstaller command line for the panel."""
    work = REPO / "build-baseline" / "panel-work"
    arguments = [
        "--noconfirm",
        "--clean",
        "--log-level",
        "WARN",
        "--name",
        APP_NAME,
        "--specpath",
        str(work),
        "--workpath",
        str(work),
        "--distpath",
        str(output),
        "--add-data",
        f"{ASSETS};assets",
    ]
    for dll in find_tk_runtime():
        arguments += ["--add-binary", f"{dll};."]
    arguments.append("--console" if console else "--windowed")
    if onefile:
        arguments.append("--onefile")
    if ICON.is_file():
        arguments += ["--icon", str(ICON)]
    arguments.append(str(GUI_SCRIPT))
    return arguments


def write_deployment_files(output: Path, onefile: bool) -> Path:
    """Add the checkout pointer and user README beside the built executable."""
    folder = output if onefile else output / APP_NAME
    (folder / "repo-root.txt").write_text(str(REPO) + "\n", encoding="utf-8")
    (folder / "README.txt").write_text(README_TEXT, encoding="utf-8")
    return folder


def write_checksums(folder: Path) -> Path:
    """Write SHA256SUMS.txt for every file in the built folder."""
    lines = []
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(folder).as_posix()}")
    target = folder / "SHA256SUMS.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Destination folder (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--console", action="store_true",
                        help="Keep a console window; useful for diagnosing a failed start.")
    parser.add_argument("--onefile", action="store_true",
                        help="Pack the panel into a single executable instead of a folder.")
    options = parser.parse_args(argv)

    for requirement in (GUI_SCRIPT, ICON, ASSETS):
        if not requirement.exists():
            print(f"missing required path: {requirement}", file=sys.stderr)
            return 2
    try:
        import PyInstaller  # noqa: F401  (import check only)
    except ModuleNotFoundError:
        print("PyInstaller is not installed in this interpreter.", file=sys.stderr)
        print(f'  & "{sys.executable}" -m pip install pyinstaller', file=sys.stderr)
        return 3

    output = options.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    tk_runtime = find_tk_runtime()
    if not tk_runtime:
        print("warning: no Tcl/Tk runtime DLLs found; the frozen panel will fail to import tkinter.",
              file=sys.stderr)
    server_present = SERVER_EXE.is_file()

    arguments = build_arguments(output, options.console, options.onefile)
    print(json.dumps({"output": str(output), "arguments": arguments}, indent=2), flush=True)
    completed = subprocess.run([sys.executable, "-m", "PyInstaller", *arguments], check=False)
    if completed.returncode != 0:
        print(f"PyInstaller failed with exit code {completed.returncode}", file=sys.stderr)
        return completed.returncode

    folder = write_deployment_files(output, options.onefile)
    executable = folder / f"{APP_NAME}.exe"
    if not executable.is_file():
        print(f"expected executable missing: {executable}", file=sys.stderr)
        return 4
    checksums = write_checksums(folder)
    files = [path for path in folder.rglob("*") if path.is_file()]
    report = {
        "app": APP_NAME,
        "executable": str(executable),
        "exe_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "onefile": options.onefile,
        "console": options.console,
        "repo_root": str(REPO),
        "server_exe": str(SERVER_EXE),
        "server_exe_present": server_present,
        "tk_runtime": [str(path) for path in tk_runtime],
        "checksums": str(checksums),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

