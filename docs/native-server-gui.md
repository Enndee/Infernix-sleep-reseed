# Experimental native-server GUI (Windows)

`start_native_server_gui.bat` opens a small Windows control panel for the isolated
Qwen4Exp native server. Closing the window hides it in the notification area;
left-click the Qwen tray icon to reopen it, or right-click for the native menu.
Keyboard selection (Enter/Space) also opens the panel. Starting the launcher again
opens the existing panel instead of creating a second tray icon.
The GUI is implemented with the Python environment's Tkinter and Windows' built-in
notification-area API; it adds no Python package dependencies.

## Scope and safety

- The GUI targets only `127.0.0.1:8011` and the `infernix-serve.exe` under this
  experimental checkout. It does not start, reconfigure, or stop production ports
  8000/8001.
- Stop/restart is enabled only after checking the recorded PID, executable image,
  and exact `--port 8011` command line. A responding but unverified server is
  displayed as unverified and cannot be stopped or duplicated by the GUI.
- Generation is interrupted by stopping/restarting the process; in-memory
  conversation and prefix-cache state will not survive a process restart.
- VRAM headroom is a startup sizing target, not a hard cap. Weight sleep reserves
  an additional ~5.75 GiB of pinned host RAM. KV sleep uses additional pinned RAM
  proportional to the high-water mapped cache size and snapshots it afresh on
  each sleep; it does not discard the cached conversation. Sleep settings are experimental.
- Settings are applied at the next start/restart, not dynamically to a running
  engine. The configured context must fit the model and available system memory.
- **Exit control panel** removes the panel and tray icon only. It does not stop,
  sleep, unload, or otherwise change the experimental server.
- **Full unload Qwen (release VRAM)** asks for confirmation and terminates only
  the identity-verified experimental process. This releases that process's GPU
  allocations, not VRAM owned by other applications. The panel remains available.
- Windows sign-in options are independent: **Start control panel with Windows**
  opens a tray-only panel; **Automatically load Qwen with Windows** additionally
  requests model loading at sign-in. Auto-load is off by default, and changing
  it does not load Qwen immediately.
- The two options use only `InfernixNativeServerGUI` and
  `InfernixExperimentalQwenAutoLoad` under the current user's
  `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` key. Unchecking an option
  removes its entry, not another application's entry. A per-checkout/session mutex
  and native window messages coalesce simultaneous startup launches.

## Windows and rate reporting

The Activity tab tails native stdout/stderr, parses every completed request's
server-reported prompt/output token counts and prefill/decode tok/s, and shows a
weighted aggregate decode rate. Log lines are shown as the server wrote them,
without a stream filename prefix, and long lines wrap inside the view. The panel
and log use a dark palette. Prompt text is deliberately not written to the
native logs; requests sent from the GUI additionally show their prompt preview
and an observed streamed-output tok/s rate computed from response usage tokens
after the first text delta. Other API clients appear as server request IDs and
token counts without their private prompt contents.

The Settings tab exposes startup flags supported by this tested release: context,
VRAM/RAM headroom, KV type, maximum concurrency, a default thinking-token cap, and
experimental idle scratch/weight/KV sleep controls. Maximum concurrency defaults
to 1 and accepts 1-8 sequences; it is passed as `--max-concurrency`. Raising it
allows the server to schedule more sequences concurrently, but can increase
resource use. Start with 2 if you need two simultaneous generations. The default
thinking cap is 512 tokens; set it to `0` for Infernix's automatic answer-room
behavior instead. A positive cap is passed as `--default-thinking-budget` and
applies to thinking-enabled requests unless that request supplies its own budget.
All server settings take effect on the next server start/restart, not when a
control is changed. Restart an already open panel to load updated GUI controls.
The API listener stays fixed to loopback port 8011.

The cap is an Infernix server-side numeric limit, not a conversion of Copilot's
Thinking Effort selection. VS Code documents effort as controlling how much a
model reasons and says higher effort produces more thinking tokens, but gives no
fixed token-count mapping. For a Custom Endpoint model, `supportsReasoningEffort`
can expose supported labels in VS Code's picker, and `reasoningEffortFormat` can
forward them as the Chat Completions `reasoning_effort` field. Infernix accepts
`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max` for this field;
these remain effort hints, not numeric caps.

The Custom Endpoint provider also supports `modelOptions`, an object of request
parameters sent with every request. This can set fixed fields such as
`reasoning_effort`, but VS Code does not document a dynamic per-request numeric
thinking-budget setting. Infernix's OpenAI Chat Completions parser does not map
an arbitrary numeric budget field to the thinking cap. Its Anthropic Messages
API accepts `thinking.budget_tokens` per request, but requires at least 1,024
tokens. Use this GUI's 512-token server default when an exact 512 cap is needed;
an explicit per-request budget on a supported API can take precedence.

Focused checks:

```powershell
& 'F:\Models\infernix-env\python.exe' 'D:\KI\Local_LLM\Infernix_Qwen_3.8_Flash_Next_abliterated_nvfp4\experimental\Infernix\tools\test_native_server_gui.py'
& 'F:\Models\infernix-env\python.exe' 'D:\KI\Local_LLM\Infernix_Qwen_3.8_Flash_Next_abliterated_nvfp4\experimental\Infernix\tools\native_server_gui.py' --smoke-test-ui
& 'F:\Models\infernix-env\python.exe' 'D:\KI\Local_LLM\Infernix_Qwen_3.8_Flash_Next_abliterated_nvfp4\experimental\Infernix\tools\smoke_native_server_gui.py'
```

The standalone smoke script creates real Win32/Tk windows and a notification icon,
posts native mouse/keyboard/context-menu callbacks, verifies hide/reopen, cancels
the native menu, checks Explorer re-registration and Exit cleanup, and exercises
missing-icon/failed-tray fallbacks. All server and registry-write entry points are
replaced with assertions during that script. The CLI smoke flag also blocks all
server actions and takes precedence over `--auto-load-qwen`.

## Tray implementation constraint

The tray owns a hidden top-level Win32 window, not Tk's window procedure. Its
`ctypes` callback **must never call Tk**, including `winfo_exists()` or `after()`.
On this Windows/Python 3.12 environment, doing that reproduces a fatal
`PyEval_RestoreThread` error on the first delivered tray selection. The callback
only queues Python callables; an independently scheduled Tk timer drains them.
Handle APIs have explicit pointer-sized signatures. Loaded Qwen icon handles are
destroyed on exit; the shared Windows fallback icon is not. If tray setup fails,
`--hidden` leaves the panel visible rather than stranding an invisible process.

Validation on 26 October 2026: 28 focused unit tests, compilation, CLI smoke with
both smoke/auto-load flags, and the native-event smoke passed. Actual Qwen loading
and termination were deliberately not performed as part of the tray repair.

## Standalone executable build

`packaging/windows/build_panel_exe.py` freezes the panel with PyInstaller so it
starts without the development Python environment. The frozen panel is not a
portable server: it still drives exactly one local checkout.

```powershell
# PyInstaller is a build-time dependency of the environment Python only.
& 'F:\Models\infernix-env\python.exe' -m pip install pyinstaller
& 'F:\Models\infernix-env\python.exe' '<checkout>\packaging\windows\build_panel_exe.py'
& '<checkout>\dist\InfernixPanel-windows-x64\InfernixPanel\InfernixPanel.exe' --smoke-test-ui
```

- Output: `dist\InfernixPanel-windows-x64\InfernixPanel\` containing
  `InfernixPanel.exe`, `_internal\`, `repo-root.txt`, `README.txt` and
  `SHA256SUMS.txt`. `--onefile` produces a single executable instead, and
  `--console` keeps diagnostic output visible.
- Double-click launches work: a desktop shortcut is a plain `.lnk` whose target is
  `InfernixPanel.exe`, using the executable as icon source (it embeds
  `assets\qwen.ico`). With no arguments the panel opens its window, `--hidden`
  starts it minimized to the notification area. Recreate the shortcut if the
  `dist\` folder is rebuilt elsewhere.
- The build adds `tcl86t.dll` and `tk86t.dll` from the interpreter's `Library\bin`
  explicitly. A conda `_tkinter.pyd` loads them from there, a directory
  PyInstaller's binary analysis does not search, so an unpatched frozen panel
  aborts with `ImportError: DLL load failed while importing _tkinter` (windowed
  builds show that as a modal dialog and appear to hang).
- `repo-root.txt` next to the executable, or the `INFERNIX_REPO` environment
  variable, tells the frozen panel which checkout holds `build-baseline\native-run`
  and `build-baseline\apps\Release\infernix-serve.exe`. Keep the file current if
  either path moves.
- A windowed build has no console handles. Set `INFERNIX_PANEL_LOG` to a file to
  capture the panel's stdout/stderr.
- Sign-in registration writes `InfernixPanel.exe --hidden` for a frozen panel and
  `pythonw.exe "<script>" --hidden` for the source panel, both under the same
  per-user `Run` value names. Use only one of the two launchers with the
  start-with-Windows options enabled.
- Rebuild after changing `tools/native_server_gui.py`; the executable contains a
  frozen copy. The panel and the server are separate: `infernix-serve.exe` still
  comes from `build-baseline\apps\Release`.
