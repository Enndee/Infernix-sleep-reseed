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
weighted aggregate decode rate. Prompt text is deliberately not written to the
native logs; requests sent from the GUI additionally show their prompt preview
and an observed streamed-output tok/s rate computed from response usage tokens
after the first text delta. Other API clients appear as server request IDs and
token counts without their private prompt contents.

The Settings tab exposes startup flags supported by this tested release: context,
VRAM/RAM headroom, KV type, and experimental idle scratch/weight/KV
sleep controls. KV offload defaults on in this panel; uncheck it to retain the
older KV-resident sleep tier. It takes effect at the next server start, not when
the checkbox is clicked. Restart an already open panel to load its new controls.
The API listener stays fixed to loopback port 8011.

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