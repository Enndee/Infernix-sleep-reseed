"""Exercise real Win32/Tk tray callbacks, with every server entry point forbidden."""
from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch

import native_server_gui as gui


def main() -> int:
    gui.WINDOW_CLASS += f"_validation_{os.getpid()}"
    results = []
    failures = []
    app = None
    primary = None
    secondary = None

    def forbidden(*args, **kwargs):
        raise AssertionError("A smoke test attempted a server or startup-registry action")

    try:
        with ExitStack() as stack:
            for name in ('launch_server', 'stop_owned_server', 'send_prompt', 'fetch_models',
                         'inspect_owned_server', '_port_is_open', 'set_startup_registration'):
                stack.enter_context(patch.object(gui, name, side_effect=forbidden))
            primary = gui.SingleInstance()
            secondary = gui.SingleInstance()
            assert primary.primary and not secondary.primary
            results.append('single-instance mutex')
            app = gui.NativeServerGUI(start_hidden=True, no_server_actions=True)
            app.root.report_callback_exception = lambda kind, value, trace: failures.append(str(value))
            assert app.tray.added, app.tray.error
            assert app.tray._owns_icon, 'Qwen icon did not load'
            results.append('Qwen ICO loaded; Shell_NotifyIcon add succeeded')
            user = app.tray.user32
            user.IsWindow.argtypes = [ctypes.c_void_p]
            user.IsWindow.restype = ctypes.c_int

            def pump():
                for _ in range(6):
                    app.root.update()
                    time.sleep(0.02)
                assert not failures, failures

            for event in (app.tray.WM_LBUTTONUP, app.tray.NIN_SELECT, app.tray.NIN_KEYSELECT):
                app.hide_to_tray()
                pump()
                assert app.root.state() == 'withdrawn'
                assert user.PostMessageW(app.tray.hwnd, app.tray.WM_TRAY, 0, (1 << 16) | event)
                pump()
                assert app.root.state() == 'normal'
                assert app.tray.added and user.IsWindow(app.tray.hwnd)
            results.append('native mouse/selection/keyboard callbacks reopen hidden Tk panel')

            assert secondary.notify(show=True, auto_load=True)
            pump()  # auto-load callback is blocked in no-server mode.
            results.append('secondary instance forwards safely to the primary')

            # Post cancellation to this test's hidden window, never another application's menu.
            def cancel_menu():
                time.sleep(0.4)
                user.PostMessageW(app.tray.hwnd, 0x001F, 0, 0)  # WM_CANCELMODE

            original_track = user.TrackPopupMenu
            menu_opened = []

            def track(*args):
                menu_opened.append(True)
                timer = threading.Thread(target=cancel_menu, daemon=True)
                timer.start()
                result = original_track(*args)
                timer.join(timeout=2)
                return result

            with patch.object(user, 'TrackPopupMenu', side_effect=track):
                user.PostMessageW(app.tray.hwnd, app.tray.WM_TRAY, 0, (1 << 16) | app.tray.WM_CONTEXTMENU)
                pump()
            assert menu_opened and not failures
            results.append('native context menu opens and cancels without disappearing panel')

            # Simulate Explorer restart without restarting Explorer itself.
            app.tray.shell32.Shell_NotifyIconW(app.tray.NIM_DELETE, ctypes.byref(app.tray._icon_data))
            user.PostMessageW(app.tray.hwnd, app.tray._taskbar_created, 0, 0)
            pump()
            assert app.tray.added
            results.append('Explorer TaskbarCreated re-add')
            hwnd = app.tray.hwnd
            app.tray._dispatch_menu(app.tray.MENU_EXIT)
            app.root.update()
            assert app.closed and not app.tray.added and app.tray.hwnd is None
            assert not user.IsWindow(hwnd)
            results.append('Exit removes icon and window; no server actions')
            # Missing icon -> shared Windows fallback; failed tray -> visible Tk panel.
            gui.WINDOW_CLASS += '_fallback'
            with patch.object(gui, 'QWEN_ICON', Path('missing-smoke-test-icon.ico')):
                app = gui.NativeServerGUI(start_hidden=True, no_server_actions=True)
                assert app.tray.added and not app.tray._owns_icon
                app.exit_gui()
            gui.WINDOW_CLASS += '_unavailable'
            with patch.object(gui.TrayIcon, '_initialize', side_effect=OSError('simulated tray failure')):
                app = gui.NativeServerGUI(start_hidden=True, no_server_actions=True)
                app.root.update()
                assert not app.tray.added and app.root.state() == 'normal'
                app.exit_gui()
            results.append('missing-icon fallback and failed-tray visible-panel fallback')
        print(json.dumps({'ok': True, 'checks': results}, indent=2), flush=True)
        return 0
    finally:
        if app is not None and not app.closed:
            app.exit_gui()
        if secondary is not None:
            secondary.close()
        if primary is not None:
            primary.close()


if __name__ == '__main__':
    raise SystemExit(main())