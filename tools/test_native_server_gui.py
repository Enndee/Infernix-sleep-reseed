"""Focused tests for the isolated native-server tray GUI's settings and log metrics."""

from __future__ import annotations

import importlib.util
import ctypes
import json
import queue
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


MODULE_PATH = Path(__file__).with_name("native_server_gui.py")
SPEC = importlib.util.spec_from_file_location("native_server_gui", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gui)


class SettingsTests(unittest.TestCase):
    def test_defaults_match_experimental_launcher(self) -> None:
        with patch.object(gui, "RUNNER_CONFIG", Path("missing-test-runner-config.json")):
            settings = gui.default_settings()
        self.assertEqual(settings["max_context"], 262144)
        self.assertEqual(settings["vram_headroom_mib"], 12288)
        self.assertEqual(settings["ram_headroom_mib"], 24576)
        self.assertEqual(settings["kv_dtype"], "int8")
        self.assertEqual(settings["sleep_after_seconds"], 15)
        self.assertTrue(settings["sleep_weights"])
        self.assertTrue(settings["sleep_kv"])

    def test_supported_settings_build_fixed_experimental_command(self) -> None:
        settings = gui.default_settings()
        settings.update(max_context=131072, vram_headroom_mib=8192, kv_dtype="fp8")
        arguments = gui.build_server_arguments(settings)
        self.assertEqual(arguments[arguments.index("--port") + 1], "8011")
        self.assertEqual(arguments[arguments.index("--max-context") + 1], "131072")
        self.assertEqual(arguments[arguments.index("--vram-headroom-mib") + 1], "8192")
        self.assertEqual(arguments[arguments.index("--kv-dtype") + 1], "fp8")
        self.assertEqual(arguments[arguments.index("--expert-state") + 1], str(gui.EXPERT_STATE))

    def test_rejects_sleep_backing_without_idle_sleep(self) -> None:
        settings = gui.default_settings()
        settings.update(sleep_after_seconds=0, sleep_scratch=False, sleep_kv=False, sleep_weights=True)
        self.assertEqual(gui.validate_settings(settings)["sleep_weights"], True)
        settings["sleep_scratch"] = True
        with self.assertRaises(gui.SettingsError):
            gui.validate_settings(settings)

    def test_kv_sleep_environment_and_legacy_settings(self) -> None:
        settings = gui.default_settings()
        self.assertEqual(gui._child_environment(settings)["INFERNIX_SLEEP_KV"], "1")
        settings["sleep_kv"] = False
        self.assertEqual(gui._child_environment(settings)["INFERNIX_SLEEP_KV"], "0")
        del settings["sleep_kv"]
        self.assertTrue(gui.validate_settings(settings)["sleep_kv"])
        settings.update(sleep_after_seconds=0, sleep_scratch=False, sleep_kv=True)
        with self.assertRaises(gui.SettingsError):
            gui.validate_settings(settings)

    def test_rejects_unsupported_kv_type_and_out_of_range_context(self) -> None:
        settings = gui.default_settings()
        settings["kv_dtype"] = "unsupported"
        with self.assertRaises(gui.SettingsError):
            gui.validate_settings(settings)
        settings = gui.default_settings()
        settings["max_context"] = 262145
        with self.assertRaises(gui.SettingsError):
            gui.validate_settings(settings)


class ServerLogTests(unittest.TestCase):
    def test_parses_completed_request_rate_and_token_counts(self) -> None:
        line = (
            "2026-10-25 21:45:23.766  INFO  req#13 done | openai-chat | stop token | "
            "prompt 5,162 | output 11 | cache 5,120 (99.2%, hybrid_snapshot) | TTFT 1.1s | "
            "total 1.2s | queue 778 ms | prefill 138.3 tok/s | decode 129.1 tok/s | mtp accepted 8/8"
        )
        result = gui.parse_server_request(line)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "13")
        self.assertEqual(result["prompt"], 5162)
        self.assertEqual(result["output"], 11)
        self.assertAlmostEqual(result["prefill"], 138.3)
        self.assertAlmostEqual(result["decode"], 129.1)
        self.assertEqual(result["time"], "21:45:23")
        self.assertIsNone(gui.parse_server_request("ordinary model startup log"))

    def test_parses_millisecond_time_to_first_token(self) -> None:
        line = (
            "2026-10-25 21:46:19.272  INFO  req#17 done | openai-chat | stop token | "
            "prompt 12 | output 7 | cache 0 (0.0%) | TTFT 781 ms | total 1.2s | "
            "queue 3 ms | prefill 18.2 tok/s | decode 6.1 tok/s"
        )
        result = gui.parse_server_request(line)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["id"], "17")
        self.assertEqual(result["output"], 7)


class StreamingMetricTests(unittest.TestCase):
    def test_uses_final_stream_usage_and_excludes_time_to_first_token(self) -> None:
        class FakeResponse:
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}'
                yield b'data: {"choices":[{"delta":{"content":" world"}}]}'
                yield b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":8}}'
                yield b"data: [DONE]"

        parts: list[str] = []
        result = gui.parse_streaming_completion(FakeResponse(), parts.append)
        self.assertEqual("".join(parts), "hello world")
        self.assertEqual(result["completion_tokens"], 8)
        self.assertEqual(result["prompt_tokens"], 12)
        self.assertGreater(result["tokens_per_second"], 0)
        self.assertIsNotNone(result["generation_seconds"])


class StartupTests(unittest.TestCase):
    def setUp(self) -> None:
        import winreg
        self.registry = winreg
        self.values = {"UnrelatedApp": "must remain unchanged"}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        key = MagicMock()
        key.__enter__.return_value = key
        self.stack.enter_context(patch.object(winreg, "CreateKey", return_value=key))
        self.stack.enter_context(patch.object(winreg, "OpenKey", return_value=key))

        def query(_key, name):
            if name not in self.values:
                raise FileNotFoundError(name)
            return self.values[name], winreg.REG_SZ

        def delete(_key, name):
            if name not in self.values:
                raise FileNotFoundError(name)
            del self.values[name]

        self.stack.enter_context(patch.object(winreg, "QueryValueEx", side_effect=query))
        self.stack.enter_context(patch.object(winreg, "DeleteValue", side_effect=delete))
        self.stack.enter_context(patch.object(winreg, "SetValueEx", side_effect=lambda k, n, r, t, v: self.values.__setitem__(n, v)))

    def test_all_independent_startup_combinations_round_trip(self) -> None:
        for panel, auto_load in ((False, False), (True, False), (False, True), (True, True), (False, False)):
            with self.subTest(panel=panel, auto_load=auto_load):
                gui.set_startup_registration(panel, auto_load)
                self.assertEqual(gui.registered_login_options(), (panel, auto_load))
                self.assertEqual(self.values["UnrelatedApp"], "must remain unchanged")
                if panel:
                    command = self.values[gui.APP_NAME]
                    self.assertIn('--hidden', command)
                    self.assertNotIn('--auto-load-qwen', command)
                    self.assertIn(f'"{MODULE_PATH.resolve()}"', command)
                if auto_load:
                    command = self.values[gui.QWEN_AUTO_LOAD_APP_NAME]
                    self.assertIn('--hidden --auto-load-qwen', command)

    def test_auto_load_is_off_by_default(self) -> None:
        self.assertFalse(gui.default_settings()["auto_load_qwen_at_login"])

    def test_settings_persist_to_temporary_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(gui, "RUN_DIR", directory), patch.object(gui, "RUNNER_CONFIG", directory / "config.json"), \
                    patch.object(gui, "GUI_SETTINGS", directory / "gui-settings.json"):
                settings = gui.default_settings()
                settings.update(launch_at_login=True, auto_load_qwen_at_login=True)
                gui.write_startup_configuration(settings)
                self.assertEqual(gui.load_settings(), settings)
                self.assertEqual(json.loads(gui.RUNNER_CONFIG.read_text())["env"]["INFERNIX_SLEEP_WEIGHTS"], "1")

    def test_failed_startup_toggles_restore_previous_values(self) -> None:
        app = gui.NativeServerGUI.__new__(gui.NativeServerGUI)
        app.root = MagicMock()
        app.settings = {"launch_at_login": False, "auto_load_qwen_at_login": False}
        app.startup_var = MagicMock()
        app.auto_load_var = MagicMock()
        app.startup_var.get.return_value = True
        app.auto_load_var.get.return_value = True
        app.tray = SimpleNamespace(panel_startup_enabled=False, auto_load_enabled=False)
        with patch.object(gui, "set_startup_registration", side_effect=PermissionError("denied")), \
                patch.object(gui.messagebox, "showerror"):
            app._toggle_startup()
            app._toggle_auto_load()
        app.startup_var.set.assert_called_with(False)
        app.auto_load_var.set.assert_called_with(False)
        self.assertFalse(app.tray.panel_startup_enabled)
        self.assertFalse(app.tray.auto_load_enabled)


class TrayCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tray = gui.TrayIcon.__new__(gui.TrayIcon)
        self.tray.root = MagicMock()
        self.tray._removed = False
        self.tray._pending = queue.Queue()
        self.tray._taskbar_created = 0xC123
        self.tray.callbacks = {command: MagicMock() for command in (
            gui.TrayIcon.MENU_OPEN, gui.TrayIcon.MENU_START, gui.TrayIcon.MENU_FULL_UNLOAD, gui.TrayIcon.MENU_EXIT,
        )}
        self.tray.on_auto_load = MagicMock()
        self.tray.on_auto_load_changed = MagicMock()
        self.tray.on_panel_startup_changed = MagicMock()
        self.tray.auto_load_enabled = False
        self.tray.panel_startup_enabled = False
        self.tray._show_menu = MagicMock()
        self.tray._add_icon = MagicMock()
        self.tray.user32 = MagicMock()
        self.tray.user32.DefWindowProcW.return_value = 42

    def test_mouse_and_keyboard_selection_open_without_exit(self) -> None:
        for event in (self.tray.WM_LBUTTONUP, self.tray.WM_LBUTTONDBLCLK, self.tray.NIN_SELECT, self.tray.NIN_KEYSELECT):
            self.assertEqual(self.tray._window_proc(1, self.tray.WM_TRAY, 0, (1 << 16) | event), 0)
        self.tray.root.after.assert_not_called()
        self.tray.callbacks[self.tray.MENU_OPEN].assert_not_called()
        self.tray._drain_pending()
        self.assertEqual(self.tray.callbacks[self.tray.MENU_OPEN].call_count, 4)
        self.tray.callbacks[self.tray.MENU_EXIT].assert_not_called()

    def test_context_callbacks_open_native_menu(self) -> None:
        for event in (self.tray.WM_RBUTTONUP, self.tray.WM_CONTEXTMENU):
            self.tray._window_proc(1, self.tray.WM_TRAY, 0, event)
        self.tray._drain_pending()
        self.assertEqual(self.tray._show_menu.call_count, 2)

    def test_full_unload_and_exit_are_distinct_menu_actions(self) -> None:
        self.tray._dispatch_menu(self.tray.MENU_EXIT)
        self.tray._drain_pending()
        self.tray.callbacks[self.tray.MENU_EXIT].assert_called_once_with()
        self.tray.callbacks[self.tray.MENU_FULL_UNLOAD].assert_not_called()
        self.tray._dispatch_menu(self.tray.MENU_FULL_UNLOAD)
        self.tray._drain_pending()
        self.tray.callbacks[self.tray.MENU_FULL_UNLOAD].assert_called_once_with()

    def test_toggle_is_not_optimistic_before_registration_succeeds(self) -> None:
        self.tray._dispatch_menu(self.tray.MENU_AUTO_LOAD)
        self.tray._drain_pending()
        self.tray.on_auto_load_changed.assert_called_once_with(True)
        self.assertFalse(self.tray.auto_load_enabled)

    def test_existing_instance_messages_are_deferred(self) -> None:
        self.tray._window_proc(1, self.tray.WM_INSTANCE, 1, 1)
        self.tray._drain_pending()
        self.tray.callbacks[self.tray.MENU_OPEN].assert_called_once_with()
        self.tray.on_auto_load.assert_called_once_with()

    def test_explorer_restart_readds_icon(self) -> None:
        self.tray.added = True
        self.tray._window_proc(1, self.tray._taskbar_created, 0, 0)
        self.tray._drain_pending()
        self.tray._add_icon.assert_called_once_with()

    def test_remove_is_idempotent_and_does_not_destroy_shared_icons(self) -> None:
        for owned in (False, True):
            tray = gui.TrayIcon.__new__(gui.TrayIcon)
            tray.available = tray.added = tray._class_registered = True
            tray._drain_after = None
            tray.hwnd = 0x123456789
            tray.instance = 0x123456780
            tray._class_name = "test"
            tray._icon_data = tray.NotifyIconData()
            tray._icon_handle = 0x234567891
            tray._owns_icon = owned
            tray.user32 = MagicMock()
            tray.shell32 = MagicMock()
            tray.remove()
            tray.remove()
            tray.user32.DestroyWindow.assert_called_once()
            self.assertEqual(tray.user32.DestroyIcon.call_count, int(owned))
            tray.shell32.Shell_NotifyIconW.assert_called_once()

    def test_initialization_failure_returns_fully_initialized_fallback(self) -> None:
        root = MagicMock()
        with patch.object(gui.TrayIcon, '_initialize', side_effect=OSError('tray unavailable')):
            tray = gui.TrayIcon(root, *(MagicMock() for _ in range(5)), False, False, MagicMock())
        self.assertFalse(tray.added)
        self.assertIn('tray unavailable', tray.error)
        tray.remove()


class ActionSafetyTests(unittest.TestCase):
    def test_full_unload_requires_confirmation(self) -> None:
        app = gui.NativeServerGUI.__new__(gui.NativeServerGUI)
        app.no_server_actions = False
        app.root = MagicMock()
        app._stop_verified_server = MagicMock()
        with patch.object(gui.messagebox, "askyesno", return_value=False):
            app.full_unload_server()
        app._stop_verified_server.assert_not_called()
        with patch.object(gui.messagebox, "askyesno", return_value=True):
            app.full_unload_server()
        app._stop_verified_server.assert_called_once_with("Full unload")

    def test_exit_only_removes_panel(self) -> None:
        app = gui.NativeServerGUI.__new__(gui.NativeServerGUI)
        app.root = MagicMock()
        app.tray = MagicMock()
        app.closed = False
        with patch.object(gui, "stop_owned_server") as stop, patch.object(gui, "launch_server") as start:
            app.exit_gui()
            app.exit_gui()
        stop.assert_not_called()
        start.assert_not_called()
        app.root.destroy.assert_called_once()
        app.tray.remove.assert_called_once()

    def test_no_server_mode_blocks_every_action(self) -> None:
        app = gui.NativeServerGUI.__new__(gui.NativeServerGUI)
        app.no_server_actions = True
        with patch.object(gui, "launch_server") as start, patch.object(gui, "stop_owned_server") as stop, \
                patch.object(gui, "send_prompt") as send, patch.object(gui.messagebox, "askyesno") as confirm:
            for method in (app.auto_load_server, app.start_server, app.stop_server,
                           app.full_unload_server, app.restart_server, app.submit_prompt):
                method()
        for function in (start, stop, send, confirm):
            function.assert_not_called()

    def test_exact_port_parser_rejects_production_or_ambiguous_commands(self) -> None:
        prefix = f'"{gui.SERVER_EXE}" "model with spaces.infernix" '
        for options in ('--port 8011', '--port "8011"', '--port=8011'):
            self.assertTrue(gui.is_experimental_command(prefix + options))
        for options in ('--port 8000', '--port 8001', '--port 80110', '--port 8011 --port 8001',
                        '--port 8000 --port=8011', '--port', ''):
            self.assertFalse(gui.is_experimental_command(prefix + options))

    def test_unload_rejects_foreign_image_or_port_without_termination(self) -> None:
        for image, command in ((r'C:\other\infernix-serve.exe', '--port 8011'),
                               (str(gui.SERVER_EXE), '--port 8001'),
                               (str(gui.SERVER_EXE), '--port 8011 --port 8000')):
            kernel = MagicMock()
            kernel.OpenProcess.return_value = 0x123456789

            def query(handle, flags, buffer, size):
                buffer.value = image
                return True

            kernel.QueryFullProcessImageNameW.side_effect = query
            with patch.object(gui, '_kernel32', return_value=kernel), \
                    patch.object(gui.Path, 'read_text', return_value='12345'), \
                    patch.object(gui, 'process_command_line', return_value={'CommandLine': command}):
                ok, _ = gui.stop_owned_server()
            self.assertFalse(ok)
            kernel.TerminateProcess.assert_not_called()
            kernel.CloseHandle.assert_called_once()

    def test_auto_load_does_not_duplicate_existing_listener(self) -> None:
        app = gui.NativeServerGUI.__new__(gui.NativeServerGUI)
        app.no_server_actions = app.busy = False
        app.start_server = MagicMock()
        with patch.object(gui.Path, 'read_text', side_effect=FileNotFoundError), \
                patch.object(gui, '_port_is_open', return_value=True):
            app.auto_load_server()
        app.start_server.assert_not_called()

    def test_verified_unload_uses_one_handle_and_confirms_exit(self) -> None:
        for outcome, expected in ((gui.WAIT_OBJECT_0, True), (gui.WAIT_TIMEOUT, False), (0xFFFFFFFF, False)):
            kernel = MagicMock()
            handle = 0x123456789
            kernel.OpenProcess.return_value = handle
            kernel.TerminateProcess.return_value = True
            kernel.WaitForSingleObject.return_value = outcome

            def query(process, flags, buffer, size):
                self.assertEqual(process, handle)
                buffer.value = str(gui.SERVER_EXE)
                return True

            kernel.QueryFullProcessImageNameW.side_effect = query
            with patch.object(gui, '_kernel32', return_value=kernel), \
                    patch.object(gui.Path, 'read_text', return_value='12345'), \
                    patch.object(gui.Path, 'unlink') as unlink, \
                    patch.object(gui, 'process_command_line', return_value={'CommandLine': 'infernix-serve.exe --port 8011'}):
                ok, _ = gui.stop_owned_server()
            self.assertEqual(ok, expected)
            kernel.TerminateProcess.assert_called_once_with(handle, 1)
            kernel.WaitForSingleObject.assert_called_once_with(handle, 20_000)
            kernel.CloseHandle.assert_called_once_with(handle)
            self.assertEqual(unlink.call_count, int(expected))


class MainTests(unittest.TestCase):
    def test_smoke_flag_overrides_auto_load_and_single_instance(self) -> None:
        app = MagicMock()
        app.closed = False
        with patch.object(gui, 'NativeServerGUI', return_value=app) as factory, \
                patch.object(gui, 'SingleInstance') as guard, patch.object(gui, 'WINDOW_CLASS', 'test-smoke'):
            self.assertEqual(gui.main(['--smoke-test-ui', '--auto-load-qwen']), 0)
        factory.assert_called_once_with(start_hidden=True, no_server_actions=True)
        guard.assert_not_called()
        scheduled = [call.args[1] for call in app.root.after.call_args_list]
        self.assertEqual(scheduled, [app.exit_gui])

    def test_secondary_launch_only_forwards_to_primary(self) -> None:
        instance = MagicMock()
        instance.primary = False
        instance.notify.return_value = True
        with patch.object(gui, 'SingleInstance', return_value=instance), \
                patch.object(gui, 'NativeServerGUI') as factory:
            self.assertEqual(gui.main(['--hidden', '--auto-load-qwen']), 0)
        instance.notify.assert_called_once_with(show=False, auto_load=True)
        instance.close.assert_called_once()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)