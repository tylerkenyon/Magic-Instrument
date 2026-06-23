from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Optional


class ControlPanel:
    def __init__(
        self,
        camera_indices: Iterable[int],
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_refresh_cameras: Callable[[], None],
        on_set_mode: Callable[[str], None],
        on_toggle_recording: Callable[[], None],
        on_playback: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Invisible Instrument Control Panel")
        self.root.geometry("440x320")
        self.root.protocol("WM_DELETE_WINDOW", on_quit)

        self.on_start = on_start
        self.on_stop = on_stop
        self.on_refresh_cameras = on_refresh_cameras
        self.on_set_mode = on_set_mode
        self.on_toggle_recording = on_toggle_recording
        self.on_playback = on_playback

        self.camera_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="piano")
        self.status_var = tk.StringVar(value="Ready")
        self.session_var = tk.StringVar(value="Stopped")
        self.recording_var = tk.StringVar(value="Recording: OFF")
        self.current_note_var = tk.StringVar(value="Current note: -")

        self._build_menu(on_quit)
        self._build_layout(list(camera_indices))

    def _build_menu(self, on_quit: Callable[[], None]) -> None:
        menu_bar = tk.Menu(self.root)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Start Session", command=self.on_start)
        file_menu.add_command(label="Stop Session", command=self.on_stop)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=on_quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        instrument_menu = tk.Menu(menu_bar, tearoff=0)
        instrument_menu.add_command(label="Piano", command=lambda: self._set_mode_from_menu("piano"))
        instrument_menu.add_command(label="Synthesizer", command=lambda: self._set_mode_from_menu("synth"))
        menu_bar.add_cascade(label="Instrument", menu=instrument_menu)

        recording_menu = tk.Menu(menu_bar, tearoff=0)
        recording_menu.add_command(label="Start/Stop Recording", command=self.on_toggle_recording)
        recording_menu.add_command(label="Playback Latest", command=self.on_playback)
        menu_bar.add_cascade(label="Recording", menu=recording_menu)

        camera_menu = tk.Menu(menu_bar, tearoff=0)
        camera_menu.add_command(label="Refresh Cameras", command=self.on_refresh_cameras)
        menu_bar.add_cascade(label="Camera", menu=camera_menu)

        self.root.config(menu=menu_bar)

    def _build_layout(self, camera_indices: list[int]) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Webcam").grid(row=0, column=0, sticky="w")
        self.camera_combo = ttk.Combobox(
            frame,
            state="readonly",
            width=24,
            textvariable=self.camera_var,
        )
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.set_camera_indices(camera_indices)

        ttk.Button(frame, text="Refresh", command=self.on_refresh_cameras).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(frame, text="Instrument").grid(row=1, column=0, sticky="w", pady=(14, 0))
        mode_frame = ttk.Frame(frame)
        mode_frame.grid(row=1, column=1, columnspan=2, sticky="w", pady=(14, 0))
        ttk.Radiobutton(
            mode_frame,
            text="Piano",
            value="piano",
            variable=self.mode_var,
            command=lambda: self.on_set_mode(self.mode_var.get()),
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_frame,
            text="Synth",
            value="synth",
            variable=self.mode_var,
            command=lambda: self.on_set_mode(self.mode_var.get()),
        ).pack(side=tk.LEFT, padx=(12, 0))

        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(22, 0))
        ttk.Button(controls, text="Start Session", command=self.on_start).pack(side=tk.LEFT)
        ttk.Button(controls, text="Stop Session", command=self.on_stop).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Record", command=self.on_toggle_recording).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Playback", command=self.on_playback).pack(side=tk.LEFT, padx=(10, 0))

        status_frame = ttk.LabelFrame(frame, text="Status", padding=12)
        status_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(22, 0))
        ttk.Label(status_frame, textvariable=self.session_var).pack(anchor="w")
        ttk.Label(status_frame, textvariable=self.recording_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(status_frame, textvariable=self.current_note_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=360).pack(anchor="w", pady=(10, 0))

        hint = ttk.Label(
            frame,
            text="The camera feed still appears in an OpenCV window when the session is running.",
            wraplength=400,
        )
        hint.grid(row=4, column=0, columnspan=3, sticky="w", pady=(18, 0))

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)

    def _set_mode_from_menu(self, mode: str) -> None:
        self.mode_var.set(mode)
        self.on_set_mode(mode)

    def set_camera_indices(self, camera_indices: list[int]) -> None:
        values = [str(index) for index in camera_indices] or ["0"]
        self.camera_combo["values"] = values
        if self.camera_var.get() not in values:
            self.camera_var.set(values[0])

    def get_camera_index(self) -> int:
        value = self.camera_var.get().strip() or "0"
        return int(value)

    def set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def set_session_state(self, running: bool) -> None:
        self.session_var.set(f"Session: {'Running' if running else 'Stopped'}")

    def set_recording_state(self, recording: bool) -> None:
        self.recording_var.set(f"Recording: {'ON' if recording else 'OFF'}")

    def set_current_note(self, note_name: str) -> None:
        self.current_note_var.set(f"Current note: {note_name}")
