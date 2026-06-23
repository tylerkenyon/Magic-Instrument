from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import cv2

import config
from app_logging import setup_logging
from audio.piano_engine import PianoEngine
from audio.recorder import PerformanceRecorder
from audio.synth_engine import SynthEngine
from ui.control_panel import ControlPanel
from ui.overlay import OverlayRenderer
from vision.gesture_detector import GestureDetector
from vision.hand_tracker import HandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="webcam hand gestures to play instruments")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX, help="camera index")
    parser.add_argument("--width", type=int, default=config.FRAME_WIDTH, help="Camera width")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="Camera height")
    return parser.parse_args()


def create_engine(mode: str):
    if mode == "synth":
        return SynthEngine()
    return PianoEngine()


def discover_camera_indices(max_index: int = 6) -> list[int]:
    discovered: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            continue
        ok, _frame = capture.read()
        capture.release()
        if ok:
            discovered.append(index)
    return discovered


class InvisibleInstrumentApp:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.instrument_mode = "piano"
        self.current_note_name = "-"
        self.current_waveform = config.DEFAULT_SYNTH_WAVEFORM
        self.expression = 0.7
        self.octave_shift = 0
        self.key_index: Optional[int] = None
        self.audio_error: Optional[str] = None

        self.tracker: Optional[HandTracker] = None
        self.capture = None
        self.engine = None
        self.recorder = PerformanceRecorder(config.RECORDINGS_DIR)
        self.overlay = OverlayRenderer()
        self.gesture_detector = GestureDetector()
        self.running = False
        self.window_ready = False

        cameras = discover_camera_indices()
        if args.camera_index not in cameras:
            cameras = [args.camera_index] + cameras
        self.logger.info("Detected camera indices: %s", cameras or [args.camera_index])

        self.panel = ControlPanel(
            camera_indices=cameras,
            on_start=self.start_session,
            on_stop=self.stop_session,
            on_refresh_cameras=self.refresh_cameras,
            on_set_mode=self.set_mode,
            on_toggle_recording=self.toggle_recording,
            on_playback=self.playback_latest,
            on_quit=self.quit,
        )
        self.panel.set_mode(self.instrument_mode)
        self.panel.set_recording_state(False)
        self.panel.set_session_state(False)
        self.panel.set_current_note(self.current_note_name)
        self.panel.set_status("Choose a camera")

    def start_session(self) -> None:
        if self.running:
            self.logger.info("session already running")
            self.panel.set_status("session already running")
            return

        camera_index = self.panel.get_camera_index()
        self.logger.info("Starting session - camera %s", camera_index)
        self.panel.set_status(f"Starting camera {camera_index}...")
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self.logger.info("Starting Hand Tracking")
            self.tracker = HandTracker()
            self.logger.info("Hand tracker ready")
        except Exception as exc:
            self.logger.error("Hand tracker start failed: %s", exc)
            self.panel.set_status(f"Hand tracker failed: {exc}")
            self.tracker = None
            return

        self.logger.info("Opening webcam")
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        self.capture.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        if not self.capture.isOpened():
            self.logger.error("Could not open webcam %s", camera_index)
            self.panel.set_status(f"Failed to open webcam {camera_index}.")
            self.cleanup_runtime()
            return

        try:
            self.logger.info("Starting audio engine for %s mode", self.instrument_mode)
            self.engine = create_engine(self.instrument_mode)
            self.audio_error = None
            self.logger.info("Audio engine ready.")
        except Exception as exc:
            self.engine = None
            self.audio_error = str(exc)
            self.logger.warning("Audio engine unavailable: %s", exc)

        try:
            self.logger.info("Creating Window")
            cv2.namedWindow("Magic Intrusment", cv2.WINDOW_NORMAL)
            self.window_ready = True
            self.logger.info("window created")
        except Exception as exc:
            self.window_ready = False
            self.logger.error("OpenCV window init failed: %s", exc)
            self.panel.set_status(f"window failed: {exc}")
            self.cleanup_runtime()
            return

        self.running = True
        self.panel.set_session_state(True)
        self.panel.set_status("Session running")
        self.schedule_next_frame()

    def schedule_next_frame(self) -> None:
        self.panel.root.after(10, self.process_frame)

    def process_frame(self) -> None:
        if not self.running:
            return
        if self.capture is None or self.tracker is None:
            self.logger.error("missing object")
            self.panel.set_status("missing camera or tracker/deps.")
            self.stop_session()
            return

        ok, frame = self.capture.read()
        if not ok:
            self.logger.error("frame read failed")
            self.panel.set_status("frame read failed")
            self.stop_session()
            return

        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        hands = self.tracker.process(frame)

        if not self.recorder.is_playing_back:
            gesture_result = self.gesture_detector.update(hands, frame.shape, self.instrument_mode, now)
            self.current_note_name = gesture_result.current_note_name
            self.key_index = gesture_result.key_index
            self.expression = gesture_result.control_state.expression
            self.octave_shift = gesture_result.control_state.octave_shift
            if self.instrument_mode == "synth":
                self.current_waveform = gesture_result.control_state.waveform
                if self.engine is not None and isinstance(self.engine, SynthEngine):
                    self.engine.set_waveform(self.current_waveform)

            if self.engine is not None:
                self.engine.set_master_volume(self.expression)

            for event in gesture_result.note_events:
                if self.engine is not None:
                    dispatch_note_event(self.engine, event, self.current_waveform)
                self.recorder.record_note_event(event)
        else:
            playback_events = self.recorder.poll_playback_events()
            for event in playback_events:
                if event.get("type") == "mode_change":
                    self.set_mode(str(event.get("instrument", self.instrument_mode)), from_playback=True)
                    continue

                event_mode = str(event.get("instrument", self.instrument_mode))
                if event_mode != self.instrument_mode:
                    self.set_mode(event_mode, from_playback=True)
                if event_mode == "synth":
                    self.current_waveform = str(event.get("waveform", self.current_waveform))
                if event.get("type") == "note_on":
                    self.current_note_name = config.midi_to_name(int(event.get("note", 0)))
                if self.engine is not None:
                    dispatch_note_event(self.engine, event, self.current_waveform)

            if not self.recorder.is_playing_back and self.engine is not None:
                self.engine.all_notes_off()
                self.current_note_name = "-"

        status_text = self.recorder.last_status_message
        display = self.overlay.draw(
            frame=frame,
            hands=hands,
            instrument_mode=self.instrument_mode,
            current_note=self.current_note_name,
            recording=self.recorder.is_recording,
            playback=self.recorder.is_playing_back,
            status_text=status_text,
            waveform=self.current_waveform,
            expression=self.expression,
            octave_shift=self.octave_shift,
            key_index=self.key_index,
            audio_error=self.audio_error,
        )

        try:
            cv2.imshow("Magic Instrument", display)
        except Exception as exc:
            self.logger.error("cv2.imshow failed: %s", exc)
            self.panel.set_status(f"Preview display failed: {exc}")
            self.stop_session()
            return

        key = cv2.waitKey(1) & 0xFF
        self.handle_keypress(key)
        self.panel.set_recording_state(self.recorder.is_recording)
        self.panel.set_current_note(self.current_note_name)
        self.panel.set_status(status_text)
        self.schedule_next_frame()

    def handle_keypress(self, key: int) -> None:
        if key in (27, ord("q"), ord("Q")):
            self.quit()
        elif key == ord("1"):
            self.set_mode("piano")
        elif key == ord("2"):
            self.set_mode("synth")
        elif key in (ord("r"), ord("R")):
            self.toggle_recording()
        elif key in (ord("p"), ord("P")):
            self.playback_latest()

    def set_mode(self, new_mode: str, from_playback: bool = False) -> None:
        if new_mode == self.instrument_mode:
            self.panel.set_mode(new_mode)
            return

        self.logger.info("Switching instrument to %s", new_mode)
        if not self.running and self.engine is None: #fixed window hanging after switch
            self.instrument_mode = new_mode
            self.panel.set_mode(new_mode)
            self.recorder.last_status_message = f"{self.instrument_mode.title()} mode selected"
            self.panel.set_status(self.recorder.last_status_message)
            return

        release_events = self.gesture_detector.force_note_offs(self.instrument_mode)
        for event in release_events:
            if self.engine is not None:
                dispatch_note_event(self.engine, event, self.current_waveform)
            self.recorder.record_note_event(event)

        if self.engine is not None:
            self.engine.all_notes_off()
            self.engine.close()

        self.instrument_mode = new_mode
        self.panel.set_mode(new_mode)
        self.gesture_detector.reset()

        try:
            self.engine = create_engine(self.instrument_mode)
            self.audio_error = None
        except Exception as exc:
            self.engine = None
            self.audio_error = str(exc)
            self.logger.warning("Audio engine unavailable: %s", exc)

        if self.instrument_mode != "synth":
            self.current_waveform = config.DEFAULT_SYNTH_WAVEFORM

        if not from_playback and self.recorder.is_recording:
            self.recorder.record_mode_change(self.instrument_mode)
        self.recorder.last_status_message = f"{self.instrument_mode.title()} mode"

    def toggle_recording(self) -> None:
        if not self.running:
            self.logger.warning("session unavailable.")
            self.panel.set_status("session unavailable")
            return
        if self.recorder.is_recording:
            self.logger.info("Stopping recording")
            release_events = self.gesture_detector.force_note_offs(self.instrument_mode)
            for event in release_events:
                if self.engine is not None:
                    dispatch_note_event(self.engine, event, self.current_waveform)
                self.recorder.record_note_event(event)
            save_result = self.recorder.stop_recording()
            self.logger.info(save_result.message)
            self.panel.set_status(save_result.message)
        else:
            self.logger.info("Starting recording.")
            self.recorder.start_recording(self.instrument_mode)
            self.panel.set_status("Recording started.")
        self.panel.set_recording_state(self.recorder.is_recording)

    def playback_latest(self) -> None:
        self.logger.info("playback request")
        if not self.running:
            self.logger.warning("session is inactive")
            self.panel.set_status("session is inactive")
            return
        if self.recorder.is_recording:
            self.toggle_recording()
        if self.engine is not None:
            self.engine.all_notes_off()
        self.gesture_detector.reset()
        started = self.recorder.start_playback()
        if started:
            self.logger.info("Playback started")
            self.panel.set_status("Playback started")
        else:
            self.logger.warning("No recording available for playback")
            self.panel.set_status("No recording available")

    def refresh_cameras(self) -> None:
        cameras = discover_camera_indices()
        self.logger.info("Refreshed cameras: %s", cameras or [self.args.camera_index])
        if self.args.camera_index not in cameras:
            cameras = [self.args.camera_index] + cameras
        self.panel.set_camera_indices(cameras)
        self.panel.set_status("Camera list refreshed.")

    def stop_session(self) -> None:
        if not self.running and self.capture is None and self.tracker is None:
            self.panel.set_session_state(False)
            return
        self.logger.info("Stopping session")
        self.running = False
        self.cleanup_runtime()
        self.panel.set_session_state(False)
        self.panel.set_current_note("-")
        self.panel.set_status("Session stopped")

    def cleanup_runtime(self) -> None:
        if self.engine is not None:
            self.engine.all_notes_off()
            self.engine.close()
            self.engine = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        if self.window_ready:
            cv2.destroyWindow("Magic Instrument")
            self.window_ready = False

    def quit(self) -> None:
        self.logger.info("Quitting application.")
        self.stop_session()
        cv2.destroyAllWindows()
        self.panel.root.quit()
        self.panel.root.destroy()


def dispatch_note_event(engine, event: dict, current_waveform: str) -> None:
    event_type = event.get("type")
    note = int(event.get("note", -1))
    if note < 0:
        return

    if event_type == "note_on":
        waveform = str(event.get("waveform", current_waveform))
        brightness = float(event.get("brightness", 0.5))
        if isinstance(engine, SynthEngine):
            engine.set_waveform(waveform)
        engine.note_on(
            note=note,
            velocity=float(event.get("velocity", 0.7)),
            waveform=waveform,
            brightness=brightness,
        )
    elif event_type == "note_off":
        engine.note_off(note)


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    logger.info("Launching Magic Instrument")
    app = InvisibleInstrumentApp(args, logger)
    app.panel.root.mainloop()
    logger.info("app closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
