from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import config
from audio.piano_engine import PianoEngine
from audio.synth_engine import SynthEngine


@dataclass
class SaveResult:
    json_path: Path
    wav_path: Optional[Path]
    message: str


class PerformanceRecorder:
    def __init__(self, recordings_dir: Path) -> None:
        self.recordings_dir = recordings_dir
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.is_playing_back = False
        self._recording_started_at = 0.0
        self._playback_started_at = 0.0
        self._playback_index = 0
        self._events: List[dict] = []
        self._playback_events: List[dict] = []
        self.latest_recording_path: Optional[Path] = None
        self.last_status_message = "Idle"
        self._last_recorded_mode: Optional[str] = None

    def start_recording(self, initial_mode: str) -> None:
        self.is_recording = True
        self.is_playing_back = False
        self._recording_started_at = time.perf_counter()
        self._events = []
        self._last_recorded_mode = None
        self.record_mode_change(initial_mode, force=True)
        self.last_status_message = "Recording"

    def stop_recording(self) -> SaveResult:
        self.is_recording = False
        duration = self._current_record_time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = self.recordings_dir / f"performance_{timestamp}.json"
        wav_path = self.recordings_dir / f"performance_{timestamp}.wav"

        payload = {
            "created_at": timestamp,
            "duration_seconds": duration,
            "sample_rate": config.AUDIO_SAMPLE_RATE,
            "events": self._events,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.latest_recording_path = json_path

        wav_message = "WAV export stopped"
        saved_wav: Optional[Path] = None
        try:
            self._export_wav(payload, wav_path)
            saved_wav = wav_path
            wav_message = f"WAV exported to {wav_path.name}."
        except Exception as exc:
            wav_message = f"WAV export stopped: {exc}"

        self.last_status_message = f"Saved {json_path.name}"
        return SaveResult(
            json_path=json_path,
            wav_path=saved_wav,
            message=f"Saved {json_path.name}. {wav_message}",
        )

    def record_mode_change(self, instrument_mode: str, force: bool = False) -> None:
        if not self.is_recording:
            return
        if not force and instrument_mode == self._last_recorded_mode:
            return
        self._last_recorded_mode = instrument_mode
        self._events.append(
            {
                "time": self._current_record_time(),
                "type": "mode_change",
                "instrument": instrument_mode,
            }
        )

    def record_note_event(self, event: dict) -> None:
        if not self.is_recording:
            return
        record = dict(event)
        record["time"] = self._current_record_time()
        self._events.append(record)

    def start_playback(self, path: Optional[Path] = None) -> bool:
        target = path or self.latest_recording_path
        if target is None or not target.exists():
            self.last_status_message = "No recording available"
            return False

        payload = json.loads(target.read_text(encoding="utf-8"))
        self._playback_events = sorted(payload.get("events", []), key=lambda item: item.get("time", 0.0))
        self._playback_index = 0
        self._playback_started_at = time.perf_counter()
        self.is_playing_back = True
        self.last_status_message = f"Playing {target.name}"
        return True

    def stop_playback(self) -> None:
        self.is_playing_back = False
        self._playback_events = []
        self._playback_index = 0
        self.last_status_message = "Playback stopped"

    def poll_playback_events(self) -> List[dict]:
        if not self.is_playing_back:
            return []

        elapsed = time.perf_counter() - self._playback_started_at
        due: List[dict] = []
        while self._playback_index < len(self._playback_events):
            event = self._playback_events[self._playback_index]
            if float(event.get("time", 0.0)) > elapsed:
                break
            due.append(event)
            self._playback_index += 1

        if self._playback_index >= len(self._playback_events):
            self.is_playing_back = False
            self.last_status_message = "Playback finished"

        return due

    def _current_record_time(self) -> float:
        if self._recording_started_at == 0.0:
            return 0.0
        return round(time.perf_counter() - self._recording_started_at, 4)

    def _export_wav(self, payload: dict, wav_path: Path) -> None:
        from scipy.io import wavfile

        events = payload.get("events", [])
        duration = float(payload.get("duration_seconds", 0.0))
        sample_rate = int(payload.get("sample_rate", config.AUDIO_SAMPLE_RATE))

        synth_events = [event for event in events if event.get("instrument") == "synth" and event.get("type") in {"note_on", "note_off"}]
        piano_events = [event for event in events if event.get("instrument") == "piano" and event.get("type") in {"note_on", "note_off"}]

        mix = np.zeros(int((duration + 1.0) * sample_rate), dtype=np.float32)
        if synth_events:
            mix += SynthEngine.render_timeline(synth_events, duration, sample_rate=sample_rate)
        if piano_events:
            mix += PianoEngine.render_timeline(piano_events, duration, sample_rate=sample_rate)

        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.0:
            mix = mix / peak
        wavfile.write(wav_path, sample_rate, (mix * 32767).astype(np.int16))
