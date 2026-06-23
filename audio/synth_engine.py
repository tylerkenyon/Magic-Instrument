from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import sounddevice as sd

import config


@dataclass
class Voice:
    note: int
    frequency: float
    velocity: float
    waveform: str
    brightness: float
    phase: float = 0.0
    age: float = 0.0
    released: bool = False
    release_age: float = 0.0
    finished: bool = False


class AudioEngineBase:
    def __init__(
        self,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        block_size: int = config.AUDIO_BLOCK_SIZE,
        master_volume: float = config.MASTER_VOLUME,
        start_stream: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.master_volume = master_volume
        self.lock = threading.Lock()
        self.active_voices: Dict[int, Voice] = {}
        self.stream: Optional[sd.OutputStream] = None
        if start_stream:
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()

    def set_master_volume(self, volume: float) -> None:
        self.master_volume = float(np.clip(volume, 0.05, 1.0))

    def note_on(
        self,
        note: int,
        velocity: float,
        waveform: Optional[str] = None,
        brightness: float = 0.5,
    ) -> None:
        voice = Voice(
            note=note,
            frequency=config.midi_to_frequency(note),
            velocity=float(np.clip(velocity, 0.05, 1.0)),
            waveform=waveform or config.DEFAULT_SYNTH_WAVEFORM,
            brightness=float(np.clip(brightness, 0.0, 1.0)),
        )
        with self.lock:
            self.active_voices[note] = voice

    def note_off(self, note: int) -> None:
        with self.lock:
            voice = self.active_voices.get(note)
            if voice is not None:
                voice.released = True

    def all_notes_off(self) -> None:
        with self.lock:
            for voice in self.active_voices.values():
                voice.released = True

    def close(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, outdata, frames, _time_info, status) -> None:
        if status:
            pass
        with self.lock:
            block = self._render_active_voices(frames)
        outdata[:, 0] = block

    def _render_active_voices(self, frames: int) -> np.ndarray:
        mix = np.zeros(frames, dtype=np.float32)
        finished_notes: List[int] = []
        for note, voice in self.active_voices.items():
            mix += self._voice_wave(voice, frames)
            if voice.finished:
                finished_notes.append(note)
        for note in finished_notes:
            self.active_voices.pop(note, None)
        return np.clip(mix * self.master_volume, -1.0, 1.0)

    def _voice_wave(self, voice: Voice, frames: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def render_timeline(
        cls,
        events: Iterable[dict],
        duration_seconds: float,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
    ) -> np.ndarray:
        engine = cls(sample_rate=sample_rate, start_stream=False)
        return engine._render_timeline(events, duration_seconds)

    def _render_timeline(self, events: Iterable[dict], duration_seconds: float) -> np.ndarray:
        total_frames = int(max(duration_seconds, 0.0) * self.sample_rate) + self.sample_rate
        output = np.zeros(total_frames, dtype=np.float32)
        ordered_events = sorted(events, key=lambda item: float(item.get("time", 0.0)))
        cursor = 0

        for event in ordered_events:
            event_frame = min(int(float(event.get("time", 0.0)) * self.sample_rate), total_frames)
            if event_frame > cursor:
                with self.lock:
                    output[cursor:event_frame] = self._render_active_voices(event_frame - cursor)
                cursor = event_frame
            self._apply_timeline_event(event)

        if cursor < total_frames:
            with self.lock:
                output[cursor:] = self._render_active_voices(total_frames - cursor)

        return np.clip(output, -1.0, 1.0)

    def _apply_timeline_event(self, event: dict) -> None:
        event_type = event.get("type")
        note = int(event.get("note", -1))
        if event_type == "note_on" and note >= 0:
            self.note_on(
                note=note,
                velocity=float(event.get("velocity", 0.7)),
                waveform=event.get("waveform"),
                brightness=float(event.get("brightness", 0.5)),
            )
        elif event_type == "note_off" and note >= 0:
            self.note_off(note)


class SynthEngine(AudioEngineBase):
    def __init__(self, *args, waveform: str = config.DEFAULT_SYNTH_WAVEFORM, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.waveform = waveform

    def set_waveform(self, waveform: str) -> None:
        if waveform in config.SYNTH_WAVEFORMS:
            self.waveform = waveform

    def note_on(
        self,
        note: int,
        velocity: float,
        waveform: Optional[str] = None,
        brightness: float = 0.5,
    ) -> None:
        super().note_on(
            note=note,
            velocity=velocity,
            waveform=waveform or self.waveform,
            brightness=brightness,
        )

    def _voice_wave(self, voice: Voice, frames: int) -> np.ndarray:
        t = np.arange(frames, dtype=np.float32) / self.sample_rate
        ages = voice.age + t
        vibrato = 1.0 + 0.0025 * np.sin(2.0 * math.pi * 5.5 * ages)
        phases = voice.phase + np.cumsum(
            2.0 * math.pi * voice.frequency * vibrato / self.sample_rate
        ).astype(np.float32)
        voice.phase = float(phases[-1] % (2.0 * math.pi))

        waveform = voice.waveform
        if waveform == "square":
            wave = np.sign(np.sin(phases))
        elif waveform == "saw":
            wave = 2.0 * ((phases / (2.0 * math.pi)) % 1.0) - 1.0
        else:
            wave = np.sin(phases)

        harmonic = np.sin(phases * 2.0) * (0.12 + 0.20 * voice.brightness)
        wave = wave + harmonic

        attack = 0.01
        decay = 0.08
        sustain = 0.72
        release = 0.14

        if voice.released:
            release_curve = np.clip(1.0 - (voice.release_age + t) / release, 0.0, 1.0)
            envelope = release_curve * sustain
            voice.release_age += frames / self.sample_rate
            if voice.release_age >= release:
                voice.finished = True
        else:
            attack_curve = np.clip(ages / attack, 0.0, 1.0)
            decay_progress = np.clip((ages - attack) / decay, 0.0, 1.0)
            envelope = np.where(
                ages < attack,
                attack_curve,
                1.0 - (1.0 - sustain) * decay_progress,
            )

        voice.age += frames / self.sample_rate
        return (wave * envelope * voice.velocity).astype(np.float32)
