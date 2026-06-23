from __future__ import annotations

import math

import numpy as np

from audio.synth_engine import AudioEngineBase, Voice


class PianoEngine(AudioEngineBase):
    def _voice_wave(self, voice: Voice, frames: int) -> np.ndarray:
        t = np.arange(frames, dtype=np.float32) / self.sample_rate
        ages = voice.age + t
        phases = voice.phase + np.cumsum(
            np.full(frames, 2.0 * math.pi * voice.frequency / self.sample_rate, dtype=np.float32)
        )
        voice.phase = float(phases[-1] % (2.0 * math.pi))

        base = np.sin(phases)
        second = 0.55 * np.sin(phases * 2.0 + 0.03)
        third = 0.28 * np.sin(phases * 3.0 + 0.09)
        fourth = 0.12 * np.sin(phases * 4.0 + 0.16)
        hammer = 0.015 * np.random.uniform(-1.0, 1.0, frames).astype(np.float32)

        body = base + second + third + fourth + hammer
        tonal_decay = np.exp(-3.8 * ages)
        upper_decay = np.exp(-7.5 * ages)
        body = (base * tonal_decay) + ((second + third + fourth + hammer) * upper_decay)

        release = 0.18
        if voice.released:
            release_curve = np.clip(1.0 - (voice.release_age + t) / release, 0.0, 1.0)
            body = body * release_curve
            voice.release_age += frames / self.sample_rate
            if voice.release_age >= release:
                voice.finished = True

        voice.age += frames / self.sample_rate
        return (body * voice.velocity).astype(np.float32)
