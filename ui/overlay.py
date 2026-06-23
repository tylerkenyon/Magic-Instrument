from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

import config
from vision.hand_tracker import HAND_CONNECTIONS, HandObservation


class OverlayRenderer:
    def draw(
        self,
        frame: np.ndarray,
        hands: Iterable[HandObservation],
        instrument_mode: str,
        current_note: str,
        recording: bool,
        playback: bool,
        status_text: str,
        waveform: str,
        expression: float,
        octave_shift: int,
        key_index: Optional[int],
        audio_error: Optional[str],
    ) -> np.ndarray:
        canvas = frame.copy()
        height, width = canvas.shape[:2]

        self._draw_virtual_keys(canvas, key_index)
        self._draw_hands(canvas, hands)

        header_lines = [
            f"Mode: {instrument_mode.upper()}",
            f"Note: {current_note}",
            f"Waveform: {waveform}",
            f"Expression: {expression:.2f}",
            f"Octave Shift: {octave_shift:+d}",
            f"Recording: {'ON' if recording else 'OFF'}",
            f"Playback: {'ON' if playback else 'OFF'}",
            "Controls: 1 Piano | 2 Synth | R Record | P Play | Q Quit",
        ]

        y = 28
        for line in header_lines:
            cv2.putText(
                canvas,
                line,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.67,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            y += 28

        footer_color = (0, 220, 120) if "Saved" in status_text or playback else (255, 210, 80)
        cv2.rectangle(canvas, (12, height - 42), (width - 12, height - 8), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            status_text,
            (24, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            footer_color,
            2,
            cv2.LINE_AA,
        )

        if audio_error:
            cv2.rectangle(canvas, (width - 460, 12), (width - 12, 60), (30, 30, 120), -1)
            cv2.putText(
                canvas,
                "Audio disabled",
                (width - 442, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                audio_error[:44],
                (width - 442, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def _draw_virtual_keys(self, frame: np.ndarray, key_index: Optional[int]) -> None:
        height, width = frame.shape[:2]
        key_width = width / config.VIRTUAL_KEYS

        for idx in range(config.VIRTUAL_KEYS):
            x1 = int(idx * key_width)
            x2 = int((idx + 1) * key_width)
            color = (40, 120, 240) if idx == key_index else (80, 80, 80)
            thickness = 2 if idx == key_index else 1
            cv2.rectangle(frame, (x1, int(height * 0.55)), (x2, height - 48), color, thickness)

    def _draw_hands(self, frame: np.ndarray, hands: Iterable[HandObservation]) -> None:
        for hand in hands:
            for start_idx, end_idx in HAND_CONNECTIONS:
                start_point = hand.landmarks_px[start_idx]
                end_point = hand.landmarks_px[end_idx]
                cv2.line(frame, start_point, end_point, (0, 255, 120), 2, cv2.LINE_AA)

            for idx, point in enumerate(hand.landmarks_px):
                radius = 6 if idx in (4, 8) else 4
                color = (0, 200, 255) if idx == 8 else (0, 255, 120)
                cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

            cv2.putText(
                frame,
                hand.label,
                (hand.landmarks_px[0][0] + 10, hand.landmarks_px[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
