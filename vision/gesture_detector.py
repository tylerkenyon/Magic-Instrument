from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from vision.hand_tracker import HandObservation


@dataclass
class ControlState:
    waveform: str = config.DEFAULT_SYNTH_WAVEFORM
    expression: float = 0.7
    octave_shift: int = 0


@dataclass
class GestureResult:
    note_events: List[dict] = field(default_factory=list)
    current_note_name: str = "-"
    key_index: Optional[int] = None
    control_state: ControlState = field(default_factory=ControlState)


class GestureDetector:
    def __init__(self) -> None:
        self.last_positions: Dict[str, Tuple[float, float, float]] = {}
        self.active_pinch_note: Optional[int] = None
        self.pending_note_offs: List[Tuple[float, int, str]] = []
        self.last_trigger_time = 0.0
        self.display_note_name = "-"
        self.display_note_until = 0.0

    def reset(self) -> None:
        self.last_positions.clear()
        self.active_pinch_note = None
        self.pending_note_offs.clear()
        self.display_note_name = "-"
        self.display_note_until = 0.0

    def force_note_offs(self, instrument_mode: str) -> List[dict]:
        events = self._release_active_pinch(instrument_mode)
        for _release_time, note, note_instrument in self.pending_note_offs:
            events.append(
                {
                    "type": "note_off",
                    "note": note,
                    "instrument": note_instrument or instrument_mode,
                }
            )
        self.pending_note_offs.clear()
        self.last_positions.clear()
        return events

    def update(
        self,
        hands: List[HandObservation],
        frame_shape: Tuple[int, int, int],
        instrument_mode: str,
        now: float,
    ) -> GestureResult:
        result = GestureResult()
        result.note_events.extend(self._flush_pending_note_offs(now, instrument_mode))

        right_hand = next((hand for hand in hands if hand.label == "Right"), None)
        left_hand = next((hand for hand in hands if hand.label == "Left"), None)
        if right_hand is None and hands:
            right_hand = hands[0]
        if left_hand is None and len(hands) > 1:
            left_hand = hands[1] if hands[1] is not right_hand else None

        control_state = self._compute_controls(left_hand, instrument_mode)
        result.control_state = control_state

        if right_hand is None:
            result.note_events.extend(self._release_active_pinch(instrument_mode))
            if now > self.display_note_until:
                result.current_note_name = "-"
            else:
                result.current_note_name = self.display_note_name
            return result

        note, key_index = self._note_from_hand(right_hand, instrument_mode, control_state.octave_shift)
        result.key_index = key_index
        velocity = self._velocity_from_height(right_hand.index_tip_norm[1], control_state.expression)
        brightness = float(np.clip(1.0 - right_hand.pinch_distance * 8.0, 0.0, 1.0))

        state = self.last_positions.get(right_hand.label)
        if state is None:
            velocity_y = 0.0
        else:
            _last_x, last_y, last_t = state
            dt = max(now - last_t, 1e-6)
            velocity_y = (right_hand.index_tip_px[1] - last_y) / dt
        self.last_positions[right_hand.label] = (
            float(right_hand.index_tip_px[0]),
            float(right_hand.index_tip_px[1]),
            now,
        )

        pinch_active = right_hand.pinch_distance <= config.PINCH_THRESHOLD
        if pinch_active:
            result.note_events.extend(
                self._handle_pinch_note(
                    note,
                    velocity,
                    instrument_mode,
                    control_state.waveform,
                    brightness,
                    now,
                )
            )
        else:
            result.note_events.extend(self._release_active_pinch(instrument_mode))

        in_trigger_zone = right_hand.index_tip_norm[1] >= config.TRIGGER_ZONE_RATIO
        downward_tap = velocity_y >= config.DOWNWARD_VELOCITY_THRESHOLD and in_trigger_zone
        if (
            downward_tap
            and not pinch_active
            and (now - self.last_trigger_time) >= config.NOTE_DEBOUNCE_SECONDS
        ):
            result.note_events.append(
                {
                    "type": "note_on",
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument_mode,
                    "waveform": control_state.waveform,
                    "brightness": brightness,
                }
            )
            self.pending_note_offs.append(
                (now + config.TAP_NOTE_LENGTH_SECONDS, note, instrument_mode)
            )
            self.last_trigger_time = now
            self._set_display_note(note, now)

        if now > self.display_note_until and self.active_pinch_note is None:
            result.current_note_name = "-"
        else:
            result.current_note_name = self.display_note_name

        return result

    def _compute_controls(
        self,
        left_hand: Optional[HandObservation],
        instrument_mode: str,
    ) -> ControlState:
        if left_hand is None:
            return ControlState()

        wrist_y = left_hand.wrist_norm[1]
        if wrist_y < 0.35:
            octave_shift = 1
        elif wrist_y > 0.72:
            octave_shift = -1
        else:
            octave_shift = 0

        expression = float(np.clip(1.1 - left_hand.index_tip_norm[1], 0.2, 1.0))
        waveform = config.DEFAULT_SYNTH_WAVEFORM
        if instrument_mode == "synth":
            waveform_index = min(
                int(left_hand.index_tip_norm[0] * len(config.SYNTH_WAVEFORMS)),
                len(config.SYNTH_WAVEFORMS) - 1,
            )
            waveform = config.SYNTH_WAVEFORMS[waveform_index]

        return ControlState(
            waveform=waveform,
            expression=expression,
            octave_shift=octave_shift,
        )

    def _note_from_hand(
        self,
        hand: HandObservation,
        instrument_mode: str,
        octave_shift: int,
    ) -> Tuple[int, int]:
        key_index = min(int(hand.index_tip_norm[0] * config.VIRTUAL_KEYS), config.VIRTUAL_KEYS - 1)
        note = config.BASE_MIDI_NOTES[instrument_mode] + key_index + (octave_shift * 12)
        return note, key_index

    def _velocity_from_height(self, y_position: float, expression: float) -> float:
        base = float(np.clip(1.1 - y_position, 0.2, 1.0))
        return float(np.clip(base * expression, 0.15, 1.0))

    def _handle_pinch_note(
        self,
        note: int,
        velocity: float,
        instrument_mode: str,
        waveform: str,
        brightness: float,
        now: float,
    ) -> List[dict]:
        events: List[dict] = []
        if self.active_pinch_note is None:
            events.append(
                {
                    "type": "note_on",
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument_mode,
                    "waveform": waveform,
                    "brightness": brightness,
                }
            )
            self.active_pinch_note = note
            self._set_display_note(note, now)
        elif self.active_pinch_note != note:
            events.append(
                {
                    "type": "note_off",
                    "note": self.active_pinch_note,
                    "instrument": instrument_mode,
                }
            )
            events.append(
                {
                    "type": "note_on",
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument_mode,
                    "waveform": waveform,
                    "brightness": brightness,
                }
            )
            self.active_pinch_note = note
            self._set_display_note(note, now)
        return events

    def _release_active_pinch(self, instrument_mode: str) -> List[dict]:
        if self.active_pinch_note is None:
            return []
        note = self.active_pinch_note
        self.active_pinch_note = None
        return [{"type": "note_off", "note": note, "instrument": instrument_mode}]

    def _flush_pending_note_offs(self, now: float, instrument_mode: str) -> List[dict]:
        due: List[dict] = []
        remaining: List[Tuple[float, int, str]] = []
        for release_time, note, note_instrument in self.pending_note_offs:
            if now >= release_time:
                due.append(
                    {
                        "type": "note_off",
                        "note": note,
                        "instrument": note_instrument or instrument_mode,
                    }
                )
            else:
                remaining.append((release_time, note, note_instrument))
        self.pending_note_offs = remaining
        return due

    def _set_display_note(self, note: int, now: float) -> None:
        self.display_note_name = config.midi_to_name(note)
        self.display_note_until = now + config.DISPLAY_NOTE_HOLD_SECONDS
