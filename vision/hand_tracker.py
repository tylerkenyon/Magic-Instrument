from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

import config


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


@dataclass
class HandObservation:
    label: str
    landmarks_norm: List[Tuple[float, float]]
    landmarks_px: List[Tuple[int, int]]
    pinch_distance: float

    @property
    def index_tip_norm(self) -> Tuple[float, float]:
        return self.landmarks_norm[8]

    @property
    def index_tip_px(self) -> Tuple[int, int]:
        return self.landmarks_px[8]

    @property
    def thumb_tip_norm(self) -> Tuple[float, float]:
        return self.landmarks_norm[4]

    @property
    def wrist_norm(self) -> Tuple[float, float]:
        return self.landmarks_norm[0]


class HandTracker:
    def __init__(self) -> None:
        self.backend = "tasks"
        self.hands = None
        self._video_timestamp_ms = 0
        self._init_tracker()

    def _init_tracker(self) -> None:

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            )
            return

        self.backend = "tasks"
        model_path = self._ensure_hand_landmarker_model()
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _ensure_hand_landmarker_model(self) -> Path:
        model_path = config.HAND_LANDMARKER_MODEL_PATH
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(config.HAND_LANDMARKER_MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                f"the model couldn't be fetched from {config.HAND_LANDMARKER_MODEL_URL}. "
                f"to {model_path}."
            ) from exc
        return model_path

    def process(self, frame_bgr) -> List[HandObservation]:
        if self.backend == "solutions":
            return self._process_with_solutions(frame_bgr)
        return self._process_with_tasks(frame_bgr)

    def _process_with_solutions(self, frame_bgr) -> List[HandObservation]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return []

        hands: List[HandObservation] = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            landmarks_norm: List[Tuple[float, float]] = []
            landmarks_px: List[Tuple[int, int]] = []
            for landmark in hand_landmarks.landmark:
                x = min(max(landmark.x, 0.0), 1.0)
                y = min(max(landmark.y, 0.0), 1.0)
                landmarks_norm.append((x, y))
                landmarks_px.append((int(x * width), int(y * height)))

            thumb_x, thumb_y = landmarks_norm[4]
            index_x, index_y = landmarks_norm[8]
            pinch_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

            hands.append(
                HandObservation(
                    label=label,
                    landmarks_norm=landmarks_norm,
                    landmarks_px=landmarks_px,
                    pinch_distance=pinch_distance,
                )
            )

        return hands

    def _process_with_tasks(self, frame_bgr) -> List[HandObservation]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._video_timestamp_ms += max(int(1000 / max(config.TARGET_FPS, 1)), 1)
        results = self.hands.detect_for_video(mp_image, self._video_timestamp_ms)

        if not results.hand_landmarks or not results.handedness:
            return []

        hands: List[HandObservation] = []
        for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            label = handedness[0].category_name
            landmarks_norm: List[Tuple[float, float]] = []
            landmarks_px: List[Tuple[int, int]] = []
            for landmark in hand_landmarks:
                x = min(max(float(landmark.x), 0.0), 1.0)
                y = min(max(float(landmark.y), 0.0), 1.0)
                landmarks_norm.append((x, y))
                landmarks_px.append((int(x * width), int(y * height)))

            thumb_x, thumb_y = landmarks_norm[4]
            index_x, index_y = landmarks_norm[8]
            pinch_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

            hands.append(
                HandObservation(
                    label=label,
                    landmarks_norm=landmarks_norm,
                    landmarks_px=landmarks_px,
                    pinch_distance=pinch_distance,
                )
            )

        return hands

    def close(self) -> None:
        if self.hands is not None:
            self.hands.close()
