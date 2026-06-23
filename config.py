from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
MODELS_DIR = PROJECT_ROOT / "models"

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.65
MIN_TRACKING_CONFIDENCE = 0.60

VIRTUAL_KEYS = 14
BASE_MIDI_NOTES = {
    "piano": 60,
    "synth": 60,
}

TRIGGER_ZONE_RATIO = 0.55
DOWNWARD_VELOCITY_THRESHOLD = 650.0
PINCH_THRESHOLD = 0.055
NOTE_DEBOUNCE_SECONDS = 0.12
TAP_NOTE_LENGTH_SECONDS = 0.22
DISPLAY_NOTE_HOLD_SECONDS = 0.50

AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 256
MASTER_VOLUME = 0.35

DEFAULT_SYNTH_WAVEFORM = "sine"
SYNTH_WAVEFORMS = ("sine", "square", "saw")

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_frequency(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def midi_to_name(note: int) -> str:
    octave = (note // 12) - 1
    return f"{NOTE_NAMES[note % 12]}{octave}"
