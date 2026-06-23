# Magic Instrument

Magic Instrument is a real time webcam instrument for Python. It tracks your hands with MediaPipe, turns hand gestures into notes, and lets you switch between a piano engine and a synthesizer. Also supports playback and recording

## Features

- Real-time webcam hand tracking with MediaPipe Hands
- Virtual air keys mapped across the camera frame
- Two instrument modes:
  - `1` Piano mode
  - `2` Synthesizer mode
- Gesture controls:
  - Right hand index finger chooses pitch
  - Pinch thumb + index to hold a note
  - Fast downward taps trigger short notes
  - Left hand adjusts octave and expression
  - In synth mode, left hand horizontal position selects waveform
- On-screen overlay for camera feed, hand landmarks, note, instrument mode, recording state, and playback state
- Timestamped recording export to JSON
- Optional WAV export rendered from the recorded note events


## Requirements

- Python 3.9+
- A webcam
- A working audio output device
- Internet access on first run if the newer MediaPipe Tasks backend needs to
  download the official hand landmarker model automatically

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
py main.py
```

Optional camera override:

```bash
py main.py --camera-index 1
```

## Controls

- `1`: switch to piano mode
- `2`: switch to synthesizer mode
- `R`: start or stop recording
- `P`: play the latest saved recording
- `Q` or `Esc`: quit

## Gesture Design

- The camera frame is divided into horizontal note zones.
- The right hand index fingertip selects the current note.
- A note is triggered when:
  - the right thumb and index finger pinch together, or
  - the index finger moves downward quickly into the lower half of the frame
- Left hand controls:
  - hand height changes octave
  - index height changes expression / volume
  - in synth mode, left-hand horizontal position picks `sine`, `square`, or `saw`

You can tune thresholds in `config.py`.

## Recording

When recording is enabled, the app saves a timestamped JSON file containing:

- mode changes
- note on/off events
- timing information
- synth waveform metadata when relevant

If `scipy` is installed and audio rendering is fine, it will write a WAV file along with the JSON file.

Playback uses the latest recording inside the app when you press `P`.


## Troubleshooting

- If the webcam cannot be opened, try a different `--camera-index`.
- Watch the terminal logs. The app now prints colored startup steps for tracker, webcam, audio, and preview window creation.
- If MediaPipe fails to detect hands, improve lighting and keep your hand inside the frame.
- If audio fails to initialize, the app keeps running and shows the error in the overlay.
- On slower machines, reduce the camera resolution or tweak the gesture thresholds in [`config.py`](/C:/Users/Tyler/Documents/Invisible%20instument/config.py).
