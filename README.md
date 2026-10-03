# SemiX26 AI Workshop - FIDO

FIDO is a voice-controlled "fetch" robot demo that runs entirely on an Intel AI PC with
OpenVINO: it listens for a spoken command, looks through the camera, and uses a small
language model to decide which visible object to bring.

## Versions

Each version is a tagged commit - browse older ones under **Tags** (or `git checkout v1.0`).

| Tag | Main program | What changed |
|---|---|---|
| v1.0 | `multimodal10.py` + `brain_Int8.py` | Original code |
| **v2.0** (this) | `multimodal10_updated.py` | One program: continuous camera + YOLO, Silero voice detection, Whisper-medium, Phi-3 decides using recently seen objects, task queue |

## Version 2.0

| File | Purpose |
|---|---|
| `installation_instructions_new1.txt` | PowerShell setup script, updated for v2: adds PyTorch (Silero VAD) and exports Whisper-medium instead of Whisper-tiny |
| `multimodal10_updated.py` | Speech (Whisper-medium + Silero VAD), live camera with YOLOv8n, 30-second object memory, Phi-3-mini (INT8) picks the object for a fetch request |

### Setup

Run the commands in `installation_instructions_new1.txt` in PowerShell, from the repository folder.
The first run of `multimodal10_updated.py` downloads Silero VAD, so it needs internet.

### Run

```
python multimodal10_updated.py
```

Say e.g. "fido bring me something to drink", "fido go forward", "fido turn left", or "stop" to quit.

Model folders (`yolov8n_openvino_model/`, `phi3_openvino_int8/`, `whisper-medium-ov/`) are created
by the setup script and are not stored in this repository.
