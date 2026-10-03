# SemiX26 AI Workshop - FIDO

FIDO is a voice-controlled "fetch" robot demo that runs entirely on an Intel AI PC with
OpenVINO: it listens for a spoken command, looks through the camera, and uses a small
language model to decide which visible object to bring.

## Versions

Each version is a tagged commit - browse older ones under **Tags** (or `git checkout v1.0`).

| Tag | Main program | What changed |
|---|---|---|
| v1.0 | `multimodal10.py` + `brain_Int8.py` | Original code |
| v2.0 | `multimodal10_updated.py` | One program: continuous camera + YOLO, Silero voice detection, Whisper-medium, Phi-3 decides using recently seen objects |
| **v3.0** (this) | `fido_mk2.py` | Param-1 (2.9B, INT4) replaces Phi-3, live HUD, simulated motors and fetch missions |

## Version 3.0 - FIDO Mk2

| File | Purpose |
|---|---|
| `steps.txt` | Installation and run instructions |
| `requirements.txt` | Pinned Python packages |
| `export.py` | Downloads `bharatgenai/Param-1-2.9B-Instruct` and exports it to OpenVINO INT4 (`param-1-ov-int4/`) |
| `test.py` | Quick check that the exported Param-1 model loads and answers |
| `fido_mk2.py` | FIDO with a live HUD window |

**How FIDO Mk2 works**

- **Speech:** Whisper-base + Silero VAD. Every command must start with "fido".
- **Vision:** YOLOv8n on the camera feed; FIDO only considers objects detected in the last second.
- **Decision:** the spoken sentence (verbatim) and the list of objects in view go to Param-1, which
  replies with the object to fetch, or "none". There are no keyword rules for choosing objects.
- **Motion (simulated):** forward / back = 1 m, left / right = 90° turn, return, home; a fetch runs
  NAVIGATE → ALIGN → GRASP → RETURN → DELIVER and ends at the start position.
- **One command at a time:** while FIDO is busy, only "fido stop" is accepted.
- **Hardware:** YOLO, Whisper and Param-1 run on the Intel integrated GPU via OpenVINO, with CPU fallback.

### Setup and run

Follow `steps.txt`. In short:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
yolo export model=yolov8n.pt format=openvino
optimum-cli export openvino --model openai/whisper-base whisper-base-ov
python export.py
python test.py
python fido_mk2.py
```

### Voice commands

| Say | FIDO does |
|---|---|
| fido &lt;your request&gt; | the LLM picks an object in view and FIDO fetches it |
| fido forward / back | drive 1 m, then wait |
| fido left / right | turn 90°, then wait |
| fido return | U-turn and drive back to the start |
| fido home | go to the start, face forward, reset position |
| fido stop | cancel the current task (while busy) / shut down (when idle) |

Model folders (`yolov8n_openvino_model/`, `whisper-base-ov/`, `param-1-ov-int4/`) are created by the
setup steps and are not stored in this repository.
