# SemiX26 AI Workshop - FIDO

FIDO is a voice-controlled "fetch" robot demo that runs entirely on an Intel AI PC with
OpenVINO: it listens for a spoken command, looks through the camera, and uses a small
language model to decide which visible object to bring.

## FIDO Mk3

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned Python packages |
| `export_mk3.py` | Downloads `Qwen/Qwen2.5-0.5B-Instruct` and exports it to OpenVINO INT8 (`qwen2.5-0.5b-instruct-ov/`, about 0.5 GB) |
| `test_mk3.py` | Sanity test: loads the exported model and asks it two questions |
| `fido_mk3.py` | FIDO with a live HUD window |

**How FIDO Mk3 works**

- **Speech:** Whisper-base + Silero VAD. Every command must start with "fido".
- **Vision:** YOLOv8n on the camera feed; FIDO only considers objects detected in the last second.
- **Decision:** the spoken sentence (verbatim) and the list of objects in view go to Qwen2.5-0.5B-Instruct,
  which replies with the object to fetch, or "none". There are no keyword rules for choosing objects.
- **Motion (simulated):** forward / back = 1 m, left / right = 90° turn, return, home; a fetch runs
  NAVIGATE → ALIGN → GRASP → RETURN → DELIVER and ends at the start position.
- **One command at a time:** while FIDO is busy, only "fido stop" is accepted.
- **Hardware:** YOLO, Whisper and Qwen run on the Intel integrated GPU via OpenVINO, with CPU fallback.

### Setup

```
python -m venv .venv
.venv\Scripts\Activate.ps1            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install hf_xet
yolo export model=yolov8n.pt format=openvino
optimum-cli export openvino --model openai/whisper-base whisper-base-ov
python export_mk3.py
python test_mk3.py
python fido_mk3.py
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

Model folders (`yolov8n_openvino_model/`, `whisper-base-ov/`, `qwen2.5-0.5b-instruct-ov/`) are created by
the setup steps and are not stored in this repository.
