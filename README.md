# SemiX26 AI Workshop - FIDO

FIDO is a voice-controlled "fetch" robot demo that runs entirely on an Intel AI PC with
OpenVINO: it listens for a spoken command, looks through the camera, and uses a small
language model to decide which visible object to bring.

## Version 1.0 - original code

| File | Purpose |
|---|---|
| `installation_instructions_new1.txt` | PowerShell setup script: creates the environment, installs OpenVINO / Ultralytics, exports YOLOv8n, Phi-3-mini (INT8) and Whisper-tiny |
| `multimodal10.py` | Speech (Whisper-tiny) + vision (YOLOv8n). Moves on "fido forward / back / left / right"; on "fetch ..." it captures a frame and writes the command and the detected objects to `fido_senses.json` |
| `brain_Int8.py` | Reads `fido_senses.json` and asks Phi-3-mini (INT8, CPU) which of the seen objects fits the command |

### Setup

Run the commands in `installation_instructions_new1.txt` in PowerShell, from the repository folder.

### Run

```
python multimodal10.py     # speak a command, e.g. "fetch something to drink"
python brain_Int8.py       # Phi-3 picks the object from fido_senses.json
```

Model folders (`yolov8n_openvino_model/`, `phi3_openvino_int8/`, `whisper-tiny-ov/`) are created
by the setup script and are not stored in this repository.
