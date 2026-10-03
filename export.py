import os
import json
import subprocess
from huggingface_hub import snapshot_download

model_id = "bharatgenai/Param-1-2.9B-Instruct"
local_dir = "param-1-local"
export_dir = "param-1-ov-int4"

print(f"1. Downloading {model_id} to {local_dir}...")

# Already-downloaded files are skipped, so re-running is cheap
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)

print("2. Patching config.json for Optimum compatibility...")

# Param-1 is Llama-architecture, but its config declares a custom model_type
# ("parambharatgen") that Optimum doesn't recognise. Point it at "llama" so
# the OpenVINO exporter can use its built-in Llama export config.
config_path = os.path.join(local_dir, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

if config.get("model_type") != "llama":
    print(f"   model_type: {config.get('model_type')!r} -> 'llama'")
    config["model_type"] = "llama"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
else:
    print("   config.json already patched, skipping.")

print("3. Triggering Optimum OpenVINO compilation...")

# We add the --task flag here
command = [
    "optimum-cli", "export", "openvino",
    "--model", local_dir,
    "--task", "text-generation-with-past",
    "--weight-format", "int4",
    "--trust-remote-code",
    export_dir
]

subprocess.run(command, check=True)
print(f"\nDone! Your INT4 OpenVINO model is ready in: {export_dir}")
