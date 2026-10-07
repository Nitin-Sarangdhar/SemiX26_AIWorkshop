import os
import shutil
import subprocess
import sys
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
local_dir = "qwen2.5-0.5b-instruct-local"
export_dir = "qwen2.5-0.5b-instruct-ov"
weight_format = "int8"   # 0.5B is small: INT8 (~0.5 GB) keeps accuracy, INT4 loses noticeably more

print(f"1. Downloading {model_id} to {local_dir}...")

# Already-downloaded files are skipped, so re-running is cheap
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
)

# Qwen2 is supported by Optimum out of the box - no config.json patching needed
# (unlike Param-1, whose custom model_type has to be changed to "llama")

print(f"2. Triggering Optimum OpenVINO compilation ({weight_format})...")

# Find optimum-cli next to this Python, so it works even when the
# virtual environment is not activated
scripts_dir = os.path.dirname(sys.executable)
optimum_cli = shutil.which("optimum-cli", path=scripts_dir) or shutil.which("optimum-cli") or "optimum-cli"

command = [
    optimum_cli, "export", "openvino",
    "--model", local_dir,
    "--task", "text-generation-with-past",
    "--weight-format", weight_format,
    export_dir
]

subprocess.run(command, check=True)
print(f"\nDone! Your {weight_format.upper()} OpenVINO model is ready in: {export_dir}")
