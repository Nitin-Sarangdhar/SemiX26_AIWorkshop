import openvino_genai as ov_genai
import json
import os
import time

# --- Configuration ---
MODEL_PATH = "phi3_openvino_int8/"
SENSES_FILE = "fido_senses.json"

def initialize_brain():
    print(f"[Brain] Loading Phi-3-mini INT8 via GenAI Pipeline...")
    # The Pipeline handles the CPU compilation and KV-cache state internally
    try:
        pipe = ov_genai.LLMPipeline(MODEL_PATH, "CPU")
        print("  ✅ Brain Pipeline Ready.")
        return pipe
    except Exception as e:
        print(f"  ❌ Failed to load pipeline: {e}")
        return None

def process_fido_logic(pipe):
    if not os.path.exists(SENSES_FILE):
        print(f"  ⚠️ Error: {SENSES_FILE} not found.")
        return

    with open(SENSES_FILE, 'r') as f:
        data = json.load(f)

    command = data.get("command", "")
    seen_objects = data.get("seen", [])
    
    print(f"\n[Input Received]")
    print(f"  > Speech: \"{command}\"")
    print(f"  > Vision: {seen_objects}")

    # Build a Structured Chat Prompt
    # This forces Phi-3 into "Instruction Following" mode
    scene_str = ", ".join(seen_objects)
    
    prompt = f"""<|system|>
You are a helpful robot. I will give you a list of objects I see and a command. 
Respond with ONLY the name of the ONE object from the list that best fits the command.
Objects: {scene_str}<|end|>
<|user|>
{command}<|end|>
<|assistant|>"""

    print("[Brain] Reasoning (INT8)...")
    start_time = time.perf_counter()

    # Increase max_new_tokens slightly to allow for full word retrieval
    # temperature=0.0 makes the output deterministic (best for robotics)
    result = pipe.generate(prompt, max_new_tokens=10, do_sample=False)
    
    latency = (time.perf_counter() - start_time) * 1000
    
    # Clean up: strip whitespace and punctuation
    decision = result.strip().lower().split('\n')[0].replace('.', '').replace('!', '')

    print(f"  ✅ Decision: {decision.upper()} ({latency:.2f}ms)")
    print(f"\n[FIDO]: Fido will bring the {decision}!\n")

if __name__ == "__main__":
    fido_pipe = initialize_brain()
    if fido_pipe:
        process_fido_logic(fido_pipe)
