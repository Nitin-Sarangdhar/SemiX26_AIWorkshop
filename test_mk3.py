"""Sanity test for FIDO Mk3: loads the exported Qwen2.5-0.5B-Instruct model and asks it two questions."""
import sys
import openvino_genai as ov_genai

model_dir = "qwen2.5-0.5b-instruct-ov"


def chat(system, user):
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


pipe = None
for device in ("GPU", "CPU"):
    try:
        print(f"Loading OpenVINO model on {device}...")
        pipe = ov_genai.LLMPipeline(model_dir, device)
        break
    except Exception as e:
        print(f"  {device} failed: {e}")
if pipe is None:
    print("Could not load the model.")
    sys.exit(1)

print("\nGenerating response...")
answer = str(pipe.generate(chat("You are a helpful assistant.", "two+two"),
                           max_new_tokens=50, do_sample=False, apply_chat_template=False)).strip()
print("\nAssistant Output:\n", answer)

# The same kind of question FIDO asks
pick = str(pipe.generate(chat("You are FIDO, a home robot. Your camera currently sees these objects:\n"
                              "- person\n- cup\n- laptop\n\n"
                              "The user gives you a spoken command. Choose the ONE object from the list above "
                              "that best fulfils the command. Reply with only the object name, exactly as "
                              "written in the list. If none of the objects can fulfil it, reply none.",
                              "fido bring me something to drink"),
                         max_new_tokens=12, do_sample=False, apply_chat_template=False)).strip()
print("\nFIDO test - 'fido bring me something to drink' with person, cup, laptop in view:\n", pick)

if not answer:
    sys.exit(1)
