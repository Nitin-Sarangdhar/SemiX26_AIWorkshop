import warnings
import transformers
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

# Suppress the harmless tokenizer warning so it doesn't clutter your terminal
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")

model_dir = "param-1-ov-int4"

print("Loading OpenVINO model...")
model = OVModelForCausalLM.from_pretrained(
    model_dir, 
    device="GPU",
    trust_remote_code=True,
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)

# Reverted: Removed fix_mistral_regex=True
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "two+two"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)

print("\nGenerating response...")
outputs = model.generate(
    inputs,
    max_new_tokens=150,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print("\nAssistant Output:\n", response)