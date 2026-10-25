"""One-off: merge adapters/m1b_mix7b into the bf16 7B base and save it unquantized
(models/m1b_mix7b-bf16), so vLLM can serve it with online FP8 as a check on the AWQ step."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "adapters/m1b_mix7b").merge_and_unload()
model.save_pretrained("models/m1b_mix7b-bf16")
AutoTokenizer.from_pretrained(BASE).save_pretrained("models/m1b_mix7b-bf16")
print("wrote models/m1b_mix7b-bf16")
