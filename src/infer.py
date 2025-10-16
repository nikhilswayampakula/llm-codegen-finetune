from __future__ import annotations
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CodeGenService:
    def __init__(self, model_dir='models/lora-gpt2', max_new_tokens=120):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def generate(self, prompt: str, temperature=0.2, top_p=0.95):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        out = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=self.max_new_tokens)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):]
