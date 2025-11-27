
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, query):
        inp = self.tokenizer(query, return_tensors='pt')
        out = self.model.generate(**inp, output_scores=True, return_dict_in_generate=True)
        text = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
        probs = [s.softmax(dim=-1).max().item() for s in out.scores]
        return text, probs
