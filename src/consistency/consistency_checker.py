
from generation.generator import LLMGenerator
from sentence_transformers import SentenceTransformer, util

class ConsistencyChecker:
    def __init__(self, samples=3):
        self.gen = LLMGenerator()
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')
        self.samples = samples

    def compute(self, query):
        outs = [self.gen.generate(query)[0] for _ in range(self.samples)]
        emb = self.embed.encode(outs)
        sim = util.cos_sim(emb, emb).mean().item()
        return 1 - sim
