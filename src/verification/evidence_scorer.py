
from sentence_transformers import SentenceTransformer, util

class EvidenceScorer:
    def __init__(self):
        self.embed = SentenceTransformer('all-MiniLM-L6-v2')

    def score(self, claims, evidence):
        if not evidence:
            return 1.0
        scores=[]
        for c in claims:
            c_emb = self.embed.encode(c)
            e_emb = self.embed.encode(evidence)
            sim = util.cos_sim(c_emb, e_emb).max().item()
            scores.append(sim)
        return 1 - max(scores)
