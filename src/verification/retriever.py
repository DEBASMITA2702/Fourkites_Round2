
from rank_bm25 import BM25Okapi

class EvidenceRetriever:
    def __init__(self):
        with open('data/sample_corpus.txt') as f:
            self.corpus = [l.strip() for l in f]
        self.tokens = [c.split() for c in self.corpus]
        self.model = BM25Okapi(self.tokens)

    def retrieve(self, answer, k=3):
        return self.model.get_top_n(answer.split(), self.corpus, n=k)
