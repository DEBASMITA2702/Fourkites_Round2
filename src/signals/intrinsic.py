
import numpy as np

class IntrinsicSignalAnalyzer:
    def compute(self, probs):
        entropy = -np.sum([p * np.log(p + 1e-9) for p in probs])
        low = np.mean([1 if p < 0.25 else 0 for p in probs])
        return min(1.0, 0.5 * entropy + 0.5 * low)
