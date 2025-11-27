
from generation.generator import LLMGenerator
from signals.intrinsic import IntrinsicSignalAnalyzer
from consistency.consistency_checker import ConsistencyChecker
from verification.retriever import EvidenceRetriever
from verification.evidence_scorer import EvidenceScorer
from scoring.aggregator import ScoreAggregator
from utils.text_processing import extract_claims

def run(query):
    llm = LLMGenerator()
    answer, probs = llm.generate(query)

    S_int = IntrinsicSignalAnalyzer().compute(probs)
    S_con = ConsistencyChecker().compute(query)
    evidence = EvidenceRetriever().retrieve(answer)
    claims = extract_claims(answer)
    S_evd = EvidenceScorer().score(claims, evidence)
    score = ScoreAggregator().aggregate(S_int, S_con, S_evd)

    return answer, score

if __name__=='__main__':
    q = input('Enter query: ')
    ans, sc = run(q)
    print(ans)
    print("Hallucination Score:", sc)
