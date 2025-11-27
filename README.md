# FAITH: Factuality & Hallucination Detection Framework
- A complete hallucination detection system. It is a modular framework for *detecting and quantifying hallucinations* in Large Language Model (LLM) outputs using:
  - Intrinsic model signals
  - Self & cross-consistency checks
  - Lightweight external verification via retrieval
  - A unified hallucination score over answers and spans.

---

## Repository Structure:

> Adjust names if you’ve changed them, but this is the intended layout. :contentReference[oaicite:1]{index=1}

```text
faith-framework/
├── src/
│   ├── generation/      # LLM generation, prompts, sampling
│   ├── signals/         # Intrinsic signal extractor
│   ├── consistency/     # Self- & cross-consistency analysis
│   ├── verification/    # Retrieval, NLI / evidence scoring
│   ├── scoring/         # Aggregates scores into hallucination score
│   ├── benchmark/       # Synthetic & weakly supervised benchmarks
│   └── utils/           # Shared helpers, config, logging
├── requirements.txt     # Python dependencies
├── README.md            # This file for basic information.
