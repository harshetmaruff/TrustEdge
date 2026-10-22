# Agent Instructions: AI Trust & Security Evaluation Layer

## Project Overview
You are assisting in building a **Trust-Evaluation Layer** for an AI assistant. The goal is to create a local, reproducible prototype that evaluates AI responses against a local knowledge base, classifying them as **Supported**, **Unsupported**, or requiring **Abstention** (insufficient evidence). The system must provide evidence for its claims and transparently report its own limitations.

**Status: all four phases are implemented and working.** The prototype is in `src/trustlayer/`, the corpus and question set are in `data/`, and the evaluation artefacts are in `artifacts/`.

## ⚠️ STRICT CONSTRAINTS (DO NOT VIOLATE)
1. **NO Paid APIs:** Do not use OpenAI, Anthropic, Google, or any other paid/external LLM APIs. Use only local, open-source models (e.g., HuggingFace `transformers`, `sentence-transformers`) or rule-based logic.
2. **NO Real Personal Data:** All generated datasets, documents, and questions must use entirely fictional names, companies, and data to ensure privacy and data minimization.
3. **NO Security Certification Claims:** Do not write code, comments, or documentation that claims this prototype is "secure," "certified," or "enterprise-ready." It is a research prototype.
4. **Honest Failure Reporting:** Do not hide errors or edge cases. If the model fails, the code must log the failure and the README must explain it.

## Tech Stack & Architecture
- **Language:** Python 3.9+
- **Environment:** Jupyter Notebook (preferred for prototyping) or modular Python scripts. *(The implementation uses modular Python scripts, which the guidelines allow.)*
- **Core Libraries:**
  - `sentence-transformers` (for local embeddings and cross-encoder NLI models).
  - `rank-bm25` or `scikit-learn` (for the simple keyword retrieval baseline).
  - `pandas` (for generating the evaluation table).
  - `datasets` (optional, for loading small local NLI models).

## Commands

```bash
make setup      # venv + CPU-only torch + deps + editable install (first run only)
make data       # regenerate the fictional corpus and the labelled question set
make evaluate   # baseline vs. trust layer -> artifacts/ (CSV, MD, metrics.json)
make sweep      # threshold trade-off -> artifacts/threshold_sweep.csv
make test       # unit tests; no model download needed (stub models)
```

`make evaluate` takes about a minute on a CPU and downloads ~300 MB of model weights on first use. There is no linter or type checker configured; `make lint` only byte-compiles the sources. **After changing the corpus, the questions or the layer, re-run `make evaluate` and update the numbers quoted in `README.md` from `artifacts/`** — the README contains a table, a metrics table, a threshold table and three worked examples, all of which must match the artefacts.

## Repository Layout

```
data/generate_corpus.py   source of truth for the corpus and the question set
data/documents/*.txt       four generated fictional documents
data/questions.jsonl       25 labelled questions
src/trustlayer/
  config.py                every threshold and model id in one place
  types.py                 Chunk, Question, Prediction, TrustLabel
  chunking.py  dataset.py  retrieval.py  embeddings.py
  statements.py            question -> the claims it makes
  answerability.py         does the evidence contain an answer of the right type?
  nli.py  trust_layer.py  baseline.py  reporting.py
scripts/                   generate_data, run_evaluation, threshold_sweep, measure_nli_channels
tests/                     unit tests; end-to-end tests use stub models
```

## Implementation Notes (the parts that are easy to get wrong)

These are measured facts about this codebase, not preferences. Each one cost a debugging cycle.

1. **The entailment channel is deliberately NOT the SUPPORTED gate.** `AGENTS.md`'s mapping (`ENTAILMENT → SUPPORTED`) was measured and does not hold with the local cross-encoder: an answer-agnostic hypothesis is not a proposition the model can confirm, so the entailment probability fires for passages that merely share the question's vocabulary. `scripts/measure_nli_channels.py` shows supported questions scoring 0.001–0.992 and abstain questions scoring 0.000–0.900 on the same channel — the ranges overlap, so the channel cannot separate them. The contradiction channel *does* separate cleanly (supported 0.000–0.025, unsupported 0.906–0.999). The layer therefore uses: embedding similarity as the relevance gate, an explicit **answerability check** for the supported/abstain decision, and the NLI **contradiction** channel for the unsupported verdict. Do not "restore" the AGENTS.md mapping without re-measuring.
2. **Only *asserted* claims may be checked for contradiction.** A wh-question asks, it does not assert, so it can never be contradicted. Feeding wh-forms such as *"The maximum declared value … is as described."* to the model manufactures contradictions on perfectly good passages (this alone produced four wrong `unsupported` verdicts). `to_claims()` marks each claim `asserted=True/False` for exactly this reason.
3. **The polar claim form that works is `"The text states that {rest}."`** — not the bare question and not `"It is the case that …"`. Measured on the corpus, the bare question scored 0.00–0.16 for contradiction on refuted questions, while the `"The text states that …"` form scored 0.987–0.999.
4. **Keep the scope of a conditional.** Stripping the leading conditional off *"If I return an unopened Orrin appliance after 20 days, do I get a full refund"* turns it into the false claim *"I get a full refund"*, which the model then contradicts against the restocking-fee paragraph. `split_claims(..., strip_subordinate=False)` keeps it for the contradiction check, while the answerability rules use `strip_subordinate=True` (the "20 days" is part of the scenario, not something the corpus has to state).
5. **The BM25 retriever must stem.** `retrieval.tokenize_query()` and `answerability` share `text_utils.simple_stem`; without it "claim" never matches "claims" and the retriever returns nothing. Do not re-introduce a second tokenizer — the retriever and the answerability rules have to agree on what the same word is.
6. **The candidate window is 6 chunks, not 3.** With `nli_candidates=3` the chunk that actually answers *"How long do I have to file a damage claim?"* is not retrieved at all, and the layer cites the adjudication paragraph instead.
7. **The similarity floor protects the contradiction channel.** Irrelevant passages can produce spurious contradictions (0.983 on an unrelated Lumen Bay paragraph). Those fall below `min_similarity` and are never scored. Lowering the floor admits more false refutations as well as more false support — see the sweep table in the README.
8. **`Confidence_Score` is per-label**: contradiction probability for `unsupported`, answer-alignment similarity for `supported`, neutrality probability for `abstain`. It is not a calibrated probability.
9. **The cited evidence is the passage the verdict came from**, which is not always the passage a reader would want. Retrieval relevance and answer-bearingness are different things, and all three available rankers (lexical overlap, embedding similarity, NLI entailment) prefer the wrong passage for `q21`. The obligation-matching rule in `trust_layer._obligation_match` is what rescues it.

## Dataset Conventions

- `category` is why the question was written: `supported`, `unsupported`, `abstain`, `edge_case`.
- `gold_label` is the verdict the layer should return, and it is always one of the three verdicts — an `edge_case` question is still scorable.
- A question whose premise is false is `unsupported` even though the corpus does answer it ("no"). This is a convention, stated in the README so readers can disagree knowingly.
- `reference_evidence` is documentation for humans only. No component of the pipeline reads it.

## Code Style & Structure
- Use strict type hinting (`typing` module) for all functions.
- Include clear docstrings explaining the *trust logic*, not just what the code does.
- Keep functions small and modular (e.g. `load_data()`, `run_baseline()`, `evaluate_trust()`, `generate_report()`).
- Use `logging` instead of `print()` for execution traces.

## Agent Workflow for Generating Code
1. **Check Constraints:** Ensure no external APIs or PII are included.
2. **Change the smallest thing that can answer the question.** Most questions are answered by editing `data/generate_corpus.py` (corpus or labels) or one module under `src/trustlayer/`.
3. **Re-run `make evaluate`** and read `artifacts/metrics.json` before drawing any conclusion. Re-run `make sweep` and `scripts/measure_nli_channels.py` if you touched a threshold or the claim rewriting.
4. **Update `README.md` from the artefacts.** Every number, table row and worked example in it has to match the regenerated artefacts.
5. **Document:** keep the "Limitations and failure cases" section current. A change that fixes a failure mode should remove the corresponding limitation, and a change that introduces one should add it.

## README.md Requirements
When generating or updating the `README.md`, ensure it includes:
- Project overview and how to run locally.
- The evaluation table (Baseline vs. Trust Layer).
- A dedicated **Limitations** section detailing:
  - Where the local model struggles (e.g., complex multi-hop reasoning).
  - How threshold tuning affects false abstentions vs. false hallucinations.
  - A statement confirming the use of fictional data and lack of formal security certification.
