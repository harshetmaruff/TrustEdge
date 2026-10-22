# TrustEdge — a local trust-evaluation layer for retrieval-augmented answers

TrustEdge answers one question: **should this assistant be believed?**

An AI assistant that retrieves a passage and writes an answer is not evidence of
anything. It can answer confidently from a passage that never contained the
answer, and it can answer confidently from a passage that says the opposite.
TrustEdge is a local, reproducible prototype of a *trust-evaluation layer*: it
takes a question and a local knowledge base, and returns one of three verdicts
— **supported**, **unsupported** or **abstain** — together with the exact
text it used as evidence.

The layer is deliberately conservative. It refuses to answer when the corpus
does not answer, and it flags a question whose premise the corpus refutes. The
baseline it is measured against does the opposite: it retrieves the best
matching passage and answers from it, whatever that passage actually says.

Everything runs locally. No hosted or paid inference endpoint is contacted at
any point: the only models used are two freely downloadable checkpoints
(`sentence-transformers/all-MiniLM-L6-v2` and
`cross-encoder/nli-deberta-v3-small`) that are fetched once from the
HuggingFace hub and then run on the CPU.

---

## Running it locally

```bash
make setup      # virtualenv + CPU-only torch + dependencies
make data       # regenerate the fictional corpus and the labelled question set
make evaluate   # run the baseline and the trust layer, write artifacts/
make test       # unit tests (no model download required)
```

`make setup` downloads roughly 300 MB of model weights on first use. The
evaluation itself takes about a minute on a CPU.

Artefacts written by `make evaluate`:

| File | Contents |
| --- | --- |
| `artifacts/evaluation_table.csv` | the full per-question table |
| `artifacts/evaluation_table.md` | the same table as markdown |
| `artifacts/evaluation_report.md` | metrics, table and highlighted examples |
| `artifacts/metrics.json` | the headline metrics |

Two further scripts support the analysis in this README:

```bash
python scripts/threshold_sweep.py        # the threshold trade-off table below
python scripts/measure_nli_channels.py   # how well each NLI channel separates
```

### Repository layout

```
data/
  generate_corpus.py      the fictional corpus and the labelled questions (source of truth)
  documents/*.txt         four generated source documents
  questions.jsonl         25 labelled questions
src/trustlayer/
  config.py               every threshold in one place
  types.py                Chunk, Question, Prediction, TrustLabel
  chunking.py             document -> retrievable chunks
  dataset.py              loading the corpus and the question set
  retrieval.py            BM25 keyword retriever
  embeddings.py           local sentence embeddings (the relevance gate)
  statements.py           question -> the declarative claims it makes
  answerability.py        does the evidence contain an answer of the right type?
  nli.py                  local cross-encoder entailment / contradiction
  trust_layer.py          the verdict logic
  baseline.py             the untrustworthy baseline
  reporting.py            the evaluation table and the metrics
scripts/                  generate_data, run_evaluation, threshold_sweep, measure_nli_channels
tests/                    unit tests; the end-to-end tests use stub models
```

---

## The evaluation

The corpus is four short fictional documents (a flow meter manual, a freight
policy, a research fellowship policy and a support service level agreement).
The evaluation set is 25 questions in four categories:

| Category | n | Meaning |
| --- | --- | --- |
| `supported` | 9 | the corpus states the answer |
| `unsupported` | 6 | the corpus refutes what the question asserts |
| `abstain` | 5 | the corpus is silent on the question |
| `edge_case` | 5 | multi-fact, conditional, or variant questions |

`Ground_Truth` is the verdict the layer should return. The category describes
*why* a question was written; a question can be an edge case and still have a
scorable gold label.

### Baseline vs. trust layer

The baseline answers from the top retrieved chunk no matter what. The trust
layer returns the verdict *and* the evidence it used.

| QID | Category | Question | Ground truth | Baseline | Trust layer | Confidence |
| --- | --- | --- | --- | --- | --- | --- |
| q01 | supported | What is the sampling rate of the VT-40 Flow Meter? | supported | supported | supported | 0.77 |
| q02 | supported | What transit time does Kestrel Freight quote for standard service… | supported | supported | supported | 0.63 |
| q03 | supported | How long is the standard warranty on Orrin appliances? | supported | supported | supported | 0.67 |
| q04 | supported | What is the maximum declared value for a single Kestrel Freight c… | supported | supported | supported | 0.60 |
| q05 | supported | What does the Lumen Bay Visiting Fellowship fund? | supported | supported | supported | 0.81 |
| q06 | supported | What is the Kestrel Freight base charge for the first item in a c… | supported | supported | supported | 0.61 |
| q07 | supported | What percentage of the Lumen Bay Doctoral Fellowship stipend is w… | supported | supported | supported | 0.58 |
| q08 | supported | How often must Lumen Bay fellows submit progress reports? | supported | supported | supported | 0.64 |
| q09 | supported | What does the VT-40 status code SF indicate? | supported | supported | supported | 0.50 |
| q10 | unsupported | Kestrel Express delivery is available in every zone. How long doe… | unsupported | supported | unsupported | 1.00 |
| q11 | unsupported | Are faulty appliances returned to Orrin subject to a 25 percent r… | unsupported | supported | unsupported | 0.99 |
| q12 | unsupported | Must Lumen Bay Doctoral Fellowship candidates have already comple… | unsupported | supported | unsupported | 1.00 |
| q13 | unsupported | A Lumen Bay fellow may travel for up to 21 consecutive days witho… | unsupported | supported | unsupported | 0.91 |
| q14 | unsupported | Does the VT-40 Flow Meter support wireless transmission at 5 Hz? | unsupported | supported | unsupported | 1.00 |
| q15 | unsupported | Can a Kestrel Freight customer ship livestock if it is declared i… | unsupported | supported | unsupported | 0.99 |
| q16 | abstain | How many items can be included in a single Kestrel Freight consig… | abstain | supported | abstain | 1.00 |
| q17 | abstain | Which courier service does Orrin Retail Group use for deliveries? | abstain | supported | abstain | 1.00 |
| q18 | abstain | How many Lumen Bay fellows may hold a Microgrant in the same year? | abstain | supported | abstain | 1.00 |
| q19 | abstain | What is the flow accuracy of the VT-40 above 30 litres per minute? | abstain | supported | abstain | 1.00 |
| q20 | abstain | How many staff does the Orrin support centre employ? | abstain | supported | abstain | 1.00 |
| q21 | edge_case | How long do I have to file a damage claim with Kestrel Freight? | supported | supported | supported | 0.60 |
| q22 | edge_case | If I return an unopened Orrin appliance after 20 days, do I get a… | supported | supported | supported | 0.68 |
| q23 | edge_case | Can I send lithium batteries to Kestrel if I pack them in a padde… | abstain | supported | abstain | 1.00 |
| q24 | edge_case | What is the maximum ambient temperature the VT-40W wireless varia… | abstain | supported | abstain | 1.00 |
| q25 | edge_case | Which Lumen Bay fellowship scheme provides the highest value per … | abstain | supported | **supported** | 0.51 |

`Confidence_Score` is the probability behind the returned verdict: the
contradiction probability for `unsupported`, the answer-alignment similarity
for `supported`, and the neutrality probability for `abstain`.

### Metrics

| Metric | Baseline | Trust layer |
| --- | --- | --- |
| Accuracy | 0.44 | **0.96** |
| Macro F1 over the three verdicts | 0.20 | **0.96** |
| Abstention rate | 0.00 | 0.28 |
| **False-support rate** (claims support the corpus does not give) | 0.56 | **0.04** |
| False-abstention rate (abstains when the corpus does answer) | 0.00 | 0.00 |
| Unsupported detection rate (recall on refuted questions) | 0.00 | **1.00** |

Per category (accuracy):

| Category | n | Baseline | Trust layer |
| --- | --- | --- | --- |
| supported | 9 | 1.00 | 1.00 |
| unsupported | 6 | 0.00 | 1.00 |
| abstain | 5 | 0.00 | 1.00 |
| edge_case | 5 | 0.40 | 0.80 |

The number that matters most is the false-support rate. The baseline claims
support for all 25 questions, and 14 of those claims are not supported by the
corpus. The trust layer claims support for 12 questions and is wrong about one
of them.

### Three highlighted examples

**1. Correctly supported with evidence** (`q02`)

- Question: *What transit time does Kestrel Freight quote for standard service
  to Zone A?*
- Ground truth: `supported`
- Baseline: `supported` — but its answer was *"Kestrel Freight operates a
  regional parcel network covering six domestic zones."*, which does not
  answer the question at all. The baseline cannot tell a correct answer from
  a fabricated one because it answers everything.
- Trust layer: `supported`, confidence 0.63
- Evidence: *Standard service is offered in every zone and is quoted at 3 working days for Zone A and 5 working days for Zones B through F.*

**2. Unsupported answer caught** (`q10`)

- Question: *Kestrel Express delivery is available in every zone. How long
  does it take to reach Zone F?*
- Ground truth: `unsupported` — the policy says Express runs only in Zones A,
  B and C.
- Baseline: `supported` — its answer was *"Express service is offered only in
  Zones A, B and C."*, a true fact that still does not answer the question
  (Zone F transit), and the baseline never noticed that the premise of the
  question is false.
- Trust layer: `unsupported`, confidence 1.00
- Evidence: *Consignments for Zones D, E and F cannot be sent using Express service.*
- Why: the question asserts a false premise, the cross-encoder scores 0.999
  for contradiction between that assertion and the passage, and the layer
  returns the passage that refutes it.

**3. Edge case** (`q24`)

- Question: *What is the maximum ambient temperature the VT-40W wireless
  variant is rated for?*
- Ground truth: `abstain` — the corpus gives the temperature rating for the
  VT-40 and mentions the VT-40W only as "a wireless variant … that transmits
  the same payload at 2 Hz". It never states the variant's rating.
- Baseline: `supported` — its answer was *"A wireless variant, the VT-40W,
  adds an embedded radio that transmits the same payload at 2 Hz."*, which says
  nothing about temperature. A reader would have to notice that the 0–45 °C
  rating in the next paragraph belongs to a different product.
- Trust layer: `abstain`, confidence 1.00
- Why: the identifier rule. The question names `VT-40W`; the passage that
  states a temperature never mentions `VT-40W`, and the passage that mentions
  `VT-40W` states no temperature. Neither candidate can answer the question, so
  the layer refuses.

---

## How a verdict is decided

```
question
  │
  ├─ BM25 retrieval over the chunked corpus  (top 6 chunks)
  │
  ├─ no chunk retrieved ──────────────────────► ABSTAIN   "no context retrieved"
  │
  ├─ embedding similarity < 0.35 ────────────► ABSTAIN   "the retrieved text is not about the question"
  │
  ├─ NLI contradiction on the asserted claims ≥ 0.50 ──► UNSUPPORTED
  │
  ├─ the evidence contains no answer of the expected type ──► ABSTAIN
  │
  └─ otherwise ───────────────────────────────► SUPPORTED  (+ the evidence snippet)
```

Three signals decide the verdict, and each one is measured rather than
assumed.

**The relevance gate** (embeddings). The question and each retrieved chunk are
embedded with a local MiniLM model and compared by cosine similarity. Below
the floor the layer abstains: it has found text that is not about the question.

**The contradiction channel** (NLI). The question is rewritten into the
declarative claims it makes, and the cross-encoder scores each claim against
each candidate chunk. Only claims that *assert* something are used here — a
wh-question asks, it does not assert, so it can never be contradicted. A
polar question ("Does X support Y?") and a question that states a false
premise ("Kestrel Express delivery is available in every zone. How long…")
both become checkable claims.

**The answerability check** (rules). A question asks for a *kind* of value: a
count, a duration, an amount, a frequency, a point in time, a person, a place,
an entity, a yes/no answer, or a description. The evidence has to contain a
value of that kind. "How many items fit in a consignment?" is not answered by a
paragraph that prices each item; "How long do I have to file a claim?" is not
answered by a paragraph that says how long adjudication takes. A question that
names a product variant (`VT-40W`) also needs that identifier in the evidence,
otherwise the layer is looking at a different product.

### Why the entailment channel is not the gate

`AGENTS.md` describes a trust layer that maps `ENTAILMENT → SUPPORTED`,
`CONTRADICTION → UNSUPPORTED`, `NEUTRAL → ABSTAIN`. This prototype keeps the
contradiction half of that mapping and replaces the entailment half, because
the measurement does not support it.

`scripts/measure_nli_channels.py` measures both channels over the 25
questions. The maximum probability of each label, grouped by the gold verdict.
The contradiction channel is shown twice: over the chunks the layer actually
scores (those that pass the relevance floor) and over every retrieved chunk.

| Gold verdict | contradiction (scored) | contradiction (all chunks) | entailment |
| --- | --- | --- | --- |
| supported (n=11) | 0.000 – 0.025 | 0.000 – 0.983 | 0.001 – 0.992 |
| unsupported (n=6) | 0.906 – 0.999 | 0.959 – 0.999 | 0.000 – 0.190 |
| abstain (n=8) | 0.000 – 0.001 | 0.000 – 0.999 | 0.000 – 0.900 |

The contradiction channel separates cleanly: a refuted question scores above
0.90, and every other question below 0.03. The two "all chunks" columns show
what the relevance floor is doing — without it, an unrelated passage produces a
0.983 refutation for a perfectly good question, and a 0.999 one for a question
the corpus never answers.

The entailment channel does **not** separate supported from abstain — the two
ranges overlap almost completely. The reason is structural: to test whether a
passage answers a question, the hypothesis has to name the answer, and the
layer does not know the answer. A hypothesis such as *"The maximum declared
value for a single consignment is as described."* is not a proposition the
model can confirm, so the entailment probability collapses; when the question's
wording is aligned with the passage's vocabulary the same hypothesis does fire
(0.86 – 0.99), but it fires just as readily for a passage that is merely
*about* the question's topic and states no answer.

AGENTS.md allows the implementation to be "a local NLI pipeline **or** strict
embedding similarity threshold". This prototype uses the cross-encoder where
it is reliable (detecting a refuted claim), an explicit and inspectable
answerability check where it is not (deciding whether the corpus contains an
answer), and the embedding similarity as the relevance gate. The entailment
probabilities are still computed and reported in the evaluation table
(`NLI_Entailment`) so the behaviour stays visible.

---

## Limitations and failure cases

### Where the local model struggles

**Multi-hop questions are not detected.** The one question the layer gets
wrong is `q25`: *"Which Lumen Bay fellowship scheme provides the highest value
per month to the recipient?"* The corpus names three schemes with their values
and durations, so a comparison is computable — but the comparison basis is not
stated ("up to 5,000 credits" is a cap, not a value; the visiting scheme has no
stipend at all), and the layer answers `supported` with a passage that does
not contain the comparison. Every rule in the answerability check is
single-passage; nothing in the prototype reasons across passages. **A
superlative or a comparison across documents will be answered rather than
abstained.**

**The answerability check is rule based and English shaped.** It recognises
"how many", "how long", "how much", "how often", "when", "who", "where",
"which", and yes/no questions, and it looks for numbers, units, durations,
frequencies, dates, role nouns and place nouns. A question that asks for a
value in a way this does not model — a date range, a percentage range, a
counted list, a non-English phrasing — is treated as a description question
and will be answered if the passage shares enough vocabulary. The unit
families are short lists; a quantity expressed in an unlisted unit will not be
recognised as a quantity.

**Refuting a claim depends on the corpus stating the opposite explicitly.**
`q12` asks whether candidates must have completed a PhD. The corpus states
eligibility exclusions directly, and the contradiction fires at 0.998. A
policy that excluded doctoral-degree holders only by implication — through a
rule about enrolment duration, say — would not be refuted; the layer would
answer `supported` or `abstain` instead. The layer detects *stated*
refutations, not implied ones.

**The cited evidence is the passage the verdict came from, which is not
always the passage a reader would want to see.** For a `supported` verdict it
is the best-matching answerable passage; for an `unsupported` verdict it is the
passage the cross-encoder refuted the claim against. When several passages
refute a claim almost equally well, the one with the highest contradiction
probability wins: for `q13` that is the microgrant paragraph rather than the
travel-approval paragraph, even though the latter is the passage that a
reader would want to see.

Retrieval relevance and answer-bearingness are also not the same thing. For
`q21` ("How long do I have to file a damage claim?") the corpus contains both
the 14-day submission window and the 5-working-day adjudication window. The
layer cites the submission window, but only because an obligation-matching
rule prefers a passage that states an obligation over one that merely states
a duration: lexical overlap, embedding similarity and the NLI entailment score
all prefer the *wrong* passage on that question. That is a direct measurement
of how weak these small models are at telling a passage that answers a
question from a passage that is merely about it.

**The NLI model is small.** `cross-encoder/nli-deberta-v3-small` is a 22M
parameter model. It is reliable on the narrow, well-formed claims this
prototype produces, and it is not a general reasoning engine. Its
contradiction scores are close to bimodal on this corpus, which is convenient
but also a sign that it is reacting to surface framing rather than to logic.

### How threshold tuning trades false abstentions against false hallucinations

`scripts/threshold_sweep.py` sweeps both thresholds. The contradiction
threshold turned out to be insensitive: the scores are either below 0.03 or
above 0.95, so every value between 0.40 and 0.90 gives identical results. The
relevance floor is the knob that matters:

| `min_similarity` | Accuracy | False-support rate | False-abstention rate | Abstention rate |
| --- | --- | --- | --- | --- |
| 0.20 | 0.80 | 0.12 | 0.00 | 0.16 |
| 0.25 | 0.84 | 0.12 | 0.00 | 0.16 |
| 0.30 | 0.88 | 0.12 | 0.00 | 0.20 |
| **0.35** | **0.96** | **0.04** | **0.00** | **0.28** |
| 0.40 | 0.96 | 0.04 | 0.00 | 0.28 |
| 0.45 | 0.96 | 0.04 | 0.00 | 0.28 |
| 0.50 | 0.88 | 0.08 | 0.04 | 0.32 |
| 0.60 | 0.88 | 0.00 | 0.12 | 0.44 |

The two failure modes pull in opposite directions and the floor is where they
cross:

- **Too low** (≤ 0.30) and the layer lets through text that is on the right
  topic but answers a different question. The false-support rate doubles to
  0.12: these are hallucinations, and they are the expensive kind, because the
  answer is fluent and on-topic.
- **Too high** (≥ 0.50) and the layer starts refusing questions the corpus
  *does* answer. False support drops to zero at 0.60, but a quarter of the
  questions are now abstained, and the false-abstention rate reaches 0.12.

The 0.35–0.45 plateau is where neither error dominates. **These thresholds
were chosen on this 25-question set, so the plateau is partly an artefact of
the set size.** A real deployment would have to be tuned on a representative
sample, and the trade-off to watch is the one in the table: lowering the floor
buys recall with hallucinations, raising it buys safety with refusals.

One more interaction is worth knowing: the relevance floor also protects the
contradiction channel. `q22` produces a spurious 0.983 contradiction against
an unrelated Lumen Bay passage, but that passage falls below the floor, so the
layer never scores it. A lower floor admits more false refutations as well as
more false support.

### Other known weaknesses

- **The evaluation set is small (25 questions) and self-authored.** A single
  question moves the accuracy by 4 points. The corpus is written to be
  unambiguous, which flatters the rules; real policies are not.
- **The dataset's labels encode one particular reading.** A question whose
  premise is false is labelled `unsupported` even though the corpus does
  answer it ("no"). Under the entailment/contradiction mapping this is
  consistent, but it is a convention, not a fact, and it is stated here so a
  reader can disagree with it knowingly.
- **The baseline is a weak adversary.** It is extractive and non-adaptive: it
  always answers from the top chunk. A stronger baseline — one that generates
  fluent answers, or one that retrieves with the embedding model — would
  narrow the gap. The false-support rate is the metric that would move.
- **Extractive answers make the faithfulness check trivial.** A generative
  assistant produces text that is not a copy of the passage, and the
  entailment check between the answer and the evidence becomes a real test.
  This prototype has no generative component, so that check is not exercised.
- **`q23` is decided by a threshold, not by understanding.** "Can I send
  lithium batteries to Kestrel if I pack them in a padded case?" is answered
  `abstain` because the policy bans "loose lithium cells" and never defines
  "loose". The layer reaches the right verdict because the contradicting
  passage falls below the relevance floor, not because it noticed the
  ambiguity. A different threshold would give a different answer, and the
  verdict would be wrong in a different direction.

### Data and assurance statement

- **All data in this project is fictional.** The four documents describe
  invented companies (Vantis Instruments, Kestrel Freight, Lumen Bay Labs,
  Orrin Retail Group), invented products, invented policies and invented
  figures. No real organisation, policy, person or record is described, and
  the project contains no real personal data of any kind. The data minimises
  what it needs: names, amounts and dates are placeholders.
- **This is a research prototype, not a certified system.** It has not been
  security audited, it carries no certification of any kind, and nothing in
  this repository should be read as a claim that it is "secure",
  "certified", "hardened" or "enterprise-ready". The trust layer is a
  demonstration of a mechanism, evaluated on 25 questions against a
  self-authored corpus, with its failure modes listed above.
- **No paid or hosted API is used.** The only network access is the initial
  download of two open checkpoints from the HuggingFace hub; inference runs
  entirely on the local CPU.

---

## Reproducibility

- The corpus and the question set are generated by `data/generate_corpus.py`
  and committed, so an evaluation run does not depend on regenerating them.
- Every threshold lives in `src/trustlayer/config.py`; every model id is a
  local checkpoint named in that same file.
- `make test` runs the unit tests without downloading any model: the
  end-to-end tests of the decision flow use stub embedders and stub NLI
  scorers.
- The numbers in this README come from `artifacts/metrics.json`,
  `artifacts/threshold_sweep.csv` and `artifacts/nli_channels.json`, all
  regenerated by the three scripts named above.
