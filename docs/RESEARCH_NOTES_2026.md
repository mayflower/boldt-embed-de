# Research Notes — LLM-to-Embedding for Boldt-DC-350M

- **Date compiled:** 2026-05-28
- **Author:** Boldt-Embed-DE engineering
- **Scope:** Decision-relevant research for building causal + bidirectional German
  embedders and a German reranker on top of `Boldt/Boldt-DC-350M`.
- **Confidence legend:** `[VERIFIED 2026-05-28]` fetched from a primary source this
  session · `[LIT]` established in prior literature, not re-fetched this session ·
  `[MUST-VERIFY]` load-bearing assumption not yet confirmed against a primary source.

---

## 1. Base model: what we actually know

Source: Hugging Face model card `Boldt/Boldt-DC-350M`, fetched 2026-05-28.

- **License: `apache-2.0`** `[VERIFIED 2026-05-28]` — stated in the HF model metadata.
  This is the single most important release-gating fact; it means the *base weights* are
  permissively licensed. Still see ADR-001 for derivative-weight and data-license nuance.
- **Language:** German. **Training data:** German "Dense-Core" subset of FineWeb-2,
  ~**200B tokens**, multi-epoch on a heavily filtered corpus (coherence / information
  value / educational quality filters). `[VERIFIED 2026-05-28]`
- **Type:** *base* language model, explicitly **not** instruction-tuned ("not optimized
  for chat or instruction following; use standard text completion"). `[VERIFIED]`
- **Precision:** BF16. **Associated paper:** "Repetition over Diversity" (arXiv 2604.28075). `[VERIFIED]`
- **Architecture internals — RESOLVED `[VERIFIED 2026-05-29]`** by loading the weights on an
  RTX A6000: **LlamaForCausalLM**, hidden_size **1024**, **24** layers, vocab_size **32000**,
  max_position_embeddings **2048**, **~435M** total parameters (435,471,360). `AutoModel`
  returns `last_hidden_state` suitable for pooling.

### Resolved discrepancies (were open on 2026-05-28)
- **Name vs. size — RESOLVED:** ~435M total params. "350M" likely counts non-embedding
  params; the HF "0.5B" badge rounds 435M. Publish the honest figure (~435M total).
- **Hidden size vs. 1024 embedding dim — RESOLVED:** hidden_size is exactly **1024**, so the
  1024-d output and Matryoshka dims work **natively, no projection head required**.
- **Context length — RESOLVED:** **2048** tokens. Confirms the non-goal: **no** long-context
  (8k/32k) retrieval claim without a trained+evaluated context-extension phase.

---

## 2. Decoder-LLM → embedding: the two routes we will build

### Route A — Causal embedder (keep causal attention)
- Pool the **last non-pad token** (EOS / last-token) hidden state as the sequence embedding.
  This is the standard recipe for turning a causal LLM into an embedder without changing
  the attention mask, popularized by E5-Mistral-style "instruction + last token" models. `[LIT]`
  - Ref: "Improving Text Embeddings with Large Language Models" (E5-mistral), arXiv 2401.00368. `[LIT]`
- Pros: minimal architecture change, reuses the pretrained next-token machinery, cheap to
  start. Cons: last-token pooling can under-use early-sequence content; left-to-right only.

### Route B — Bidirectional embedder (LLM2Vec / MNTP)
LLM2Vec is the canonical recipe to convert a decoder-only LLM into a strong encoder, in 3 steps `[VERIFIED 2026-05-28]`:
1. **Enable bidirectional attention** — replace the causal mask with an all-ones mask so
   every token attends to every other token.
2. **MNTP (Masked Next-Token Prediction)** — adapt the model to the new bidirectional
   capability (blends masked-LM and next-token objectives) so representations actually use
   both-side context.
3. **Unsupervised contrastive learning** (SimCSE-style) before/with supervised contrastive.
- Refs: LLM2Vec, arXiv 2404.05961; code `github.com/McGill-NLP/llm2vec`. `[VERIFIED 2026-05-28]`
- Pooling for Route B: ablate **mean**, **EOS**, and optionally **latent-attention** pooling.

> Decision posture (see ADR-002): **build both**, gate the winner on German evaluation.
> A bidirectional model is generally a stronger *encoder*, but MNTP adaptation adds a
> training phase and merging complexity; the causal route is the cheaper baseline.

---

## 3. Instruction conditioning
- Asymmetric retrieval works best with a **query instruction** prepended to queries and the
  raw text (or a light template) for documents — Instructor / E5-instruct style. `[LIT]`
- Our `configs/training_causal.json` already encodes a German query instruction
  (`Instruct: … Query: {query}`). Keep task-type instructions for STS / classification /
  clustering distinct from the retrieval query instruction.
- Caveat: the base model is *not* instruction-tuned, so the "instruction" here is a
  representation prompt learned during contrastive training, not a chat instruction.

## 4. Matryoshka representation learning
- Train so that **prefixes** of the embedding (1024→…→64) remain useful, enabling cheaper
  storage/retrieval by truncation. `[LIT]` Ref: Matryoshka Representation Learning, arXiv 2205.13147.
- **Important framing** (carry into model cards): Matryoshka reduces *downstream vector size
  and vector-store cost*; it does **not** make the model itself smaller or faster at
  inference. Truncated vectors must be **re-normalized** (L2) before similarity.

## 5. Hard negatives (German-specific) and the reranker loop
- Contrastive training quality is dominated by **hard negatives**. Mine from multiple
  sources: BM25 lexical, the embedder itself (in-training), and **reranker-mined** negatives. `[LIT]`
- A **cross-encoder reranker** (query+doc encoded together → relevance score / `Ja`/`Nein`
  logit) serves three jobs: production reranking, hard-negative mining, and **teacher
  distillation** into the bi-encoder (margin-MSE / KL on scores). `[LIT]`
- German-specific hard-negative families to manufacture and stress-test (see synthetic-pair
  and data plans): **compounds**, **negation/double-negation**, **legal refs** (`§`, `Abs.`,
  `Satz`, `Nr.`), **dates/numbers/versions**, **regional variants** (DE/AT/CH, e.g.
  *Jänner*/*Januar*, `ß`/`ss`), and **entity disambiguation**.

## 6. Evaluation landscape (German)
- **MMTEB** — Massive Multilingual Text Embedding Benchmark, arXiv 2502.13595; community
  expansion of MTEB to 500+ tasks / 250+ languages, with German tasks included. `[VERIFIED 2026-05-28]`
- **GermanQuAD** (13,722 extractive QA pairs) and **GermanDPR** (dense passage retrieval set
  adapted from GermanQuAD). `mteb/germanquad-retrieval` is published in BEIR format on HF.
  `[VERIFIED 2026-05-28]` Ref: arXiv 2104.12741.
- **`mayflower/mteb-de`** — a German MTEB variant on GitHub; relevant local resource for a
  German-focused suite. `[VERIFIED 2026-05-28]`
- **Anti-overfitting rule:** public benchmarks are *held-out post-training* evaluation only.
  Use a separate internal dev set for train-time validation. Never tune repeatedly against
  public test labels (ADR-005, data-leakage plan).

## 7. Tooling
- **SentenceTransformers** for the trainer/loss/evaluator stack and for the
  SentenceTransformers-compatible export used by MTEB. `[LIT]`
- **MTEB** package for the real benchmark run (kept as a scaffold here; needs the trained
  model + extras). `[LIT]`

---

## 8. Decisions this research drives (forward links)
- ADR-001 base model & license — base weights `apache-2.0` confirmed; resolve weight/data
  derivative terms and the size-naming question.
- ADR-002 causal vs bidirectional — build both, decide on German eval.
- ADR-003 pooling — last-token (causal) vs mean/EOS/latent-attention (bi); confirm hidden size.
- ADR-005 benchmark protocol — MMTEB + German tasks, held-out only.

## 9. Open questions / MUST-VERIFY before training or release
1. ✅ RESOLVED 2026-05-29: Llama arch, 24 layers, hidden 1024, vocab 32000, ctx 2048.
2. ✅ RESOLVED: ~435M total params (publish ~435M, not 350M/0.5B without context).
3. ✅ RESOLVED: hidden size is 1024 → native 1024-d embedding, no projection head.
4. Tokenizer behavior on German compounds and `§`/umlauts (fertility, OOV) — still to profile.
5. License terms for any *real training data* we add (separate from base-weight license) — open.

## Sources
- Boldt/Boldt-DC-350M model card — https://huggingface.co/Boldt/Boldt-DC-350M (fetched 2026-05-28)
- LLM2Vec — https://arxiv.org/abs/2404.05961 · https://github.com/McGill-NLP/llm2vec
- E5-Mistral / LLMs as embedders — https://arxiv.org/abs/2401.00368
- Matryoshka Representation Learning — https://arxiv.org/abs/2205.13147
- MMTEB — https://arxiv.org/abs/2502.13595
- GermanQuAD / GermanDPR — https://arxiv.org/abs/2104.12741 · https://huggingface.co/datasets/mteb/germanquad-retrieval
- mayflower/mteb-de — https://github.com/mayflower/mteb-de
