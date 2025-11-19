# EnronQA: Notes on Personalized RAG over Private Documents

> [🇨🇳 Chinese version](./README.md) / 🇬🇧 English version
> Unofficial reading notes and (planned) experiments for  
> **“EnronQA: Towards Personalized RAG over Private Documents”**  
> arXiv: [2505.00263](https://arxiv.org/abs/2505.00263)

---


## Quickstart
#bash
git clone https://github.com/LuoDaa/enronqa-rag-and-memory.git

cd enronqa-rag-and-memory

pip install -r requirements.txt

# optional: install openai and set API key
pip install openai
export OPENAI_API_KEY=your_key_here

# run BM25 + LLM demo
python -m src.enronqa.bm25_rag

## Table of Contents

- [0. TL;DR](#0-tldr)
- [1. Background: Why a New RAG Benchmark?](#1-background-why-a-new-rag-benchmark)
- [2. What Is EnronQA?](#2-what-is-enronqa)
- [3. Dataset Construction](#3-dataset-construction)
  - [3.1 Email Filtering Pipeline](#31-email-filtering-pipeline)
  - [3.2 LLM-Based QA Generation Pipeline](#32-llm-based-qa-generation-pipeline)
- [4. Experiments](#4-experiments)
  - [4.1 Experiment 1: RAG Calibration](#41-experiment-1-rag-calibration)
  - [4.2 Experiment 2: RAG Pipeline Baselines](#42-experiment-2-rag-pipeline-baselines)
  - [4.3 Experiment 3: Memory vs Retrieval](#43-experiment-3-memory-vs-retrieval)
- [5. Practical Lessons for Engineers](#5-practical-lessons-for-engineers)
- [6. Repository Code](#6-repository-code)
  - [6.1 Layout](#61-layout)
  - [6.2 Install](#62-install)
  - [6.3 Minimal BM25 RAG Demo](#63-minimal-bm25-rag-demo)
  - [6.4 LoRA Memorization Demo](#64-lora-memorization-demo)
- [7. Quickstart](#7-quickstart)
- [8. Contributing / Feedback](#8-contributing--feedback)
- [9. License](#9-license)




## 0. TL;DR

- Most RAG benchmarks are built on **Wikipedia / public web data**.  
  Modern LLMs have already seen this content, so they can often answer
  correctly **without any retrieval**, which makes it hard to evaluate
  retrievers.

- This paper introduces **EnronQA**, a benchmark for
  **personalized RAG over private documents**, built from the
  Enron email corpus:

  - 103,638 filtered emails  
  - 528,304 QA pairs  
  - 150 user mailboxes (supporting multi-user / personalized settings)

- The dataset is created with a **multi-stage LLM pipeline** that
  automatically generates and filters questions using four properties:

  - **Specific**: the question uniquely points to one email  
  - **Objective**: different models agree on the answer  
  - **Grounded**: the answer cannot be guessed without reading the email  
  - **High-quality**: clear, answerable, and safe

- On EnronQA, **answer accuracy is almost linearly correlated with
  retrieval recall**, and “no-context” performance is low.  
  This makes it a much more faithful benchmark for testing RAG systems.

- The paper also compares **Long Context vs RAG vs LoRA memorization**:  

  - RAG is consistently the best.  
  - LoRA can memorize tens of thousands of facts and reach performance
    close to long-context prompting, making it a promising complement,
    not a replacement.

This repository contains a **full Chinese explanation** of the paper and (planned)
**code experiments** reproducing key ideas.

---

## 1. Background: Why a New RAG Benchmark?

**Retrieval-Augmented Generation (RAG)** is now a standard way to give LLMs
access to external knowledge:

1. Retrieve relevant documents from a corpus.
2. Stuff them into the prompt as context.
3. Let the LLM generate an answer.

This has two big advantages for real-world use:

- We do **not** bake private data directly into model parameters.
- It is much easier to update knowledge and enforce access control.

However, most existing RAG benchmarks (e.g., NaturalQuestions, TriviaQA)
are built from **Wikipedia or other public web sources**.  
Modern LLMs have likely seen these documents during pre-training.

Consequence:

- A model can achieve strong performance on these benchmarks **even with
  no retrieval**, simply by recalling memorized facts.
- Retriever quality and end-to-end QA accuracy become decoupled:
  improving the retriever may not change the score much, because the model
  already “knows” the answer.

Also, these benchmarks do **not** reflect typical enterprise scenarios,
where:

- Data is private (emails, tickets, internal docs).
- There are multiple users and per-user views of the corpus.
- We care about both retrieval quality and **personalization**.

**EnronQA** addresses these gaps.

---

## 2. What Is EnronQA?

EnronQA is a QA benchmark built from the public **Enron email corpus**.

- **Source corpus**: emails from Enron Corporation released through
  legal proceedings.
- **Original scale**: ~517k emails across 150 user mailboxes.
- **After filtering**: 103,638 clean, safe emails.
- **Generated QA pairs**: 528,304.
- **User structure**: questions are associated with 150 user mailboxes,
  enabling per-user and cross-user retrieval experiments.

You can think of EnronQA as a testbed for:

- RAG over **private, non-web documents** (emails).
- **Personalized / multi-user** retrieval.
- Comparing **retrieval vs parameter memorization** approaches.

---

## 3. Dataset Construction

### 3.1 Email Filtering Pipeline

The raw Enron emails are not directly usable as a benchmark.  
The authors apply several filtering steps:

1. **Deduplication**

   - Use minhash + Jaccard similarity to remove near-duplicate emails.
   - “Subset deduplication”: if one email is fully contained
     inside another (e.g., forwarded threads), remove the subset email.

2. **Quality Filtering**

   Similar in spirit to large-scale pre-training filters:

   - Remove emails that are too short / too long.
   - Filter by average token length, symbol ratio, trailing ellipsis, etc.
   - Drop automatic logs, templated reports, and generally uninformative text.

3. **Language Filtering**

   - Run a language ID model (e.g., fastText).
   - Keep only high-confidence English emails.

4. **NSFW / Toxicity Filtering**

   - Use a classifier trained on datasets like Jigsaw to filter NSFW
     or highly toxic content.
   - This protects privacy and makes the dataset safer to use.

5. **Alignment with ConcurrentQA**

   - If an email is used in the existing ConcurrentQA benchmark but
     would be removed by the filters, it is re-introduced.
   - This enables cross-benchmark experiments.

After these steps, 103,638 high-quality emails remain.

---

### 3.2 LLM-Based QA Generation Pipeline

From the filtered emails, the authors build QA pairs with a multi-stage
LLM pipeline:

1. **Initial Question Generation**

   - Input: one email + the list of questions already generated for it.
   - Model: Llama-3.1-70B Instruct.
   - Objective: generate a new, single-sentence, answerable, non-duplicate
     question grounded in that email.
   - Prompts and few-shot examples are optimized with DSPy / MIPROv2.

2. **Four Automatic Evaluations**

   Each proposed question is passed through four “unit tests”:

   - **Specificity**

     - Use an embedding retriever to find 10 similar emails:
       1 true email + 9 distractors.
     - Ask an LLM to pick which email can answer the question.
     - If it cannot consistently pick the true one, the question is
       not specific enough.

   - **Objectivity**

     - Two different model families (e.g., Llama 70B and Mixtral 8x7B)
       answer the question given the email.
     - A judge LLM checks whether the answers agree.
     - If not, the question is ambiguous or subjective.

   - **Groundedness**

     - Ask the same question **without** providing the email as context.
     - If a model can still answer correctly, the question is too
       guessable and not truly grounded in the email.

   - **Quality**

     - A separate judge model (Llama 70B + hand-crafted rules) checks
       whether the question is well-formed, clear, safe, etc.,
       based on manually labeled examples.

3. **Feedback & Refinement**

   - If a question fails any check, a feedback message is generated
     explaining what went wrong:

     - “Not specific enough, please rewrite so it only fits this email.”
     - “Models disagree, please clarify the question.”
     - “Too easy to guess, make it more dependent on the email.”
     - “Violates quality rules, here is why…”

   - The question is then **rewritten** using this feedback.
   - The process repeats up to 5 times; if the question still fails,
     it is discarded.

4. **Paraphrased Questions**

   For each accepted question, the pipeline generates a **paraphrased**
   version:

   - A new question is produced by Llama 70B.
   - The model answers both the original and paraphrased question.
   - A judge LLM checks that the answers are consistent.
   - Only then is the paraphrase kept.

This yields a large set of QA pairs with both original and paraphrased
questions, which is especially useful for **memorization experiments**
(see below).

---

## 4. Experiments

The paper runs three main sets of experiments.

### 4.1 Experiment 1: RAG Calibration  
**EnronQA vs NaturalQuestions vs TriviaQA**

Goal: see how strongly **retrieval recall** affects **answer accuracy**
on different datasets.

Setup:

- Datasets: NaturalQuestions (NQ), TriviaQA, EnronQA.
- Model: Llama-3.1-70B.
- Compare:

  1. **No-context baseline**: answer questions without any documents.
  2. **With context**: supply the *gold* supporting document.

- To simulate different levels of recall:

  - With some probability `p`, provide the correct document.
  - With probability `1 - p`, provide a random incorrect document.
  - This gives an effective Recall@1 = `p`.

Key findings:

- On **NQ / TriviaQA**:

  - The no-context baseline is already quite strong.
  - The model clearly remembers many answers from pre-training.
  - You need **very high recall** to significantly beat this baseline,
    especially on TriviaQA.

- On **EnronQA**:

  - No-context performance is low (emails are unseen data).
  - Accuracy improves **almost linearly** as recall improves.
  - Roughly: every +1% Recall@1 gives about +0.6% accuracy.

**Takeaway:**  
EnronQA is much more **retrieval-sensitive** than standard web/Wiki
benchmarks.  
It is a better choice when your goal is to truly compare retrievers.

---

### 4.2 Experiment 2: RAG Pipeline Baselines

The authors then build RAG baselines on EnronQA itself.

Components:

- **Retrievers**

  - BM25 (sparse; via PySerini).
  - ColBERTv2 (dense).

- **LLMs**

  - Llama-3.1-8B Instruct.
  - Llama-3.1-70B Instruct.
  - GPT-4o.

- **RAG Variants**

  - **No Query Rewrite**: use the original question as the retrieval query.
  - **Query Rewrite**: first let an LLM rewrite the question into a
    search-oriented query, then use that for retrieval.

Highlights:

1. **BM25 is surprisingly strong**

   - On EnronQA, BM25 achieves Recall@5 ≈ 87.5%.
   - With GPT-4o (no rewrite), answer accuracy reaches ≈ 81%.
   - Because questions are designed to be **specific** and include
     concrete entities from emails, BM25 benefits a lot from exact terms.

2. **Query rewrite helps little (or sometimes hurts)**

   - Unlike open-domain QA over noisy web text, the original questions
     are already focused and structured.
   - Rewriting them does not consistently improve BM25 and can even
     degrade performance.

3. **Bigger models still help**

   - For a fixed retrieval setup, performance increases from 8B → 70B
     → GPT-4o.

**Practical note:**  
In email / enterprise settings with entity-rich text, **BM25 remains
a very strong baseline**, and query rewriting is not always necessary.

---

### 4.3 Experiment 3: Memory vs Retrieval  
**Long Context vs RAG vs LoRA**

This case study asks:

> If we want a model to “remember” a set of facts,  
> should we use long context, retrieval, or LoRA memorization?

Setup:

- Select subsets of EnronQA QA pairs as “facts”.
- For each QA:

  - Original question: Q  
  - Paraphrased question: Q′  
  - Answer: A

- Treat **Q′ → A** as the fact.
- Vary the number of facts from **10 up to 20,000**.

Three approaches:

#### (1) Long Context

- Concatenate all facts into the prompt, e.g.:

  - Fact 1: Q′₁ → A₁  
  - Fact 2: Q′₂ → A₂  
  - …  

- Then append the test question Q and ask the model to answer.
- Because of context length limits, this only scales to
  about **1,000 facts**.

#### (2) RAG

- Index the facts using ColBERTv2.
- At test time, retrieve the top-k (e.g. 100) Q′–A pairs for question Q.
- Feed those facts as context and let the model answer.

#### (3) LoRA Memorization

- Base model: Llama-3.1-8B Instruct.
- Train LoRA adapters on Q′–A pairs, so the model “memorizes” them
  in its parameters.
- LoRA hyper-parameters (for each fact count):

  - LoRA rank r ∈ {8, 16, 32, 64, 128, 256, 512, 1024, 2048}
  - 10 epochs
  - learning rate 1e-4
  - LoRA α = 4 × r
  - dropout = 0.05
  - adapters applied to **all linear layers**

- At test time, input only the **original question Q** (no context),
  and see if the LoRA-adapted model can recall the correct answer.

Evaluation uses a strong LLM (Llama-3.1-70B) as a judge model.

#### Results

- **RAG wins overall**

  - Across all fact sizes (up to 20k), RAG achieves the best accuracy.
  - For this kind of “locate the right fact among many” problem,
    retrieval + reading is still the strongest approach.

- **LoRA ≈ Long Context for moderate scales**

  - For ≤ 1k facts, LoRA performance is often comparable to long context.
  - As the number of facts grows to 20k, LoRA remains reasonable but
    starts to degrade.

- **Long Context hits the context ceiling**

  - Long-context prompting cannot scale beyond a certain point,
    while RAG and LoRA can.

**Takeaway:**  

- Today, **RAG remains the most robust solution** for large knowledge
  sets over private data.
- **LoRA memorization** is a promising complement:
  - It can store thousands of facts in a small adapter.
  - Good candidate for “hot” but relatively stable knowledge.
- A combined strategy (LoRA for hot facts + RAG for long-tail) might
  be very effective in practice.

---

## 5. Practical Lessons for Engineers

Some key lessons from the paper (with a bit of opinionated commentary):

1. **Be careful when benchmarking RAG on web/Wiki datasets.**

   - If your dataset comes from Wikipedia or popular web pages,
     your model might already know the answers.
   - You could see great scores and terrible retrievers at the same time.

2. **Use unseen or private data when you want to evaluate retrieval.**

   - EnronQA shows that when the corpus is “new” to the model,
     accuracy tracks retrieval quality much more faithfully.

3. **LLM + unit tests is a powerful pattern for synthetic data.**

   A reusable recipe:

   1. Define explicit “unit tests” (specificity, groundedness, quality…).
   2. Let an LLM generate candidates.
   3. Run them through the tests.
   4. Reject or refine until candidates pass.

4. **LoRA is a viable way to store structured factual knowledge.**

   - You can store thousands of QA-style facts in a small adapter.
   - Good for high-frequency but slowly changing knowledge.
   - Should be viewed as a complement to RAG, not a replacement.

5. **Classical BM25 is still a beast for enterprise text.**

   - In entity-heavy domains like emails, sparse retrieval often performs
     extremely well.
   - Do not skip BM25 when building your baseline.

---

## 6. Repository Contents

This repo is **not** an official implementation of EnronQA.  
It is a personal project for understanding and experimenting with the paper.

Planned structure:

```text
enronqa-rag-and-memory/
├── README.md                # English notes (this file)
├── README_zh.md             # Chinese explanation (existing)
├── LICENSE                  # MIT
├── notebooks/
│   ├── 01_enronqa_overview.ipynb        # Dataset overview & stats (TODO)
│   ├── 02_bm25_rag_baseline.ipynb       # Minimal BM25 + LLM RAG demo (TODO)
│   └── 03_lora_memory_case_study.ipynb  # Small-scale LoRA experiments (TODO)
└── src/
    ├── rag/                 # RAG utilities (planned)
    └── lora/                # LoRA training scripts (planned)
