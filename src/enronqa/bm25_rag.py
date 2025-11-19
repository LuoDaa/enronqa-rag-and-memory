"""
Minimal BM25 + LLM RAG baseline on EnronQA.

- builds a BM25 index over emails
- retrieves top-k emails for a question
- formats a prompt and calls an LLM (you can plug in your own)
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

from rank_bm25 import BM25Okapi

from .data import load_enronqa


def simple_tokenize(text: str) -> List[str]:
    """Naive whitespace tokenizer; you can replace this with spaCy, etc."""
    return text.lower().split()


@dataclass
class RetrievedDocument:
    doc_id: int
    score: float
    text: str


class LLM(Protocol):
    """Protocol for a simple text-to-text LLM."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        ...


class BM25EnronRetriever:
    """
    BM25 retriever over EnronQA emails (one document per email).
    """

    def __init__(self, documents: Sequence[str]):
        self.documents: List[str] = list(documents)
        tokenized_docs = [simple_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            RetrievedDocument(i, float(scores[i]), self.documents[i])
            for i in top_indices
        ]


class SimpleRAGPipeline:
    """
    Very small RAG pipeline:

    - retrieve top-k emails with BM25
    - build a prompt that contains these emails
    - call an LLM to answer the question
    """

    def __init__(
        self,
        retriever: BM25EnronRetriever,
        llm: LLM,
        system_prompt: Optional[str] = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions based only on the "
            "given email excerpts. If the context is not sufficient, say you are "
            "not sure instead of guessing."
        )

    def build_prompt(
        self,
        question: str,
        docs: List[RetrievedDocument],
    ) -> str:
        parts: List[str] = [self.system_prompt, "\n\n"]
        for i, doc in enumerate(docs, start=1):
            parts.append(f"[Email {i}]\n{doc.text}\n\n")
        parts.append(f"Question: {question}\nAnswer:")
        return "".join(parts)

    def answer(
        self,
        question: str,
        k: int = 5,
        **llm_kwargs,
    ) -> Tuple[str, List[RetrievedDocument]]:
        docs = self.retriever.search(question, k=k)
        prompt = self.build_prompt(question, docs)
        answer = self.llm.generate(prompt, **llm_kwargs)
        return answer, docs


# --- Example OpenAI client (optional) ---------------------------------------


class OpenAIChatLLM:
    """
    Minimal wrapper around the OpenAI Chat Completions API.

    You need:
        pip install openai
        export OPENAI_API_KEY=...
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()


# --- CLI demo ----------------------------------------------------------------


def build_bm25_from_split(split: str = "dev") -> BM25EnronRetriever:
    """
    Build a BM25 index over the emails in a given split.

    For quick experiments, using the 'dev' split is usually enough.
    """
    ds = load_enronqa(split=split)
    emails = [row["email"] for row in ds]
    return BM25EnronRetriever(emails)


def demo():
    """
    Small end-to-end demo:

    - builds BM25 index over the dev split
    - asks a question using GPT via OpenAI API
    """
    retriever = build_bm25_from_split("dev")
    llm = OpenAIChatLLM()
    rag = SimpleRAGPipeline(retriever, llm)

    question = input("Enter a question about the Enron emails: ")
    answer, docs = rag.answer(question, k=5, max_tokens=256, temperature=0.0)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Top-5 retrieved emails (truncated) ===")
    for i, d in enumerate(docs, start=1):
        snippet = d.text[:200].replace("\n", " ")
        print(f"[{i}] score={d.score:.3f}  {snippet}...")


if __name__ == "__main__":
    demo()
