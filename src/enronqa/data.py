"""
Utilities for loading the EnronQA dataset from Hugging Face.

Dataset: MichaelR207/enron_qa_0922
Fields (simplified):
- email: str
- questions: list[str]
- rephrased_questions: list[str]
- gold_answers: list[str]
- user: str
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import Dataset, load_dataset


HF_DATASET_NAME = "MichaelR207/enron_qa_0922"


@dataclass
class QAExample:
    email: str
    question: str
    rephrased_question: str
    answer: str
    user: str


def load_enronqa(split: str = "train") -> Dataset:
    """
    Load one split of the EnronQA dataset.

    Valid splits (according to the dataset card):
    - "train"
    - "dev"
    - "test"
    """
    ds = load_dataset(HF_DATASET_NAME, split=split)
    return ds


def iter_qa_examples(
    ds: Dataset,
    use_rephrased: bool = False,
    include_email: bool = True,
    max_questions_per_email: Optional[int] = None,
) -> Iterable[QAExample]:
    """
    Iterate over individual QA examples.

    Each row in the dataset contains multiple questions per email.
    This helper flattens them into QAExample objects.
    """
    for row in ds:
        email_text = row["email"] if include_email else ""
        questions = row["rephrased_questions"] if use_rephrased else row["questions"]
        rephrased = row["rephrased_questions"]
        answers = row["gold_answers"]
        user = row["user"]

        n = len(questions)
        if max_questions_per_email is not None:
            n = min(n, max_questions_per_email)

        for i in range(n):
            yield QAExample(
                email=email_text,
                question=questions[i],
                rephrased_question=rephrased[i],
                answer=answers[i],
                user=user,
            )
