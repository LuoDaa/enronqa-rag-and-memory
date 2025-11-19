
"""
Train a LoRA adapter to memorize a subset of EnronQA QA pairs.

Simplified version of the memorization experiment in Section 6
of the EnronQA paper.
"""

import argparse
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


HF_DATASET_NAME = "MichaelR207/enron_qa_0922"


def build_facts_dataset(
    split: str,
    num_facts: int,
    seed: int = 42,
):
    """
    Build a dataset of `num_facts` text samples:

        "Question: <rephrased_question>\\nAnswer: <gold_answer>"
    """
    ds = load_dataset(HF_DATASET_NAME, split=split)
    ds = ds.shuffle(seed=seed)

    def to_text(row):
        texts: List[str] = []
        for q, rq, a in zip(
            row["questions"],
            row["rephrased_questions"],
            row["gold_answers"],
        ):
            question = rq or q
            texts.append(f"Question: {question}\nAnswer: {a}")
        return {"text": texts[0] if texts else ""}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["text"].strip()) > 0)

    if num_facts < len(ds):
        ds = ds.select(range(num_facts))

    return ds


def find_all_linear_names(model) -> List[str]:
    """
    Collect the names of all Linear submodules to use as LoRA targets.
    """
    import torch.nn as nn

    linear_cls = nn.Linear
    target_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            target_modules.add(name.split(".")[-1])

    return sorted(target_modules)


@dataclass
class LoraTrainConfig:
    model_name: str
    rank: int = 64
    lr: float = 1e-4
    num_facts: int = 1000
    max_length: int = 512
    batch_size: int = 4
    num_epochs: int = 3
    output_dir: str = "outputs/lora_memory"
    split: str = "train"
    seed: int = 42


def train_lora(cfg: LoraTrainConfig):
    print(f"Loading dataset {HF_DATASET_NAME} [{cfg.split}] ...")
    ds = build_facts_dataset(cfg.split, cfg.num_facts, seed=cfg.seed)

    print(f"Loaded {len(ds)} facts")

    print(f"Loading tokenizer and base model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    print("Finding linear modules for LoRA...")
    target_modules = find_all_linear_names(model)
    print(f"Will apply LoRA to {len(target_modules)} module types: {target_modules[:10]}...")

    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=4 * cfg.rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
        )

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Starting LoRA training...")
    trainer.train()

    print("Saving LoRA adapter to", cfg.output_dir)
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print("Done.")


def parse_args() -> LoraTrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter to memorize EnronQA QA pairs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base causal LM to adapt.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="LoRA rank r.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--num_facts",
        type=int,
        default=1000,
        help="Number of QA facts to memorize.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_memory",
        help="Where to save the LoRA adapter.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()
    return LoraTrainConfig(
        model_name=args.model_name,
        rank=args.rank,
        lr=args.lr,
        num_facts=args.num_facts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_lora(cfg)
