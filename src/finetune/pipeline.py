from itertools import chain
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          default_data_collator)

from datasets import load_dataset


class LanguageModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            from_tf=False,
            config=self.config,
            ignore_mismatched_sizes=False,
            trust_remote_code=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def get_dummy_loaders(
    model_name: str,
    batch_size: int,
    split: str = "train",
    indices: List[int] = None,
) -> torch.utils.data.DataLoader:
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 512

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split in ["train", "eval_train"]:
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=default_data_collator,
    )
