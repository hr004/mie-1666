from itertools import chain
from pathlib import Path
from typing import List, Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer,
                          default_data_collator, GenerationConfig)

from datasets import load_dataset


class LanguageModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        assert model_name == "gpt2"
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


class ConditionalLanguageModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        assert model_name in [
            "t5-small",
            "t5-base",
            "Salesforce/codet5p-220m",
            "Salesforce/codet5p-770m-py",
        ]
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_masks: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=input_masks,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_masks,
        ).logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        input_masks: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_masks: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=input_masks,
            decoder_input_ids=decoder_ids,
            decoder_attention_mask=decoder_masks,
        ).loss

    def generate(
        self,
        input_ids: torch.Tensor,
        input_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        generation_config = GenerationConfig.from_pretrained("t5-small")
        # generation_config.max_length = 768
        generation_config.max_new_tokens = 1024
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=input_masks,
            generation_config=generation_config,
        )


class OptimizationDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: Any, max_length: int=768, data_path: str="../../datasets"):
        self.all_contents = []
        self.all_results = []
        for file in Path(data_path).rglob("*description.txt"):
            try:
                content = Path(file).read_text()
                output_file = "/".join(str(file).split("/")[:-1] + ["gptcode.py"])
                output_content = Path(output_file).read_text()
            except FileNotFoundError:
                continue

            self.all_contents.append("Write a Python Gurobi code to solve this problem. " + content + "Gurobi code: ")
            self.all_results.append(output_content)

        self.problem_size = len(self.all_contents)
        assert self.problem_size == len(self.all_results)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return self.problem_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        text = self.all_contents[index]
        ctext = self.all_results[index]

        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }


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


def get_loaders(
    model_name: str,
    batch_size: int,
    split: str = "train",
) -> torch.utils.data.DataLoader:
    assert model_name in [
        "t5-small",
        "t5-base",
        "Salesforce/codet5p-220m",
        "Salesforce/codet5p-770m-py",
    ]

    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = OptimizationDataset(tokenizer=tokenizer, max_length=768)
    if split in ["train", "eval_train"]:
        dataset = torch.utils.data.Subset(dataset, list(range(40)))
    else:
        dataset = torch.utils.data.Subset(dataset, [40, 41, 42, 43])

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
    )
