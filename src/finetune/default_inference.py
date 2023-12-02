from itertools import chain
from pathlib import Path
from typing import List, Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer,
                          default_data_collator)

from datasets import load_dataset

from src.finetune.pipeline import ConditionalLanguageModel, get_loaders

MODEL_NAME = "t5-small"
# MODEL_NAME = "Salesforce/codet5p-220m"


def make_texts_to_tokens(text: str, tokenizer: Any) -> torch.Tensor:
    tokens = tokenizer(
        "Solve this question: " + text + " Solution:",
        return_tensors="pt",
    ).input_ids
    return tokens


def verify_correctness():
    model = ConditionalLanguageModel(model_name=MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    input_string = "What is the capital city of South Korea?"
    tokens = make_texts_to_tokens(input_string, tokenizer)
    outputs = model.generate(
        input_ids=tokens
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main():
    model = ConditionalLanguageModel(model_name=MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    valid_loader = get_loaders(batch_size=1, model_name=MODEL_NAME, split="valid")

    for batch in valid_loader:
        print("Problem:")
        print(tokenizer.decode(batch["source_ids"][0]))
        outputs = model.generate(
            input_ids=batch["source_ids"],
            input_masks=batch["source_mask"]
        )
        print("Response:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        print("Expected Answer:")
        print(tokenizer.decode(batch["target_ids"][0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
