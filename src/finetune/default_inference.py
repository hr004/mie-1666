from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from src.finetune.pipeline import construct_model, get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def make_texts_to_tokens(text: str, tokenizer: Any) -> torch.Tensor:
    tokens = tokenizer(
        "Solve this question: " + text,
        return_tensors="pt",
    ).input_ids
    return tokens


def verify_correctness(model_name: str):
    model = construct_model(model_name=model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_string = "What is the capital city of South Korea?"
    print(f"Input: {input_string}")
    tokens = make_texts_to_tokens(input_string, tokenizer).to(DEVICE)
    generation_config = GenerationConfig.from_pretrained("t5-small")
    generation_config.max_new_tokens = 2048
    outputs = model.generate(input_ids=tokens, generation_config=generation_config)
    outputs = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("  ", "")
    print(f"Output: {outputs}")


def main(model_name: str):
    model = construct_model(model_name=model_name).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    valid_loader = get_loaders(batch_size=1, model_name=model_name, split="valid")
    generation_config = GenerationConfig.from_pretrained("t5-small")
    generation_config.max_new_tokens = 2048

    for batch in valid_loader:
        print("Problem:")
        print(tokenizer.decode(batch["input_ids"][0]))
        outputs = model.generate(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            generation_config=generation_config,
        )
        print("Response:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        print("Expected Answer:")
        print(tokenizer.decode(batch["raw_labels"][0], skip_special_tokens=True))


if __name__ == "__main__":
    model_name_lst = ["t5-small", "t5-base", "Salesforce/codet5p-220m", "Salesforce/codet5p-770m-py"]
    for mn in model_name_lst:
        print(mn)
        verify_correctness(mn)
    # main()
