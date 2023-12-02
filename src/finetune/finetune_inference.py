from typing import Any

import torch
from transformers import AutoTokenizer

from src.finetune.pipeline import construct_model, get_loaders

MODEL_NAME = "t5-small"

# MODEL_NAME = "t5-base"
# MODEL_NAME = "Salesforce/codet5p-770m-py"

# MODEL_NAME = "Salesforce/codet5p-220m"
# FINETUNE_MODEL_NAME = "Salesforce/codet5p-770m-py/lightning_logs/version_0/checkpoints/'epoch=9-step=20.ckpt'"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def main(model_name: str):
    model = construct_model(model_name=MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(f"checkpoints/{MODEL_NAME}.pt"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=2048)
    valid_loader = get_loaders(batch_size=1, model_name=MODEL_NAME, split="valid")

    for batch in valid_loader:
        print("#" * 80)
        print("Problem:")
        print(tokenizer.decode(batch["input_ids"][0]))
        outputs = model.generate(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            max_length=2048,
        )
        print("Response:")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        print("Expected Answer:")
        print(tokenizer.decode(batch["raw_labels"][0], skip_special_tokens=True))


if __name__ == "__main__":
    model_name_lst = [
        "t5-small",
        "t5-base",
        "Salesforce/codet5p-220m",
        "Salesforce/codet5p-770m-py",
    ]
    for mn in model_name_lst:
        print(mn)
        main(mn)
