from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from src.finetune.pipeline import construct_model, get_loaders

# MODEL_NAME = "t5-small"
# MODEL_NAME = "t5-base"
MODEL_NAME = "Salesforce/codet5p-220m"
# MODEL_NAME = "Salesforce/codet5p-770m-py"

FINETUNE_MODEL_NAME = "Salesforce/codet5p-770m-py/lightning_logs/version_0/checkpoints/'epoch=9-step=20.ckpt'"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def main():
    model = construct_model(model_name=FINETUNE_MODEL_NAME).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    valid_loader = get_loaders(batch_size=1, model_name=MODEL_NAME, split="valid")
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
    main()
