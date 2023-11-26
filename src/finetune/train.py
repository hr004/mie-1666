import math
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator
from torch import nn
from torch.nn import CrossEntropyLoss

from src.finetune.pipeline import LanguageModel, get_dummy_loaders

DATA_NAME = "dummy"
MODEL_NAME = "gpt2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    lr: float = 1e-5,
    weight_decay: float = 1e-2,
    model_id: int = 0,
    save_name: Optional[str] = None,
) -> nn.Module:
    save = save_name is not None

    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = CrossEntropyLoss(reduction="mean")
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    epochs = 3
    num_iter = 0
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            lm_logits = model(
                batch["input_ids"],
                batch["attention_mask"],
            )
            labels = batch["labels"]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            accelerator.backward(loss)
            optimizer.step()
            num_iter += 1

            if save and num_iter % 194 == 0:
                torch.save(
                    model.state_dict(),
                    f"../files/checkpoints/{save_name}/model_{model_id}/iter_{num_iter}.pt",
                )
    return model


def model_evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    accelerator = Accelerator()
    model, loader = accelerator.prepare(model, loader)

    loss_fn = CrossEntropyLoss(reduction="sum")
    total_loss, total_num = 0.0, 0
    for step, batch in enumerate(loader):
        with torch.no_grad():
            lm_logits = model(
                batch["input_ids"],
                batch["attention_mask"],
            )
            labels = batch["labels"]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            loss = (
                loss_fn(reshaped_shift_logits, shift_labels.view(-1)).detach().float()
            )
            total_loss += loss.cpu().float().item()
        total_num += reshaped_shift_logits.shape[0]
    return total_loss / total_num


def main(num_train: int = 1) -> None:
    os.makedirs("../files", exist_ok=True)
    os.makedirs("../files/checkpoints", exist_ok=True)

    if DATA_NAME == "dummy":
        train_loader = get_dummy_loaders(
            model_name=MODEL_NAME, split="train", batch_size=8
        )
        valid_loader = get_dummy_loaders(
            model_name=MODEL_NAME, split="valid", batch_size=8
        )
    else:
        raise NotImplementedError()

    save_name = f"data_{DATA_NAME}"
    for i in range(num_train):
        print(f"Training {i}th model.")
        start_time = time.time()

        model = LanguageModel(model_name=MODEL_NAME)
        model.train()
        model = train(
            model=model,
            loader=train_loader,
            model_id=i,
            save_name=save_name,
        )

        model.eval()
        valid_loss = model_evaluate(model=model, loader=valid_loader)
        print(f"Validation Perp: {math.exp(valid_loss)}")
        del model
        print(f"Took {time.time() - start_time} seconds.")


if __name__ == "__main__":
    main(num_train=1)
