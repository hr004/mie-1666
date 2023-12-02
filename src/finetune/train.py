import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import (AdamW, T5ForConditionalGeneration,
                          get_linear_schedule_with_warmup)

from src.finetune.pipeline import (LanguageModel, construct_model,
                                   get_dummy_loaders, get_loaders)


class T5Module(pl.LightningModule):
    def __init__(self, model_name: str, lr: float = 1e-05, num_epochs: int = 3):
        super().__init__()
        self.model = construct_model(model_name)
        self.t_loader = get_loaders(model_name, batch_size=4, split="train")
        self.v_loader = get_loaders(model_name, batch_size=1, split="valid")
        self.lr = lr
        self.num_epochs = num_epochs

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def common_step(self, batch, batch_idx):
        del batch_idx
        del batch["raw_labels"]
        outputs = self(**batch)
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("training_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        num_train_optimization_steps = self.num_epochs * len(self.t_loader)
        lr_scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=5,
                num_training_steps=num_train_optimization_steps,
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return self.t_loader

    def val_dataloader(self):
        return self.v_loader

    def test_dataloader(self):
        return self.v_loader


def main():
    model_list = [
        # "t5-small",
        # "t5-base",
        "Salesforce/codet5p-220m",
        "Salesforce/codet5p-770m-py",
    ]

    for mn in model_list:
        model = T5Module(model_name=mn)
        early_stop_callback = EarlyStopping(
            monitor="validation_loss", patience=3, strict=False, verbose=False, mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = Trainer(
            default_root_dir=f"{mn}/",
            callbacks=[early_stop_callback, lr_monitor],
        )
        trainer.fit(model)


if __name__ == "__main__":
    main()
