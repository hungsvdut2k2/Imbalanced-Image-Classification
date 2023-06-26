import timm
import pytorch_lightning as pl
import torch
from torchmetrics import F1Score, Recall, Precision, Accuracy
from .focal_loss import FocalLoss


class BaseModel(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=1
        )
        if num_classes == 2:
            self.precision = Precision(
                task="binary", average="macro", num_classes=num_classes
            )
            self.recall = Recall(
                task="binary", average="macro", num_classes=num_classes
            )
            self.f1_score = F1Score(task="binary", num_classes=num_classes)
            self.accuracy = Accuracy(task="binary", num_classes=num_classes)
        else:
            self.precision = Precision(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.recall = Recall(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.f1_score = F1Score(task="multiclass", num_classes=num_classes)
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = FocalLoss(gamma=2, alpha=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        precision = self.precision(logits, y)
        recall = self.recall(logits, y)
        f1_score = self.f1_score(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_accuracy", accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001)
