from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import timm
from torchmetrics import F1Score, Recall, Precision


class BaseModel(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        if num_classes == 2:
            self.precision = Precision(
                task="binary", average="macro", num_classes=num_classes
            )
            self.recall = Recall(
                task="binary", average="macro", num_classes=num_classes
            )
            self.f1_score = F1Score(task="binary", num_classes=num_classes)
        else:
            self.precision = Precision(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.recall = Recall(
                task="multiclass", average="macro", num_classes=num_classes
            )
            self.f1_score = F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        logits = self.forward(x)

    def validation_step(self, batch):
        x, y = batch
        logits = self.forward(x)
