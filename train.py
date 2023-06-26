import pytorch_lightning as pl
from model.base_model import BaseModel
from dataset.build_dataset import build_dataloader, build_transforms
from pytorch_lightning.loggers import CSVLogger
from argparse import ArgumentParser


def train(
    model_name: str,
    num_classes: int,
    batch_size: int,
    epochs: int,
    is_augmented: bool,
    train_path: str,
    val_path: str,
):
    transforms = build_transforms(train_path, is_augmented)
    train_dataloader = build_dataloader(train_path, batch_size, transforms=transforms)
    valid_dataloader = build_dataloader(
        val_path, batch_size=batch_size, transforms=transforms
    )
    base_model = BaseModel(model_name=model_name, num_classes=num_classes)
    logger = CSVLogger("logs", name="imbalanced_classifier")
    trainer = pl.Trainer(max_epochs=epochs, logger=logger)
    trainer.fit(base_model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--val-dir", type=str)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument(
        "--is-augmented", type=lambda x: (str(x).lower() == "true"), default=False
    )

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_path=args.train_dir,
        val_path=args.val_dir,
        is_augmented=args.is_augmented,
    )
