"""Train TabularClassifier on UCI Adult without WandB (CSV logger)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

from models.classifiers import TabularClassifier
from training.lit_classifier import LitClassifier
from training.datamodules.adult import AdultDataModule, INPUT_TYPES, CARDINALITIES

dm = AdultDataModule(data_dir="data/", batch_size=256, seed=42, num_workers=4)

backbone = TabularClassifier(
    input_types=INPUT_TYPES,
    cardinalities=CARDINALITIES,
    hidden_dims=[32, 8],
    num_classes=2,
    dropout=0.2,
)
lit = LitClassifier(model=backbone, initial_lr=1e-3, final_lr=1e-6,
                    weight_decay=1e-4, warmup_steps=200)

trainer = L.Trainer(
    max_epochs=50,
    accelerator="auto",
    logger=CSVLogger("logs", name="adult_classifier"),
    callbacks=[
        ModelCheckpoint(
            dirpath="checkpoints/adult_classifier/",
            filename="Adult-Classifier-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ],
)

trainer.fit(lit, dm)
print("Best checkpoint:", trainer.checkpoint_callback.best_model_path)
