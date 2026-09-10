# src/train.py

from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "YOUR_BEST_PRETRAINED_MODEL"

TRAIN_FILE = "data/train_pairs.csv"
OUTPUT_DIR = "models/sentence-transformer"

BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

SEED = 42


# ============================================================
# 2. Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. Load training data
# ============================================================

dataset = load_dataset(
    "csv",
    data_files=TRAIN_FILE,
)["train"]

print(dataset)
print(dataset.column_names)


# ============================================================
# 4. Load Sentence Transformer
# ============================================================

model = SentenceTransformer(MODEL_NAME)

print(f"Model: {MODEL_NAME}")
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")


# ============================================================
# 5. Define loss
# ============================================================

loss = losses.MultipleNegativesRankingLoss(model)


# ============================================================
# 6. Training arguments
# ============================================================

args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    warmup_ratio=WARMUP_RATIO,

    fp16=torch.cuda.is_available(),

    batch_sampler=BatchSamplers.NO_DUPLICATES,

    logging_steps=50,

    save_strategy="epoch",

    save_total_limit=2,

    seed=SEED,

    report_to="none",
)


# ============================================================
# 7. Trainer
# ============================================================

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    loss=loss,
)


# ============================================================
# 8. Train
# ============================================================

print("\nStarting training...\n")

trainer.train()


# ============================================================
# 9. Save final model
# ============================================================

final_model_path = Path(OUTPUT_DIR) / "final"

model.save_pretrained(final_model_path)

print("\nTraining finished.")
print(f"Model saved to: {final_model_path}")
