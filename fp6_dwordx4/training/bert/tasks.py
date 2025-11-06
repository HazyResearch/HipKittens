import os
from typing import Tuple, List
import wandb
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import AdamW

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (AutoTokenizer, AutoModel,get_linear_schedule_with_warmup)


use_base = False
use_aiter = False
use_hipkittens = True 
do_save = False # whether to save the best model ckpt
PRE_TRAINED_MODEL_NAME = "bert-base-cased"
EPOCHS = 10
BATCH_SIZE = 16
MAX_LEN = 512 
if use_hipkittens:
    MAX_LEN = 1024
RANDOM_SEED = 42
LR = 2e-5
wandb.init(
    project=f"amd-hipkittens",
    name=f"bert-imdb-sentiment-len{MAX_LEN}-base{use_base}-aiter{use_aiter}-hipkittens{use_hipkittens}",
    config={
        "model_name": PRE_TRAINED_MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "lr": LR,
        "use_base": use_base,
        "use_aiter": use_aiter,
        "use_hipkittens": use_hipkittens, 
    }
)

# ----------------------------
# Repro & device
# ----------------------------
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Data
# ----------------------------
from datasets import load_dataset
ds = load_dataset("imdb")  # splits: train, test, unsupervised
df_train = ds["train"].to_pandas()
df_test  = ds["test"].to_pandas()
df = pd.concat([df_train, df_test], ignore_index=True)
df = df.rename(columns={"text": "review"})

# ----------------------------
# Tokenizer / lengths (sanity check)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
_ = tokenizer(
    "I want to learn how to do sentiment analysis using BERT and tokenizer.",
    max_length=32,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

# ----------------------------
# Dataset
# ----------------------------
class MovieReviewDataset(Dataset):
    def __init__(self, reviews: np.ndarray, labels: np.ndarray, tokenizer, max_len: int):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int):
        review = str(self.reviews[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            review,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": torch.tensor(label, dtype=torch.long),
        }


def create_data_loader(df: pd.DataFrame, tokenizer, max_len: int, batch_size: int) -> DataLoader:
    ds = MovieReviewDataset(
        reviews=df.review.values,
        labels=df.label.values,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    # safer defaults across OSes; pin_memory speeds up host→GPU transfers
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        drop_last=True,
    )

# ----------------------------
# Splits & loaders
# ----------------------------
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED, stratify=df["label"])
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=df_test["label"])
train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader   = create_data_loader(df_val,   tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader  = create_data_loader(df_test,  tokenizer, MAX_LEN, BATCH_SIZE)

# ----------------------------
# Model
# ----------------------------
from models.base import BertSelfAttention
from models.aiter import AITERBertSelfAttention
from models.hipkittens import HipKittensBertSelfAttention
from transformers import BertModel, BertConfig

if use_base:
    BertSelfAttention = BertSelfAttention
elif use_aiter: 
    BertSelfAttention = AITERBertSelfAttention
elif use_hipkittens:
    BertSelfAttention = HipKittensBertSelfAttention
assert(sum([use_base, use_aiter, use_hipkittens]) == 1, "Only one of use_base, use_aiter or use_hipkittens can be True")

class SentimentClassifier(nn.Module):
    def __init__(self, tokenizer, n_classes: int):
        super().__init__()
        config = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=128 * 64,  # h_q * d
            num_attention_heads=64,  # h_q
            num_key_value_heads=8,  # h_kv (gqa)
            num_hidden_layers=12,
            intermediate_size=128 * 4 * 64,  # 4x hidden size
            max_position_embeddings=MAX_LEN,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
        )

        self.bert = BertModel(config)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(config.hidden_size, n_classes)
        self._custom_init(mean=0.0, std=0.02)

    def _custom_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        x = self.drop(pooled)
        return self.out(x)

print("Creating model...")
class_names = ["negative", "positive"]
model = SentimentClassifier(tokenizer, n_classes=len(class_names)).to(device)
for i, layer in enumerate(model.bert.encoder.layer):
    new_self = BertSelfAttention(model.bert.config, layer_idx=i)
    layer.attention.self = new_self
model = model.to(device).to(torch.bfloat16)
print("Model created.")

# Shape sanity check
_batch = next(iter(train_data_loader))
print("input_ids shape:", _batch["input_ids"].shape)
print("attention_mask shape:", _batch["attention_mask"].shape)

# ----------------------------
# Optimizer / Scheduler / Loss
# ----------------------------
optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)
loss_fn = nn.CrossEntropyLoss().to(device)

# ----------------------------
# Train / Eval
# ----------------------------
def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples: int,
    limit: int = None,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    correct = 0

    for i, batch in tqdm(enumerate(data_loader), desc="Training", total=len(data_loader)):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets)

        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if i % 100 == 0 and i > 0:
            val_acc, val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                len(df_val),
                limit=50,
            )
            examples_so_far = i * BATCH_SIZE
            print(f"{i}: val acc: {val_acc:.4f}, val loss: {val_loss:.4f}")
            print(f"{i}: train acc: {correct / float(examples_so_far):.4f}, train loss: {float(np.mean(losses)):.4f}")

        if limit is not None and i >= limit:
            break

    acc = correct / float(n_examples)
    return acc, float(np.mean(losses)) if losses else 0.0

def eval_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn,
    device,
    n_examples: int,
    limit: int = None,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    correct = 0

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), desc="Evaluating", total=len(data_loader)):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets)

            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            losses.append(loss.item())

            if limit is not None and i >= limit:
                break

    acc = correct / float(n_examples)
    return acc, float(np.mean(losses)) if losses else 0.0

# ----------------------------
# Train loop
# ----------------------------
# imports for timers
import time
history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
best_val_acc = 0.0

# starting val acc
# val_acc, val_loss = eval_model(
#     model,
#     val_data_loader,
#     loss_fn,
#     device,
#     len(df_val),
#     limit=50,
# )
# print(f"Starting val acc: {val_acc:.4f}, val loss: {val_loss:.4f}")


# wandb.log({
#     "epoch": 0,
#     "val_acc": val_acc,
#     "val_loss": val_loss,
#     "runtime": time.time(),
# })

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    print("-" * 10)

    torch.cuda.synchronize()
    start_time = time.time()
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train),
        limit=None,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Train loss {train_loss:.4f} | acc {train_acc:.4f}")

    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val),
        limit=None,
    )
    print(f"Val   loss {val_loss:.4f} | acc {val_acc:.4f}\n")

    history["train_acc"].append(train_acc)
    history["train_loss"].append(train_loss)
    history["val_acc"].append(val_acc)
    history["val_loss"].append(val_loss)

    wandb.log({
        "epoch": epoch,
        "train_acc": train_acc,
        "train_loss": train_loss,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "runtime": end_time - start_time,
    })

    if val_acc > best_val_acc and do_save:
        torch.save(model.state_dict(), f"best_model_state.bin")
        best_val_acc = val_acc

# ----------------------------
# Plot training curves
# ----------------------------
plt.figure()
plt.plot(history["train_acc"], label="train acc")
plt.plot(history["val_acc"], label="val acc")
plt.title("Training history")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# ----------------------------
# Test set eval
# ----------------------------
test_acc, _ = eval_model(
    model,
    test_data_loader,   
    loss_fn,
    device,
    len(df_test),
    limit=None,
)
print("Test accuracy:", round(test_acc, 4))

# ----------------------------
# Predictions helper
# ----------------------------
@torch.no_grad()
def get_predictions(model: nn.Module, data_loader: DataLoader):
    model.eval()
    review_texts, predictions, prediction_probs, real_values = [], [], [], []

    for batch in tqdm(data_loader, desc="Getting predictions", total=len(data_loader)):
        texts = batch["review_text"]
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        review_texts.extend(texts)
        predictions.append(preds.cpu())
        prediction_probs.append(probs.cpu())
        real_values.append(targets.cpu())

    predictions = torch.cat(predictions)
    prediction_probs = torch.cat(prediction_probs)
    real_values = torch.cat(real_values)

    return review_texts, predictions, prediction_probs, real_values


y_texts, y_pred, y_pred_probs, y_true = get_predictions(model, test_data_loader)

print(
    classification_report(
        y_true.numpy(), y_pred.numpy(), target_names=class_names, digits=4
    )
)

