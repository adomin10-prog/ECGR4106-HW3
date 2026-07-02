# ECGR 4106 Homework 3: Sequence-to-Sequence Machine Translation

Below is the executed notebook organized as follows:

1. Requirements
2. Dataset download and setup
3. Helper functions
4. Plot functions
5. Problem 1 - Baseline sequence-to-sequence RNN model
6. Problem 1 plots and results
7. Problem 2 - Sequence-to-sequence model with attention
8. Problem 2 plots and results
9. Problem 3 - Custom sentence testing
10. Problem 3 results and model comparison


TO TEST CODE PLEASE CLICK LINK BELOW:  
https://colab.research.google.com/github/adomin10-prog/ECGR4106-HW3/blob/main/ECGR_4106_HW3.ipynb

Link to GitHub repository:  
https://github.com/adomin10-prog/ECGR4106-HW3

## Requirements

```bash
!pip -q install torch numpy pandas matplotlib tqdm nltk scikit-learn requests

from pathlib import Path
import requests

DATASET_URL = "https://raw.githubusercontent.com/adomin10-prog/ECGR4106-HW3/main/vast_english_french.txt"
DATASET_NAME = "vast_english_french.txt"

DATASET_PATH = Path("/content") / DATASET_NAME

response = requests.get(DATASET_URL, timeout=30)
response.raise_for_status()

DATASET_PATH.write_bytes(response.content)

if DATASET_PATH.stat().st_size == 0:
    raise RuntimeError("Dataset downloaded but the file is empty.")

print("Dataset ready:", DATASET_PATH)
print("File size:", DATASET_PATH.stat().st_size, "bytes")
```

## Imports and Configuration

```python
import os
import re
import random
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from IPython.display import display


RANDOM_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


COMMON_CONFIG = {
    "data_path": "vast_english_french.txt",
    "max_examples": None,
    "train_size": 0.80,
    "val_size": 0.20,
    "max_len": 25,
    "min_freq": 1,


    "batch_size": 32,


    "embed_size": 48,
    "hidden_size": 96,
    "num_layers": 1,


    "epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-3,
    "dropout": 0.35,
    "early_stopping_patience": 4,
    "grad_clip": 1.0,


    "lr_scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 1,


    "label_smoothing": 0.05,


    "selection_min_delta": 0.002,
    "overfit_penalty_weight": 0.01,


    "decode_strategy": "beam",
    "beam_width": 3,
    "length_penalty_alpha": 0.7,
    "min_decode_len": 1,
    "mask_special_tokens": True,

    "optimizer": "Adam",
    "loss_function": "Cross-Entropy Loss with <pad> ignored and label smoothing",
    "random_seed": RANDOM_SEED,
}

CONTROLLED_TRAINING_CONFIG = {
    "epochs": COMMON_CONFIG["epochs"],
    "learning_rate": COMMON_CONFIG["learning_rate"],
    "weight_decay": COMMON_CONFIG["weight_decay"],
    "dropout": COMMON_CONFIG["dropout"],
    "early_stopping_patience": COMMON_CONFIG["early_stopping_patience"],
    "lr_scheduler": COMMON_CONFIG["lr_scheduler"],
    "scheduler_factor": COMMON_CONFIG["scheduler_factor"],
    "scheduler_patience": COMMON_CONFIG["scheduler_patience"],
    "label_smoothing": COMMON_CONFIG["label_smoothing"],
    "selection_min_delta": COMMON_CONFIG["selection_min_delta"],
    "overfit_penalty_weight": COMMON_CONFIG["overfit_penalty_weight"],
    "decode_strategy": COMMON_CONFIG["decode_strategy"],
    "beam_width": COMMON_CONFIG["beam_width"],
    "length_penalty_alpha": COMMON_CONFIG["length_penalty_alpha"],
    "min_decode_len": COMMON_CONFIG["min_decode_len"],
    "mask_special_tokens": COMMON_CONFIG["mask_special_tokens"],
}


EXPERIMENT_CONFIGS = {
    "p1_baseline": {
        **CONTROLLED_TRAINING_CONFIG,
"problem": "Problem 1",
        "model_label": "Baseline GRU",
        "model_name": "Problem 1 English-to-French Baseline GRU",
        "architecture_type": "baseline",
        "architecture": "Baseline GRU Encoder-Decoder with Fixed Context",
        "attention_type": None,
        "direction": "English-to-French",
        "source_language": "English",
        "target_language": "French",
        "source_column": "English Input",
        "target_column": "Target French",
        "prediction_column": "Predicted French",
},
    "p2_attention": {
        **CONTROLLED_TRAINING_CONFIG,
"problem": "Problem 2",
        "model_label": "GRU with Bahdanau Attention",
        "model_name": "Problem 2 English-to-French GRU with Attention",
        "architecture_type": "attention",
        "architecture": "GRU Encoder-Decoder with Attention",
        "attention_type": "Bahdanau Additive Attention",
        "direction": "English-to-French",
        "source_language": "English",
        "target_language": "French",
        "source_column": "English Input",
        "target_column": "Target French",
        "prediction_column": "Predicted French",
},
    "p3_baseline": {
        **CONTROLLED_TRAINING_CONFIG,
"problem": "Problem 3A",
        "model_label": "Baseline GRU",
        "model_name": "Problem 3A French-to-English Baseline GRU",
        "architecture_type": "baseline",
        "architecture": "Baseline GRU Encoder-Decoder with Fixed Context",
        "attention_type": None,
        "direction": "French-to-English",
        "source_language": "French",
        "target_language": "English",
        "source_column": "French Input",
        "target_column": "Target English",
        "prediction_column": "Predicted English",
},
    "p3_attention": {
        **CONTROLLED_TRAINING_CONFIG,
"problem": "Problem 3B",
        "model_label": "GRU with Bahdanau Attention",
        "model_name": "Problem 3B French-to-English GRU with Attention",
        "architecture_type": "attention",
        "architecture": "GRU Encoder-Decoder with Attention",
        "attention_type": "Bahdanau Additive Attention",
        "direction": "French-to-English",
        "source_language": "French",
        "target_language": "English",
        "source_column": "French Input",
        "target_column": "Target English",
        "prediction_column": "Predicted English",
},
}


DATA_PATH = COMMON_CONFIG["data_path"]
MAX_EXAMPLES = COMMON_CONFIG["max_examples"]
TRAIN_SIZE = COMMON_CONFIG["train_size"]
VAL_SIZE = COMMON_CONFIG["val_size"]
MAX_LEN = COMMON_CONFIG["max_len"]
MIN_FREQ = COMMON_CONFIG["min_freq"]
BATCH_SIZE = COMMON_CONFIG["batch_size"]
EMBED_SIZE = COMMON_CONFIG["embed_size"]
HIDDEN_SIZE = COMMON_CONFIG["hidden_size"]
NUM_LAYERS = COMMON_CONFIG["num_layers"]
GRAD_CLIP = COMMON_CONFIG["grad_clip"]
EARLY_STOPPING_PATIENCE = COMMON_CONFIG["early_stopping_patience"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)
print("Controlled training configuration used by every experiment:")
display(pd.DataFrame([CONTROLLED_TRAINING_CONFIG]))

display(pd.DataFrame(EXPERIMENT_CONFIGS).T[[
    "problem", "model_label", "direction", "architecture_type",
    "epochs", "learning_rate", "weight_decay", "dropout", "early_stopping_patience",
    "label_smoothing", "selection_min_delta", "overfit_penalty_weight",
    "lr_scheduler", "decode_strategy", "beam_width", "min_decode_len"
]])
```

**Output**

```text
Using device: cpu
Controlled training configuration used by every experiment:
```

|   epochs |   learning_rate |   weight_decay |   dropout |   early_stopping_patience | lr_scheduler      |   scheduler_factor |   scheduler_patience |   label_smoothing |   selection_min_delta |   overfit_penalty_weight | decode_strategy   |   beam_width |   length_penalty_alpha |   min_decode_len | mask_special_tokens   |
|---------:|----------------:|---------------:|----------:|--------------------------:|:------------------|-------------------:|---------------------:|------------------:|----------------------:|-------------------------:|:------------------|-------------:|-----------------------:|-----------------:|:----------------------|
|       50 |           0.001 |          0.001 |      0.35 |                         4 | ReduceLROnPlateau |                0.5 |                    1 |              0.05 |                 0.002 |                     0.01 | beam              |            3 |                    0.7 |                1 | True                  |

| Unnamed: 0   | problem    | model_label                 | direction         | architecture_type   |   epochs |   learning_rate |   weight_decay |   dropout |   early_stopping_patience |   label_smoothing |   selection_min_delta |   overfit_penalty_weight | lr_scheduler      | decode_strategy   |   beam_width |   min_decode_len |
|:-------------|:-----------|:----------------------------|:------------------|:--------------------|---------:|----------------:|---------------:|----------:|--------------------------:|------------------:|----------------------:|-------------------------:|:------------------|:------------------|-------------:|-----------------:|
| p1_baseline  | Problem 1  | Baseline GRU                | English-to-French | baseline            |       50 |           0.001 |          0.001 |      0.35 |                         4 |              0.05 |                 0.002 |                     0.01 | ReduceLROnPlateau | beam              |            3 |                1 |
| p2_attention | Problem 2  | GRU with Bahdanau Attention | English-to-French | attention           |       50 |           0.001 |          0.001 |      0.35 |                         4 |              0.05 |                 0.002 |                     0.01 | ReduceLROnPlateau | beam              |            3 |                1 |
| p3_baseline  | Problem 3A | Baseline GRU                | French-to-English | baseline            |       50 |           0.001 |          0.001 |      0.35 |                         4 |              0.05 |                 0.002 |                     0.01 | ReduceLROnPlateau | beam              |            3 |                1 |
| p3_attention | Problem 3B | GRU with Bahdanau Attention | French-to-English | attention           |       50 |           0.001 |          0.001 |      0.35 |                         4 |              0.05 |                 0.002 |                     0.01 | ReduceLROnPlateau | beam              |            3 |                1 |

## Helper Functions

```python
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

smoothing_function = SmoothingFunction().method1


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u202f", " ").replace("\xa0", " ")
    text = text.lower().strip()


    text = re.sub(r"([.!?;,¿¡:])", r" \1 ", text)


    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9.!?;,¿¡:'’\-]+", " ", text)


    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text):
    return clean_text(text).split()


def upload_dataset_if_needed(data_path):
    if not os.path.exists(data_path):
        from google.colab import files
        print(f"Upload {data_path}")
        uploaded = files.upload()

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} was not found. Upload the dataset and rerun this cell."
        )

    print("Dataset ready:", data_path)


def read_translation_pairs(path, max_examples=None):
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                continue

            english = clean_text(parts[0])
            french = clean_text(parts[1])

            if english and french:
                pairs.append((english, french))

            if max_examples is not None and len(pairs) >= max_examples:
                break

    return pairs


def create_shared_split(pairs, train_size, val_size, random_seed):
    all_indices = np.arange(len(pairs))

    train_indices, val_indices = train_test_split(
        all_indices,
        train_size=train_size,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True
    )

    train_pairs = [pairs[i] for i in train_indices]
    val_pairs = [pairs[i] for i in val_indices]

    return all_indices, train_indices, val_indices, train_pairs, val_pairs


def make_split_table(train_pairs, val_pairs, train_percent=80, val_percent=20):
    return pd.DataFrame({
        "Split": ["Training", "Validation"],
        "Percent": [train_percent, val_percent],
        "Sentence Pairs": [len(train_pairs), len(val_pairs)]
    })


def build_parallel_vocabs(pairs, src_index, tgt_index, min_freq):
    source_tokens = [tokenize(pair[src_index]) for pair in pairs]
    target_tokens = [tokenize(pair[tgt_index]) for pair in pairs]

    src_vocab = Vocab(source_tokens, min_freq=min_freq)
    tgt_vocab = Vocab(target_tokens, min_freq=min_freq)

    return src_vocab, tgt_vocab, source_tokens, target_tokens


def make_vocab_table(vocab_names, vocabs):
    return pd.DataFrame({
        "Vocabulary": vocab_names,
        "Size": [len(vocab) for vocab in vocabs]
    })


def compute_oov_diagnostic(train_pairs, val_pairs, source_label, target_label, total_pairs_count=None):
    train_src_words = set()
    train_tgt_words = set()
    val_src_words = set()
    val_tgt_words = set()

    for src_text, tgt_text in train_pairs:
        train_src_words.update(tokenize(src_text))
        train_tgt_words.update(tokenize(tgt_text))

    for src_text, tgt_text in val_pairs:
        val_src_words.update(tokenize(src_text))
        val_tgt_words.update(tokenize(tgt_text))

    src_oov = val_src_words - train_src_words
    tgt_oov = val_tgt_words - train_tgt_words

    if total_pairs_count is None:
        total_pairs_count = len(train_pairs) + len(val_pairs)

    diagnostic_df = pd.DataFrame({
        "Check": [
            "Total sentence pairs",
            "Training sentence pairs",
            "Validation sentence pairs",
            f"Unique {source_label} training source tokens",
            f"Unique {target_label} training target tokens",
            f"Validation {source_label} OOV tokens compared to training",
            f"Validation {target_label} OOV tokens compared to training",
            f"Validation {source_label} OOV %",
            f"Validation {target_label} OOV %"
        ],
        "Value": [
            total_pairs_count,
            len(train_pairs),
            len(val_pairs),
            len(train_src_words),
            len(train_tgt_words),
            len(src_oov),
            len(tgt_oov),
            100 * len(src_oov) / max(1, len(val_src_words)),
            100 * len(tgt_oov) / max(1, len(val_tgt_words))
        ]
    })

    return diagnostic_df, src_oov, tgt_oov


def make_translation_dataloaders(train_pairs, val_pairs, src_vocab, tgt_vocab, max_len, batch_size):
    train_dataset = TranslationDataset(
        sentence_pairs=train_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len
    )

    val_dataset = TranslationDataset(
        sentence_pairs=val_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataset, val_dataset, train_loader, val_loader


class Vocab:
    def __init__(self, token_lists, min_freq=1, specials=None):
        if specials is None:
            specials = SPECIAL_TOKENS

        self.idx_to_token = []
        self.token_to_idx = {}

        for token in specials:
            self.add_token(token)

        counter = Counter()

        for tokens in token_lists:
            counter.update(tokens)

        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token_to_idx:
                self.add_token(token)

        self.pad = self.token_to_idx[PAD_TOKEN]
        self.bos = self.token_to_idx[BOS_TOKEN]
        self.eos = self.token_to_idx[EOS_TOKEN]
        self.unk = self.token_to_idx[UNK_TOKEN]

    def add_token(self, token):
        self.token_to_idx[token] = len(self.idx_to_token)
        self.idx_to_token.append(token)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def to_token(self, idx):
        return self.idx_to_token[int(idx)]


def encode_sentence(text, vocab, max_len, add_bos=False, add_eos=True):
    tokens = tokenize(text)

    if add_bos:
        tokens = [BOS_TOKEN] + tokens

    if add_eos:
        tokens = tokens + [EOS_TOKEN]

    ids = [vocab[token] for token in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]

        if add_eos:
            ids[-1] = vocab[EOS_TOKEN]

    valid_length = len(ids)

    if len(ids) < max_len:
        ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long), valid_length


def ids_to_tokens(ids, vocab):
    tokens = []

    for idx in ids:
        token = vocab.to_token(idx)

        if token == EOS_TOKEN:
            break

        if token in [PAD_TOKEN, BOS_TOKEN]:
            continue

        tokens.append(token)

    return tokens


def tokens_to_sentence(tokens):
    return " ".join(tokens)


class TranslationDataset(Dataset):
    def __init__(self, sentence_pairs, src_vocab, tgt_vocab, max_len):
        self.sentence_pairs = sentence_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.sentence_pairs[idx]

        src_ids, src_len = encode_sentence(
            src_text,
            self.src_vocab,
            self.max_len,
            add_bos=False,
            add_eos=True
        )

        tgt_ids, tgt_len = encode_sentence(
            tgt_text,
            self.tgt_vocab,
            self.max_len,
            add_bos=True,
            add_eos=True
        )

        return (
            src_ids,
            torch.tensor(src_len, dtype=torch.long),
            tgt_ids,
            torch.tensor(tgt_len, dtype=torch.long)
        )


def compute_exact_match(pred_tokens, target_tokens):
    return int(pred_tokens == target_tokens)


def compute_bleu4(pred_tokens, target_tokens):
    if len(pred_tokens) == 0:
        return 0.0

    return sentence_bleu(
        [target_tokens],
        pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_function
    )


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def best_epoch_row(history_df):
    if "selection_score" in history_df.columns:
        return history_df.loc[history_df["selection_score"].idxmax()].copy()

    return history_df.loc[history_df["val_loss"].idxmin()].copy()


def make_best_epoch_table(history_df):
    best_row = best_epoch_row(history_df)
    record = {
        "Best Epoch": int(best_row["epoch"]),
        "Training Loss at Best Epoch": float(best_row["train_loss"]),
        "Best Validation Loss": float(best_row["val_loss"]),
        "Training-Validation Gap": float(best_row["val_loss"] - best_row["train_loss"]),
        "Validation Perplexity": float(np.exp(best_row["val_loss"])),
    }

    if "train_token_accuracy" in history_df.columns:
        record.update({
            "Training Token Accuracy at Best Epoch": float(best_row["train_token_accuracy"]),
            "Validation Token Accuracy at Best Epoch": float(best_row["val_token_accuracy"]),
            "Token Accuracy Gap": float(best_row["token_accuracy_gap"]),
            "Selection Score": float(best_row["selection_score"]),
        })

    return pd.DataFrame([record])


def make_dataset_table(total_pairs, train_pairs, val_pairs, train_size, val_size, max_len, random_seed):
    return pd.DataFrame({
        "Metric": [
            "Total Sentence Pairs",
            "Training Sentence Pairs",
            "Validation Sentence Pairs",
            "Training Percent",
            "Validation Percent",
            "Max Sequence Length",
            "Random Seed"
        ],
        "Value": [
            total_pairs,
            len(train_pairs),
            len(val_pairs),
            f"{train_size * 100:.0f}%",
            f"{val_size * 100:.0f}%",
            max_len,
            random_seed
        ]
    })


def make_model_size_table(model, src_vocab, tgt_vocab, embed_size, hidden_size, num_layers):
    return pd.DataFrame({
        "Metric": [
            "Trainable Parameters",
            "Source Vocabulary Size",
            "Target Vocabulary Size",
            "Embedding Size",
            "Hidden Size",
            "GRU Layers"
        ],
        "Value": [
            count_trainable_parameters(model),
            len(src_vocab),
            len(tgt_vocab),
            embed_size,
            hidden_size,
            num_layers
        ]
    })


def make_two_model_size_table(model_rows, embed_size, hidden_size, num_layers):
    return pd.DataFrame({
        "Model": [row["model_name"] for row in model_rows],
        "Trainable Parameters": [count_trainable_parameters(row["model"]) for row in model_rows],
        "Source Vocabulary Size": [len(row["src_vocab"]) for row in model_rows],
        "Target Vocabulary Size": [len(row["tgt_vocab"]) for row in model_rows],
        "Embedding Size": [embed_size for _ in model_rows],
        "Hidden Size": [hidden_size for _ in model_rows],
        "GRU Layers": [num_layers for _ in model_rows]
    })


def make_training_table(history_df):
    best_row = best_epoch_row(history_df)
    final_row = history_df.iloc[-1].copy()

    metrics = [
        "Epochs Completed",
        "Best Selected Epoch",
        "Training Loss at Best Epoch",
        "Validation Loss at Best Epoch",
        "Training-Validation Gap at Best Epoch",
        "Validation Perplexity at Best Epoch",
    ]
    values = [
        int(len(history_df)),
        int(best_row["epoch"]),
        float(best_row["train_loss"]),
        float(best_row["val_loss"]),
        float(best_row["val_loss"] - best_row["train_loss"]),
        float(np.exp(best_row["val_loss"])),
    ]

    if "train_token_accuracy" in history_df.columns:
        metrics.extend([
            "Training Token Accuracy at Best Epoch",
            "Validation Token Accuracy at Best Epoch",
            "Token Accuracy Gap at Best Epoch",
            "Validation-First Selection Score",
        ])
        values.extend([
            float(best_row["train_token_accuracy"]),
            float(best_row["val_token_accuracy"]),
            float(best_row["token_accuracy_gap"]),
            float(best_row["selection_score"]),
        ])

    metrics.extend([
        "Final Epoch",
        "Final Training Loss",
        "Final Validation Loss",
        "Final Training-Validation Gap",
        "Final Validation Perplexity",
    ])
    values.extend([
        int(final_row["epoch"]),
        float(final_row["train_loss"]),
        float(final_row["val_loss"]),
        float(final_row["val_loss"] - final_row["train_loss"]),
        float(np.exp(final_row["val_loss"])),
    ])

    if "train_token_accuracy" in history_df.columns:
        metrics.extend([
            "Final Training Token Accuracy",
            "Final Validation Token Accuracy",
            "Final Token Accuracy Gap",
            "Final Selection Score",
        ])
        values.extend([
            float(final_row["train_token_accuracy"]),
            float(final_row["val_token_accuracy"]),
            float(final_row["token_accuracy_gap"]),
            float(final_row["selection_score"]),
        ])

    return pd.DataFrame({"Metric": metrics, "Value": values})


def make_training_table_rows(model_history_rows):
    records = []

    for row in model_history_rows:
        best_row = best_epoch_row(row["history_df"])
        final_row = row["history_df"].iloc[-1].copy()

        records.append({
            "Model": row["model_name"],
            "Epochs Completed": int(len(row["history_df"])),
            "Best Validation Epoch": int(best_row["epoch"]),
            "Training Loss at Best Epoch": float(best_row["train_loss"]),
            "Best Validation Loss": float(best_row["val_loss"]),
            "Training-Validation Gap at Best Epoch": float(best_row["val_loss"] - best_row["train_loss"]),
            "Validation Perplexity at Best Epoch": float(np.exp(best_row["val_loss"])),
            "Final Epoch": int(final_row["epoch"]),
            "Final Training Loss": float(final_row["train_loss"]),
            "Final Validation Loss": float(final_row["val_loss"])
        })

    return pd.DataFrame(records)


def make_exact_match_table(results_df):
    exact_match_count = int(results_df["Exact Match"].sum())
    total_validation_count = len(results_df)
    non_exact_match_count = total_validation_count - exact_match_count

    return pd.DataFrame({
        "Metric": [
            "Validation Examples",
            "Exact Match Count",
            "Non-Exact Match Count",
            "Exact Match Accuracy"
        ],
        "Value": [
            total_validation_count,
            exact_match_count,
            non_exact_match_count,
            exact_match_count / max(1, total_validation_count)
        ]
    })


def make_bleu_distribution_table(results_df, model_name=None):
    table = pd.DataFrame({
        "Metric": [
            "Average BLEU-4",
            "Median BLEU-4",
            "Minimum BLEU-4",
            "Maximum BLEU-4",
            "Standard Deviation BLEU-4",
            "BLEU-4 >= 0.25 Count",
            "BLEU-4 >= 0.50 Count",
            "BLEU-4 >= 0.75 Count"
        ],
        "Value": [
            float(results_df["BLEU-4"].mean()),
            float(results_df["BLEU-4"].median()),
            float(results_df["BLEU-4"].min()),
            float(results_df["BLEU-4"].max()),
            float(results_df["BLEU-4"].std()),
            int((results_df["BLEU-4"] >= 0.25).sum()),
            int((results_df["BLEU-4"] >= 0.50).sum()),
            int((results_df["BLEU-4"] >= 0.75).sum())
        ]
    })

    if model_name is not None:
        table["Model"] = model_name

    return table


def top_bleu_examples(results_df, n=10):
    return (
        results_df
        .sort_values(by="BLEU-4", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def low_bleu_examples(results_df, n=10):
    return (
        results_df
        .sort_values(by="BLEU-4", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def exact_match_examples(results_df):
    return results_df[results_df["Exact Match"] == 1].reset_index(drop=True)


def make_translation_metrics_df(results_df, history_df, model_name):
    best_row = best_epoch_row(history_df)

    record = {
        "Model": model_name,
        "Best Epoch": int(best_row["epoch"]),
        "Training Loss at Best Epoch": float(best_row["train_loss"]),
        "Best Validation Loss": float(best_row["val_loss"]),
    }

    if "train_token_accuracy" in history_df.columns:
        record.update({
            "Training Token Accuracy": float(best_row["train_token_accuracy"]),
            "Validation Token Accuracy": float(best_row["val_token_accuracy"]),
            "Token Accuracy Gap": float(best_row["token_accuracy_gap"]),
            "Validation-First Selection Score": float(best_row["selection_score"]),
        })

    record.update({
        "Exact Match Accuracy": float(results_df["Exact Match"].mean()),
        "Average BLEU-4": float(results_df["BLEU-4"].mean()),
    })

    return pd.DataFrame([record])


def make_compact_report_table(problem, direction, architecture, history_df, results_df, model):
    best_row = best_epoch_row(history_df)

    record = {
        "Problem": problem,
        "Direction": direction,
        "Architecture": architecture,
        "Best Epoch": int(best_row["epoch"]),
        "Best Validation Loss": float(best_row["val_loss"]),
        "Training Loss at Best Epoch": float(best_row["train_loss"]),
        "Training-Validation Gap": float(best_row["val_loss"] - best_row["train_loss"]),
    }

    if "train_token_accuracy" in history_df.columns:
        record.update({
            "Validation Token Accuracy": float(best_row["val_token_accuracy"]),
            "Training Token Accuracy": float(best_row["train_token_accuracy"]),
            "Token Accuracy Gap": float(best_row["token_accuracy_gap"]),
            "Validation-First Selection Score": float(best_row["selection_score"]),
        })

    record.update({
        "Exact Match Accuracy": float(results_df["Exact Match"].mean()),
        "Average BLEU-4": float(results_df["BLEU-4"].mean()),
        "Trainable Parameters": count_trainable_parameters(model),
    })

    return pd.DataFrame([record])


def make_model_comparison(rows):
    records = []

    for row in rows:
        best_row = best_epoch_row(row["history_df"])
        history_df = row["history_df"]
        results_df = row["results_df"]
        model = row["model"]

        record = {
            "Problem": row["problem"],
            "Model": row["model_label"],
            "Direction": row["direction"],
            "Best Epoch": int(best_row["epoch"]),
            "Best Validation Loss": float(best_row["val_loss"]),
            "Training Loss at Best Epoch": float(best_row["train_loss"]),
            "Training-Validation Gap": float(best_row["val_loss"] - best_row["train_loss"]),
        }

        if "train_token_accuracy" in history_df.columns:
            record.update({
                "Validation Token Accuracy": float(best_row["val_token_accuracy"]),
                "Training Token Accuracy": float(best_row["train_token_accuracy"]),
                "Token Accuracy Gap": float(best_row["token_accuracy_gap"]),
                "Validation-First Selection Score": float(best_row["selection_score"]),
            })

        record.update({
            "Exact Match Accuracy": float(results_df["Exact Match"].mean()),
            "Average BLEU-4": float(results_df["BLEU-4"].mean()),
            "Trainable Parameters": count_trainable_parameters(model),
        })
        records.append(record)

    return pd.DataFrame(records)


def make_improvement_table(baseline_history, candidate_history, baseline_results, candidate_results, baseline_label, candidate_label):
    baseline_best = best_epoch_row(baseline_history)
    candidate_best = best_epoch_row(candidate_history)

    rows = [
        {
            "Metric": "Best Validation Loss",
            baseline_label: float(baseline_best["val_loss"]),
            candidate_label: float(candidate_best["val_loss"]),
            f"{candidate_label} - {baseline_label}": float(candidate_best["val_loss"] - baseline_best["val_loss"]),
            "Improved?": "Yes" if float(candidate_best["val_loss"]) < float(baseline_best["val_loss"]) else "No",
        }
    ]

    if "val_token_accuracy" in baseline_history.columns and "val_token_accuracy" in candidate_history.columns:
        rows.extend([
            {
                "Metric": "Validation Token Accuracy",
                baseline_label: float(baseline_best["val_token_accuracy"]),
                candidate_label: float(candidate_best["val_token_accuracy"]),
                f"{candidate_label} - {baseline_label}": float(candidate_best["val_token_accuracy"] - baseline_best["val_token_accuracy"]),
                "Improved?": "Yes" if float(candidate_best["val_token_accuracy"]) > float(baseline_best["val_token_accuracy"]) else "No",
            },
            {
                "Metric": "Training-Validation Loss Gap",
                baseline_label: float(baseline_best["val_loss"] - baseline_best["train_loss"]),
                candidate_label: float(candidate_best["val_loss"] - candidate_best["train_loss"]),
                f"{candidate_label} - {baseline_label}": float((candidate_best["val_loss"] - candidate_best["train_loss"]) - (baseline_best["val_loss"] - baseline_best["train_loss"])),
                "Improved?": "Yes" if float(candidate_best["val_loss"] - candidate_best["train_loss"]) < float(baseline_best["val_loss"] - baseline_best["train_loss"]) else "No",
            },
        ])

    baseline_exact = float(baseline_results["Exact Match"].mean())
    candidate_exact = float(candidate_results["Exact Match"].mean())
    baseline_bleu = float(baseline_results["BLEU-4"].mean())
    candidate_bleu = float(candidate_results["BLEU-4"].mean())

    rows.extend([
        {
            "Metric": "Exact Match Accuracy",
            baseline_label: baseline_exact,
            candidate_label: candidate_exact,
            f"{candidate_label} - {baseline_label}": candidate_exact - baseline_exact,
            "Improved?": "Yes" if candidate_exact > baseline_exact else "No",
        },
        {
            "Metric": "Average BLEU-4",
            baseline_label: baseline_bleu,
            candidate_label: candidate_bleu,
            f"{candidate_label} - {baseline_label}": candidate_bleu - baseline_bleu,
            "Improved?": "Yes" if candidate_bleu > baseline_bleu else "No",
        },
    ])

    return pd.DataFrame(rows)


def make_hyperparameter_table(choices, values):
    return pd.DataFrame({
        "Choice": choices,
        "Value": values
    })


def make_loss_and_optimizer(model, tgt_vocab, learning_rate, weight_decay):
    label_smoothing = COMMON_CONFIG.get("label_smoothing", 0.0)

    try:
        criterion = nn.CrossEntropyLoss(
            ignore_index=tgt_vocab[PAD_TOKEN],
            label_smoothing=label_smoothing
        )
    except TypeError:
        print("Warning: this PyTorch version does not support label_smoothing. Using standard CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab[PAD_TOKEN])

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    return criterion, optimizer


def make_lr_scheduler(optimizer, config):
    if config.get("lr_scheduler") == "ReduceLROnPlateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.get("scheduler_factor", 0.5),
            patience=config.get("scheduler_patience", 2)
        )

    return None


def _run_seq2seq_epoch(
    model,
    dataloader,
    criterion,
    device,
    tgt_pad_idx,
    optimizer=None,
    desc="Training",
    grad_clip=1.0
):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    context_manager = torch.enable_grad() if is_training else torch.no_grad()

    with context_manager:
        for src, src_lengths, tgt, tgt_lengths in tqdm(dataloader, desc=desc, leave=False):
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            tgt = tgt.to(device)


            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            if is_training:
                optimizer.zero_grad()

            logits = model(src, src_lengths, tgt_input)
            vocab_size = logits.shape[-1]

            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_expected.reshape(-1)
            )

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            mask = tgt_expected != tgt_pad_idx
            non_pad_tokens = mask.sum().item()
            predictions = logits.argmax(dim=-1)

            correct_tokens += ((predictions == tgt_expected) & mask).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    average_loss = total_loss / max(1, total_tokens)
    token_accuracy = correct_tokens / max(1, total_tokens)

    return average_loss, token_accuracy


def train_seq2seq_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device,
    tgt_pad_idx,
    patience,
    train_desc="Training",
    val_desc="Validating",
    grad_clip=1.0,
    scheduler=None,
    min_delta=0.002,
    overfit_penalty_weight=0.01
):
    history = []
    best_score = -float("inf")
    best_model_state = None
    best_epoch = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_token_acc = _run_seq2seq_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            tgt_pad_idx=tgt_pad_idx,
            optimizer=optimizer,
            desc=train_desc,
            grad_clip=grad_clip
        )

        val_loss, val_token_acc = _run_seq2seq_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            tgt_pad_idx=tgt_pad_idx,
            optimizer=None,
            desc=val_desc,
            grad_clip=grad_clip
        )

        loss_gap = val_loss - train_loss
        token_accuracy_gap = train_token_acc - val_token_acc
        selection_score = val_token_acc - overfit_penalty_weight * max(0.0, loss_gap)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_token_accuracy": train_token_acc,
            "val_token_accuracy": val_token_acc,
            "loss_gap": loss_gap,
            "token_accuracy_gap": token_accuracy_gap,
            "selection_score": selection_score,
            "learning_rate": current_lr
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Tok Acc: {train_token_acc:.4f} | "
            f"Val Tok Acc: {val_token_acc:.4f} | "
            f"Loss Gap: {loss_gap:.4f} | "
            f"Score: {selection_score:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if selection_score > best_score + min_delta:
            best_score = selection_score
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  New best validation-first score: {best_score:.4f}")
        else:
            patience_counter += 1
            print(f"  No useful validation improvement. Patience: {patience_counter}/{patience}")

        if scheduler is not None:
            scheduler.step(val_loss)
            next_lr = optimizer.param_groups[0]["lr"]
            if next_lr < current_lr:
                print(f"  Scheduler reduced learning rate to {next_lr:.6f}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"Loaded best model from epoch {best_epoch} with validation-first score: {best_score:.4f}")

    return pd.DataFrame(history)


def train_one_epoch_baseline(model, dataloader, optimizer, criterion, device, tgt_pad_idx):
    return _run_seq2seq_epoch(model, dataloader, criterion, device, tgt_pad_idx, optimizer=optimizer, desc="Training")


def evaluate_loss_baseline(model, dataloader, criterion, device, tgt_pad_idx):
    return _run_seq2seq_epoch(model, dataloader, criterion, device, tgt_pad_idx, optimizer=None, desc="Validating")


def train_baseline_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, tgt_pad_idx):
    return train_seq2seq_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        tgt_pad_idx=tgt_pad_idx,
        patience=EARLY_STOPPING_PATIENCE,
        train_desc="Training",
        val_desc="Validating",
        grad_clip=GRAD_CLIP
    )


def train_one_epoch_attention(model, dataloader, optimizer, criterion, device, tgt_pad_idx):
    return _run_seq2seq_epoch(model, dataloader, criterion, device, tgt_pad_idx, optimizer=optimizer, desc="Training Attention")


def evaluate_loss_attention(model, dataloader, criterion, device, tgt_pad_idx):
    return _run_seq2seq_epoch(model, dataloader, criterion, device, tgt_pad_idx, optimizer=None, desc="Validating Attention")


def train_attention_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, tgt_pad_idx, patience):
    return train_seq2seq_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=epochs,
        device=device,
        tgt_pad_idx=tgt_pad_idx,
        patience=patience,
        train_desc="Training Attention",
        val_desc="Validating Attention",
        grad_clip=GRAD_CLIP
    )


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, src, src_lengths):
        embedded = self.dropout(self.embedding(src))

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, hidden = self.gru(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1)
        )

        return outputs, hidden


class DecoderGRUFixedContext(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)


        self.gru = nn.GRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt_input, hidden, context):
        embedded = self.dropout(self.embedding(tgt_input))


        context_repeated = context.unsqueeze(1).repeat(1, embedded.size(1), 1)

        decoder_input = torch.cat((embedded, context_repeated), dim=2)

        outputs, hidden = self.gru(decoder_input, hidden)

        logits = self.fc_out(outputs)

        return logits, hidden


class Seq2SeqGRUBaseline(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, tgt_input):
        encoder_outputs, hidden = self.encoder(src, src_lengths)


        context = hidden[-1]

        logits, hidden = self.decoder(tgt_input, hidden, context)

        return logits


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_keys = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, values, src_lengths):
        batch_size, src_len, hidden_size = keys.shape

        query_expanded = query.unsqueeze(1).repeat(1, src_len, 1)

        scores = self.v(
            torch.tanh(
                self.W_query(query_expanded) + self.W_keys(keys)
            )
        ).squeeze(-1)


        mask = torch.arange(src_len, device=keys.device).unsqueeze(0) >= src_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)

        attention_weights = torch.softmax(scores, dim=1)


        context = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)

        return context, attention_weights


class DecoderGRUAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.0, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_size,
            padding_idx=pad_idx
        )

        self.dropout = nn.Dropout(dropout)

        self.attention = BahdanauAttention(hidden_size)


        self.gru = nn.GRU(
            input_size=embed_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )


        self.fc_out = nn.Linear(hidden_size + hidden_size + embed_size, vocab_size)

    def forward_step(self, input_token, hidden, encoder_outputs, src_lengths):
        embedded = self.dropout(self.embedding(input_token))


        query = hidden[-1]

        context, attention_weights = self.attention(
            query=query,
            keys=encoder_outputs,
            values=encoder_outputs,
            src_lengths=src_lengths
        )

        context_unsqueezed = context.unsqueeze(1)

        gru_input = torch.cat((embedded, context_unsqueezed), dim=2)

        output, hidden = self.gru(gru_input, hidden)

        prediction_input = torch.cat(
            (output.squeeze(1), context, embedded.squeeze(1)),
            dim=1
        )

        logits = self.fc_out(prediction_input)

        return logits, hidden, attention_weights

    def forward(self, tgt_input, hidden, encoder_outputs, src_lengths):
        batch_size = tgt_input.shape[0]
        tgt_len = tgt_input.shape[1]
        vocab_size = self.fc_out.out_features
        src_len = encoder_outputs.shape[1]

        logits_all = torch.zeros(batch_size, tgt_len, vocab_size, device=tgt_input.device)
        attentions_all = torch.zeros(batch_size, tgt_len, src_len, device=tgt_input.device)

        for t in range(tgt_len):
            input_token = tgt_input[:, t].unsqueeze(1)

            logits, hidden, attention_weights = self.forward_step(
                input_token=input_token,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                src_lengths=src_lengths
            )

            logits_all[:, t, :] = logits
            attentions_all[:, t, :] = attention_weights

        return logits_all, hidden, attentions_all


class Seq2SeqGRUAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_lengths, tgt_input, return_attention=False):
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        logits, hidden, attentions = self.decoder(
            tgt_input=tgt_input,
            hidden=hidden,
            encoder_outputs=encoder_outputs,
            src_lengths=src_lengths
        )

        if return_attention:
            return logits, attentions

        return logits


print("Shared baseline and attention model classes loaded successfully.")


def build_baseline_seq2seq_model(src_vocab, tgt_vocab, embed_size, hidden_size, num_layers, dropout, device):
    encoder = EncoderGRU(
        vocab_size=len(src_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=src_vocab[PAD_TOKEN]
    )

    decoder = DecoderGRUFixedContext(
        vocab_size=len(tgt_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=tgt_vocab[PAD_TOKEN]
    )

    return Seq2SeqGRUBaseline(encoder=encoder, decoder=decoder).to(device)


def build_attention_seq2seq_model(src_vocab, tgt_vocab, embed_size, hidden_size, num_layers, dropout, device):
    encoder = EncoderGRU(
        vocab_size=len(src_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=src_vocab[PAD_TOKEN]
    )

    decoder = DecoderGRUAttention(
        vocab_size=len(tgt_vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=tgt_vocab[PAD_TOKEN]
    )

    return Seq2SeqGRUAttention(encoder=encoder, decoder=decoder).to(device)


def _mask_inference_logits(logits, vocab, mask_special_tokens=True, min_step=None, step=None):
    if not mask_special_tokens:
        return logits

    masked = logits.clone()
    for token in [PAD_TOKEN, BOS_TOKEN, UNK_TOKEN]:
        if token in vocab.token_to_idx:
            masked[..., vocab[token]] = -1e9

    if min_step is not None and step is not None and step < min_step and EOS_TOKEN in vocab.token_to_idx:
        masked[..., vocab[EOS_TOKEN]] = -1e9


    return masked


def _length_normalized_score(score, generated_len, alpha=0.7):
    if alpha is None or alpha <= 0:
        return score

    length = max(1, generated_len)
    penalty = ((5 + length) ** alpha) / (6 ** alpha)
    return score / penalty


def _clean_generated_ids(token_ids, vocab):
    cleaned = []
    for idx in token_ids:
        token = vocab.to_token(idx)
        if token == EOS_TOKEN:
            break
        if token in [PAD_TOKEN, BOS_TOKEN, UNK_TOKEN]:
            continue
        cleaned.append(idx)
    return cleaned


def translate_baseline_greedy(model, source_sentence, src_vocab, tgt_vocab, max_len, device, mask_special_tokens=True, min_decode_len=1):
    model.eval()

    src_ids, src_len = encode_sentence(
        source_sentence,
        src_vocab,
        max_len,
        add_bos=False,
        add_eos=True
    )

    src_ids = src_ids.unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_len], dtype=torch.long).to(device)
    predicted_ids = []

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_ids, src_lengths)
        context = hidden[-1]

        decoder_input = torch.tensor(
            [[tgt_vocab[BOS_TOKEN]]],
            dtype=torch.long,
            device=device
        )

        for step in range(max_len):
            logits, hidden = model.decoder(decoder_input, hidden, context)
            step_logits = _mask_inference_logits(
                logits[:, -1, :],
                tgt_vocab,
                mask_special_tokens=mask_special_tokens,
                min_step=min_decode_len,
                step=step
            )
            next_token_id = step_logits.argmax(dim=-1).item()

            if next_token_id == tgt_vocab[EOS_TOKEN]:
                break

            predicted_ids.append(next_token_id)

            decoder_input = torch.tensor(
                [[next_token_id]],
                dtype=torch.long,
                device=device
            )

    return [tgt_vocab.to_token(idx) for idx in _clean_generated_ids(predicted_ids, tgt_vocab)]


def translate_baseline_beam(model, source_sentence, src_vocab, tgt_vocab, max_len, device, beam_width=3, length_penalty_alpha=0.7, mask_special_tokens=True, min_decode_len=1):
    model.eval()

    src_ids, src_len = encode_sentence(
        source_sentence,
        src_vocab,
        max_len,
        add_bos=False,
        add_eos=True
    )

    src_ids = src_ids.unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_len], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_ids, src_lengths)
        context = hidden[-1]

        beams = [([tgt_vocab[BOS_TOKEN]], hidden, 0.0, False)]

        for step in range(max_len):
            candidates = []

            for token_ids, beam_hidden, score, ended in beams:
                if ended:
                    candidates.append((token_ids, beam_hidden, score, ended))
                    continue

                decoder_input = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
                logits, next_hidden = model.decoder(decoder_input, beam_hidden, context)
                step_logits = _mask_inference_logits(
                    logits[:, -1, :],
                    tgt_vocab,
                    mask_special_tokens=mask_special_tokens,
                    min_step=min_decode_len,
                    step=step
                )

                log_probs = torch.log_softmax(step_logits, dim=-1).squeeze(0)
                top_log_probs, top_indices = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))

                for log_prob, token_idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_ids = token_ids + [token_idx]
                    new_score = score + float(log_prob)
                    new_ended = token_idx == tgt_vocab[EOS_TOKEN]
                    candidates.append((new_ids, next_hidden.detach().clone(), new_score, new_ended))

            candidates.sort(
                key=lambda item: _length_normalized_score(
                    item[2],
                    generated_len=max(1, len(item[0]) - 1),
                    alpha=length_penalty_alpha
                ),
                reverse=True
            )
            beams = candidates[:beam_width]

            if all(ended for _, _, _, ended in beams):
                break

        best_ids = beams[0][0][1:]

    cleaned_ids = _clean_generated_ids(best_ids, tgt_vocab)
    return [tgt_vocab.to_token(idx) for idx in cleaned_ids]


def translate_baseline(model, source_sentence, src_vocab, tgt_vocab, max_len, device, decoding_strategy="greedy", beam_width=3, length_penalty_alpha=0.7, mask_special_tokens=True, min_decode_len=1):
    if decoding_strategy == "beam":
        return translate_baseline_beam(
            model=model,
            source_sentence=source_sentence,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_len=max_len,
            device=device,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
            mask_special_tokens=mask_special_tokens,
            min_decode_len=min_decode_len
        )

    return translate_baseline_greedy(
        model=model,
        source_sentence=source_sentence,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len,
        device=device,
        mask_special_tokens=mask_special_tokens,
        min_decode_len=min_decode_len
    )


def translate_attention_greedy(model, source_sentence, src_vocab, tgt_vocab, max_len, device, mask_special_tokens=True, min_decode_len=1):
    model.eval()

    src_ids, src_len = encode_sentence(
        source_sentence,
        src_vocab,
        max_len,
        add_bos=False,
        add_eos=True
    )

    src_ids = src_ids.unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_len], dtype=torch.long).to(device)

    predicted_ids = []
    attention_rows = []

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_ids, src_lengths)

        decoder_input = torch.tensor(
            [[tgt_vocab[BOS_TOKEN]]],
            dtype=torch.long,
            device=device
        )

        for step in range(max_len):
            logits, hidden, attention_weights = model.decoder.forward_step(
                input_token=decoder_input,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                src_lengths=src_lengths
            )

            step_logits = _mask_inference_logits(
                logits,
                tgt_vocab,
                mask_special_tokens=mask_special_tokens,
                min_step=min_decode_len,
                step=step
            )
            next_token_id = step_logits.argmax(dim=-1).item()

            if next_token_id == tgt_vocab[EOS_TOKEN]:
                break

            predicted_ids.append(next_token_id)
            attention_rows.append(attention_weights.squeeze(0).detach().cpu().numpy())

            decoder_input = torch.tensor(
                [[next_token_id]],
                dtype=torch.long,
                device=device
            )

    predicted_tokens = [tgt_vocab.to_token(idx) for idx in _clean_generated_ids(predicted_ids, tgt_vocab)]
    attention_matrix = np.array(attention_rows)

    if attention_matrix.ndim == 1:
        attention_matrix = attention_matrix.reshape(1, -1)

    source_labels = [
        src_vocab.to_token(idx)
        for idx in src_ids.squeeze(0).detach().cpu().tolist()[:src_len]
    ]

    if attention_matrix.size > 0:
        attention_matrix = attention_matrix[:, :src_len]

    return predicted_tokens, source_labels, attention_matrix


def translate_attention_beam(model, source_sentence, src_vocab, tgt_vocab, max_len, device, beam_width=3, length_penalty_alpha=0.7, mask_special_tokens=True, min_decode_len=1):
    model.eval()

    src_ids, src_len = encode_sentence(
        source_sentence,
        src_vocab,
        max_len,
        add_bos=False,
        add_eos=True
    )

    src_ids = src_ids.unsqueeze(0).to(device)
    src_lengths = torch.tensor([src_len], dtype=torch.long).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_ids, src_lengths)
        beams = [([tgt_vocab[BOS_TOKEN]], hidden, 0.0, False)]

        for step in range(max_len):
            candidates = []

            for token_ids, beam_hidden, score, ended in beams:
                if ended:
                    candidates.append((token_ids, beam_hidden, score, ended))
                    continue

                decoder_input = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
                logits, next_hidden, attention_weights = model.decoder.forward_step(
                    input_token=decoder_input,
                    hidden=beam_hidden,
                    encoder_outputs=encoder_outputs,
                    src_lengths=src_lengths
                )
                step_logits = _mask_inference_logits(
                    logits,
                    tgt_vocab,
                    mask_special_tokens=mask_special_tokens,
                    min_step=min_decode_len,
                    step=step
                )

                log_probs = torch.log_softmax(step_logits, dim=-1).squeeze(0)
                top_log_probs, top_indices = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))

                for log_prob, token_idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_ids = token_ids + [token_idx]
                    new_score = score + float(log_prob)
                    new_ended = token_idx == tgt_vocab[EOS_TOKEN]
                    candidates.append((new_ids, next_hidden.detach().clone(), new_score, new_ended))

            candidates.sort(
                key=lambda item: _length_normalized_score(
                    item[2],
                    generated_len=max(1, len(item[0]) - 1),
                    alpha=length_penalty_alpha
                ),
                reverse=True
            )
            beams = candidates[:beam_width]

            if all(ended for _, _, _, ended in beams):
                break

        best_ids = beams[0][0][1:]

    cleaned_ids = _clean_generated_ids(best_ids, tgt_vocab)
    return [tgt_vocab.to_token(idx) for idx in cleaned_ids]


def translate_attention(model, source_sentence, src_vocab, tgt_vocab, max_len, device, decoding_strategy="greedy", beam_width=3, length_penalty_alpha=0.7, mask_special_tokens=True, min_decode_len=1):
    if decoding_strategy == "beam":
        pred_tokens = translate_attention_beam(
            model=model,
            source_sentence=source_sentence,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_len=max_len,
            device=device,
            beam_width=beam_width,
            length_penalty_alpha=length_penalty_alpha,
            mask_special_tokens=mask_special_tokens,
            min_decode_len=min_decode_len
        )


        src_ids, src_len = encode_sentence(source_sentence, src_vocab, max_len, add_bos=False, add_eos=True)
        source_labels = [src_vocab.to_token(idx) for idx in src_ids.tolist()[:src_len]]
        return pred_tokens, source_labels, np.empty((0, src_len))

    return translate_attention_greedy(
        model=model,
        source_sentence=source_sentence,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=max_len,
        device=device,
        mask_special_tokens=mask_special_tokens,
        min_decode_len=min_decode_len
    )


def evaluate_translation_quality(
    model,
    val_pairs,
    src_vocab,
    tgt_vocab,
    max_len,
    device,
    history_df,
    model_name,
    source_column,
    target_column,
    prediction_column,
    use_attention=False,
    decoding_strategy="greedy",
    beam_width=3,
    length_penalty_alpha=0.7,
    mask_special_tokens=True,
    min_decode_len=1
):
    records = []

    for source_sentence, target_sentence in tqdm(val_pairs, desc=f"Evaluating {model_name}"):
        if use_attention:
            pred_tokens, source_labels, attention_matrix = translate_attention(
                model=model,
                source_sentence=source_sentence,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                max_len=max_len,
                device=device,
                decoding_strategy=decoding_strategy,
                beam_width=beam_width,
                length_penalty_alpha=length_penalty_alpha,
                mask_special_tokens=mask_special_tokens,
                min_decode_len=min_decode_len
            )
        else:
            pred_tokens = translate_baseline(
                model=model,
                source_sentence=source_sentence,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                max_len=max_len,
                device=device,
                decoding_strategy=decoding_strategy,
                beam_width=beam_width,
                length_penalty_alpha=length_penalty_alpha,
                mask_special_tokens=mask_special_tokens,
                min_decode_len=min_decode_len
            )

        target_tokens = tokenize(target_sentence)
        exact_match = compute_exact_match(pred_tokens, target_tokens)
        bleu4 = compute_bleu4(pred_tokens, target_tokens)

        records.append({
            source_column: source_sentence,
            target_column: tokens_to_sentence(target_tokens),
            prediction_column: tokens_to_sentence(pred_tokens),
            "Exact Match": exact_match,
            "BLEU-4": bleu4
        })

    results_df = pd.DataFrame(records)
    metrics_df = make_translation_metrics_df(results_df, history_df, model_name)

    return results_df, metrics_df


def prepare_translation_direction(
    pairs,
    train_pairs,
    val_pairs,
    direction,
    source_language,
    target_language,
    src_index,
    tgt_index,
    common_config
):
    direction_train_pairs = [(pair[src_index], pair[tgt_index]) for pair in train_pairs]
    direction_val_pairs = [(pair[src_index], pair[tgt_index]) for pair in val_pairs]


    src_vocab, tgt_vocab, source_tokens, target_tokens = build_parallel_vocabs(
        pairs=direction_train_pairs,
        src_index=0,
        tgt_index=1,
        min_freq=common_config["min_freq"]
    )

    train_dataset, val_dataset, train_loader, val_loader = make_translation_dataloaders(
        train_pairs=direction_train_pairs,
        val_pairs=direction_val_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=common_config["max_len"],
        batch_size=common_config["batch_size"]
    )

    diagnostic_df, src_oov, tgt_oov = compute_oov_diagnostic(
        train_pairs=direction_train_pairs,
        val_pairs=direction_val_pairs,
        source_label=source_language,
        target_label=target_language,
        total_pairs_count=len(pairs)
    )

    return {
        "direction": direction,
        "source_language": source_language,
        "target_language": target_language,
        "src_index": src_index,
        "tgt_index": tgt_index,
        "train_pairs": direction_train_pairs,
        "val_pairs": direction_val_pairs,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "diagnostic_df": diagnostic_df,
        "src_oov": src_oov,
        "tgt_oov": tgt_oov,
        "vocab_table": make_vocab_table(
            vocab_names=[f"{source_language} Source Vocabulary", f"{target_language} Target Vocabulary"],
            vocabs=[src_vocab, tgt_vocab]
        ),
        "split_table": make_split_table(
            train_pairs=direction_train_pairs,
            val_pairs=direction_val_pairs,
            train_percent=int(common_config["train_size"] * 100),
            val_percent=int(common_config["val_size"] * 100)
        ),
        "preview_df": pd.DataFrame(
            [(pair[src_index], pair[tgt_index]) for pair in pairs[:10]],
            columns=[source_language, target_language]
        )
    }


def make_config_hyperparameter_table(config, common_config):
    choices = [
        "Architecture",
        "Source Language",
        "Target Language",
        "Embedding Size",
        "Hidden Size",
        "GRU Layers",
        "Dropout",
        "Batch Size",
        "Epochs",
        "Learning Rate",
        "Learning-Rate Scheduler",
        "Optimizer",
        "Weight Decay",
        "Loss Function",
        "Label Smoothing",
        "Gradient Clipping",
        "Early Stopping Patience",
        "Model Selection Metric",
        "Selection Min Delta",
        "Overfit Penalty Weight",
        "Decoding Strategy",
        "Beam Width",
        "Length Penalty Alpha",
        "Minimum Decode Length",
        "Mask Special Tokens During Decoding",
        "Max Sequence Length",
    ]

    values = [
        config["architecture"],
        config["source_language"],
        config["target_language"],
        common_config["embed_size"],
        common_config["hidden_size"],
        common_config["num_layers"],
        config["dropout"],
        common_config["batch_size"],
        config["epochs"],
        config["learning_rate"],
        config["lr_scheduler"],
        common_config["optimizer"],
        config["weight_decay"],
        common_config["loss_function"],
        common_config["label_smoothing"],
        common_config["grad_clip"],
        config["early_stopping_patience"],
        "validation token accuracy minus train-validation loss gap penalty",
        config["selection_min_delta"],
        config["overfit_penalty_weight"],
        config["decode_strategy"],
        config["beam_width"],
        config["length_penalty_alpha"],
        config["min_decode_len"],
        config["mask_special_tokens"],
        common_config["max_len"],
    ]

    if config.get("attention_type") is not None:
        choices.insert(1, "Attention Type")
        values.insert(1, config["attention_type"])

    return make_hyperparameter_table(choices, values)


def build_model_from_config(config, src_vocab, tgt_vocab, common_config, device):
    if config["architecture_type"] == "baseline":
        return build_baseline_seq2seq_model(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            embed_size=common_config["embed_size"],
            hidden_size=common_config["hidden_size"],
            num_layers=common_config["num_layers"],
            dropout=config["dropout"],
            device=device
        )

    if config["architecture_type"] == "attention":
        return build_attention_seq2seq_model(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            embed_size=common_config["embed_size"],
            hidden_size=common_config["hidden_size"],
            num_layers=common_config["num_layers"],
            dropout=config["dropout"],
            device=device
        )

    raise ValueError(f"Unknown architecture_type: {config['architecture_type']}")


def run_configured_experiment(config, resources_by_direction, common_config, device, display_setup=True):
    resource = resources_by_direction[config["direction"]]

    model = build_model_from_config(
        config=config,
        src_vocab=resource["src_vocab"],
        tgt_vocab=resource["tgt_vocab"],
        common_config=common_config,
        device=device
    )

    criterion, optimizer = make_loss_and_optimizer(
        model=model,
        tgt_vocab=resource["tgt_vocab"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = make_lr_scheduler(optimizer, config)

    hyperparameter_table = make_config_hyperparameter_table(config, common_config)

    if display_setup:
        print(f"{config['model_name']} setup")
        display(hyperparameter_table)
        print(model)
        print("Trainable parameters:", count_trainable_parameters(model))

    history_df = train_seq2seq_model(
        model=model,
        train_loader=resource["train_loader"],
        val_loader=resource["val_loader"],
        optimizer=optimizer,
        criterion=criterion,
        epochs=config["epochs"],
        device=device,
        tgt_pad_idx=resource["tgt_vocab"][PAD_TOKEN],
        patience=config["early_stopping_patience"],
        train_desc=f"Training {config['problem']} {config['model_label']}",
        val_desc=f"Validating {config['problem']} {config['model_label']}",
        grad_clip=common_config["grad_clip"],
        scheduler=scheduler,
        min_delta=config.get("selection_min_delta", common_config.get("selection_min_delta", 0.002)),
        overfit_penalty_weight=config.get("overfit_penalty_weight", common_config.get("overfit_penalty_weight", 0.01))
    )

    validation_results_df, metrics_df = evaluate_translation_quality(
        model=model,
        val_pairs=resource["val_pairs"],
        src_vocab=resource["src_vocab"],
        tgt_vocab=resource["tgt_vocab"],
        max_len=common_config["max_len"],
        device=device,
        history_df=history_df,
        model_name=config["model_name"],
        source_column=config["source_column"],
        target_column=config["target_column"],
        prediction_column=config["prediction_column"],
        use_attention=(config["architecture_type"] == "attention"),
        decoding_strategy=config.get("decode_strategy", "greedy"),
        beam_width=config.get("beam_width", 1),
        length_penalty_alpha=config.get("length_penalty_alpha", 0.0),
        mask_special_tokens=config.get("mask_special_tokens", True),
        min_decode_len=config.get("min_decode_len", 1)
    )

    result = {
        "config": config,
        "resource": resource,
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "hyperparameter_table": hyperparameter_table,
        "history_df": history_df,
        "best_epoch_df": make_best_epoch_table(history_df),
        "validation_results_df": validation_results_df,
        "metrics_df": metrics_df,
        "sample_translations_df": top_bleu_examples(validation_results_df, n=5),
    }

    return result


def make_comparison_rows_from_experiments(experiment_results, experiment_keys):
    rows = []
    for key in experiment_keys:
        experiment = experiment_results[key]
        config = experiment["config"]
        rows.append({
            "problem": config["problem"],
            "model_label": config["model_label"],
            "direction": config["direction"],
            "history_df": experiment["history_df"],
            "results_df": experiment["validation_results_df"],
            "model": experiment["model"],
        })
    return rows


def make_experiment_comparison(experiment_results, experiment_keys):
    return make_model_comparison(make_comparison_rows_from_experiments(experiment_results, experiment_keys))


def make_direction_level_comparison(comparison_df):
    metric_columns = ["Best Validation Loss"]
    if "Validation Token Accuracy" in comparison_df.columns:
        metric_columns.append("Validation Token Accuracy")
    metric_columns.extend(["Exact Match Accuracy", "Average BLEU-4"])

    return (
        comparison_df
        .groupby("Direction")[metric_columns]
        .mean()
        .reset_index()
    )


def display_single_experiment_all_results(experiment, total_pairs, common_config):
    config = experiment["config"]
    resource = experiment["resource"]
    history_df = experiment["history_df"]
    results_df = experiment["validation_results_df"]

    print("=" * 80)
    print(f"{config['model_name'].upper()} — ALL METRICS AND RESULTS")
    print("=" * 80)

    dataset_table = make_dataset_table(
        total_pairs=total_pairs,
        train_pairs=resource["train_pairs"],
        val_pairs=resource["val_pairs"],
        train_size=common_config["train_size"],
        val_size=common_config["val_size"],
        max_len=common_config["max_len"],
        random_seed=common_config["random_seed"]
    )

    model_size_table = make_model_size_table(
        model=experiment["model"],
        src_vocab=resource["src_vocab"],
        tgt_vocab=resource["tgt_vocab"],
        embed_size=common_config["embed_size"],
        hidden_size=common_config["hidden_size"],
        num_layers=common_config["num_layers"]
    )

    training_table = make_training_table(history_df)
    exact_match_table = make_exact_match_table(results_df)
    bleu_table = make_bleu_distribution_table(results_df)
    top_examples = top_bleu_examples(results_df, n=10)
    low_examples = low_bleu_examples(results_df, n=10)
    exact_examples = exact_match_examples(results_df)
    compact_table = make_compact_report_table(
        problem=config["problem"],
        direction=config["direction"],
        architecture=config["architecture"],
        history_df=history_df,
        results_df=results_df,
        model=experiment["model"]
    )

    sections = [
        ("1. Dataset and Split Table", dataset_table),
        ("2. Vocabulary Table", resource["vocab_table"]),
        ("3. Out-of-Vocabulary Diagnostic Table", resource["diagnostic_df"]),
        ("4. Hyperparameter and Design Choice Table", experiment["hyperparameter_table"]),
        ("5. Model Size Table", model_size_table),
        ("6. Full Training History", history_df),
        ("7. Best Epoch and Final Epoch Table", training_table),
        ("8. Final Translation Metrics", experiment["metrics_df"]),
        ("9. Exact Match Count Table", exact_match_table),
        ("10. BLEU-4 Distribution Table", bleu_table),
        ("11. Top 10 Validation Examples by BLEU-4", top_examples),
        ("12. Lowest 10 Validation Examples by BLEU-4", low_examples),
    ]

    for title, df in sections:
        print(f"\n{title}")
        display(df)

    print("\n13. Exact Match Examples")
    if len(exact_examples) > 0:
        display(exact_examples)
    else:
        print(f"No exact match examples were found for {config['problem']}.")

    print("\n14. Full Validation Translation Results")
    display(results_df)

    print("\n15. Compact Final Table for Report Table")
    display(compact_table)

    return {
        "dataset_table": dataset_table,
        "model_size_table": model_size_table,
        "training_table": training_table,
        "exact_match_table": exact_match_table,
        "bleu_table": bleu_table,
        "top_examples": top_examples,
        "low_examples": low_examples,
        "exact_examples": exact_examples,
        "compact_table": compact_table,
    }
```

**Output**

```text
Shared baseline and attention model classes loaded successfully.
```

## Plotting and Display Functions

```python
def plot_loss_curves(history_df, title):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Training Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss_curves_with_best_epoch(history_df, title):
    best_row = best_epoch_row(history_df)
    best_epoch = int(best_row["epoch"])
    best_val_loss = best_row["val_loss"]

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Training Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation Loss")
    plt.axvline(best_epoch, linestyle="--", label=f"Best Selected Epoch = {best_epoch}")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Best selected epoch: {best_epoch}")
    print(f"Validation loss at selected epoch: {best_val_loss:.4f}")


def plot_attention_map(
    source_tokens,
    predicted_tokens,
    attention_matrix,
    title,
    source_axis_label="Source Tokens",
    target_axis_label="Predicted Tokens"
):
    if len(predicted_tokens) == 0 or attention_matrix.size == 0:
        print("No predicted tokens available for attention map.")
        return

    plt.figure(figsize=(max(8, len(source_tokens) * 0.8), max(5, len(predicted_tokens) * 0.45)))
    plt.imshow(attention_matrix, aspect="auto")

    plt.xticks(
        ticks=np.arange(len(source_tokens)),
        labels=source_tokens,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        ticks=np.arange(len(predicted_tokens)),
        labels=predicted_tokens
    )

    plt.xlabel(source_axis_label)
    plt.ylabel(target_axis_label)
    plt.title(title)
    plt.colorbar(label="Attention Weight")
    plt.tight_layout()
    plt.show()


def display_problem_metrics(metrics_df, title):
    print(title)
    display(metrics_df)


def display_sample_translations(sample_df, title):
    print(title)
    display(sample_df)


def display_exact_examples_or_message(results_df, title, no_examples_message):
    print(title)
    examples = exact_match_examples(results_df)

    if len(examples) > 0:
        display(examples)
    else:
        print(no_examples_message)


def display_attention_examples(
    results_df,
    model,
    src_vocab,
    tgt_vocab,
    max_len,
    device,
    source_column,
    target_column,
    prediction_column,
    title_prefix,
    source_axis_label,
    target_axis_label,
    n=2
):
    examples = top_bleu_examples(results_df, n=n)

    for example_number, row in examples.iterrows():
        source_sentence = row[source_column]
        target_sentence = row[target_column]
        predicted_sentence = row[prediction_column]
        bleu_score = row["BLEU-4"]
        exact_match = row["Exact Match"]

        pred_tokens, source_labels, attention_matrix = translate_attention(
            model=model,
            source_sentence=source_sentence,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_len=max_len,
            device=device
        )

        print(f"\n{title_prefix} {example_number + 1}")
        print(f"{source_column}:", source_sentence)
        print(f"{target_column}:", target_sentence)
        print(f"{prediction_column}:", predicted_sentence)
        print("Exact Match:", exact_match)
        print("BLEU-4:", bleu_score)

        plot_attention_map(
            source_tokens=source_labels,
            predicted_tokens=pred_tokens,
            attention_matrix=attention_matrix,
            title=f"{title_prefix} {example_number + 1}",
            source_axis_label=source_axis_label,
            target_axis_label=target_axis_label
        )


def plot_configured_experiment_outputs(experiment, title=None, show_best_epoch=True):
    config = experiment["config"]
    if title is None:
        title = f"{config['problem']}: {config['direction']} {config['model_label']} Training vs Validation Loss"

    if show_best_epoch:
        plot_loss_curves_with_best_epoch(experiment["history_df"], title=title)
    else:
        plot_loss_curves(experiment["history_df"], title=title)

    display_problem_metrics(
        metrics_df=experiment["metrics_df"],
        title=f"{config['problem']} Final Validation Metrics"
    )

    display_sample_translations(
        sample_df=experiment["sample_translations_df"],
        title=f"{config['problem']} Qualitative Validation Examples"
    )


def display_configured_attention_examples(experiment, n=2):
    config = experiment["config"]

    if config["architecture_type"] != "attention":
        print(f"{config['model_name']} does not use attention, so no attention maps are available.")
        return

    resource = experiment["resource"]

    display_attention_examples(
        results_df=experiment["validation_results_df"],
        model=experiment["model"],
        src_vocab=resource["src_vocab"],
        tgt_vocab=resource["tgt_vocab"],
        max_len=COMMON_CONFIG["max_len"],
        device=DEVICE,
        source_column=config["source_column"],
        target_column=config["target_column"],
        prediction_column=config["prediction_column"],
        title_prefix=f"{config['problem']} Attention Map Example",
        source_axis_label=f"{config['source_language']} Source Tokens",
        target_axis_label=f"Predicted {config['target_language']} Tokens",
        n=n
    )


def display_configured_comparison(experiment_results, experiment_keys, title):
    comparison_df = make_experiment_comparison(experiment_results, experiment_keys)
    display_problem_metrics(metrics_df=comparison_df, title=title)
    return comparison_df


def display_configured_improvement(baseline_experiment, candidate_experiment, title):
    improvement_df = make_improvement_table(
        baseline_history=baseline_experiment["history_df"],
        candidate_history=candidate_experiment["history_df"],
        baseline_results=baseline_experiment["validation_results_df"],
        candidate_results=candidate_experiment["validation_results_df"],
        baseline_label=baseline_experiment["config"]["model_name"],
        candidate_label=candidate_experiment["config"]["model_name"]
    )
    display_problem_metrics(metrics_df=improvement_df, title=title)
    return improvement_df
```

## Dataset Setup

```python
upload_dataset_if_needed(DATA_PATH)

pairs = read_translation_pairs(DATA_PATH, max_examples=MAX_EXAMPLES)
print("Total sentence pairs loaded:", len(pairs))

display(pd.DataFrame(pairs[:10], columns=["English", "French"]))


all_indices, train_indices, val_indices, train_pairs, val_pairs = create_shared_split(
    pairs=pairs,
    train_size=COMMON_CONFIG["train_size"],
    val_size=COMMON_CONFIG["val_size"],
    random_seed=COMMON_CONFIG["random_seed"]
)

translation_resources = {
    "English-to-French": prepare_translation_direction(
        pairs=pairs,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        direction="English-to-French",
        source_language="English",
        target_language="French",
        src_index=0,
        tgt_index=1,
        common_config=COMMON_CONFIG
    ),
    "French-to-English": prepare_translation_direction(
        pairs=pairs,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        direction="French-to-English",
        source_language="French",
        target_language="English",
        src_index=1,
        tgt_index=0,
        common_config=COMMON_CONFIG
    ),
}

for direction, resource in translation_resources.items():
    print("=" * 80)
    print(direction)
    print("=" * 80)
    print("Training pairs:", len(resource["train_pairs"]))
    print("Validation pairs:", len(resource["val_pairs"]))
    print("Training batches:", len(resource["train_loader"]))
    print("Validation batches:", len(resource["val_loader"]))
    display(resource["split_table"])
    display(resource["vocab_table"])
    display(resource["diagnostic_df"])
```

**Output**

```text
Dataset ready: vast_english_french.txt
Total sentence pairs loaded: 555
```

| English                    | French                         |
|:---------------------------|:-------------------------------|
| i am cold                  | j'ai froid                     |
| you are tired              | tu es fatigué                  |
| he is hungry               | il a faim                      |
| she is happy               | elle est heureuse              |
| we are friends             | nous sommes amis               |
| they are students          | ils sont étudiants             |
| the cat is sleeping        | le chat dort                   |
| the sun is shining         | le soleil brille               |
| we love music              | nous aimons la musique         |
| she speaks french fluently | elle parle français couramment |

```text
================================================================================
English-to-French
================================================================================
Training pairs: 444
Validation pairs: 111
Training batches: 14
Validation batches: 4
```

| Split      |   Percent |   Sentence Pairs |
|:-----------|----------:|-----------------:|
| Training   |        80 |              444 |
| Validation |        20 |              111 |

| Vocabulary                |   Size |
|:--------------------------|-------:|
| English Source Vocabulary |    894 |
| French Target Vocabulary  |    994 |

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique English training source tokens             | 890      |
| Unique French training target tokens              | 990      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV %                          |  34.2618 |
| Validation French OOV %                           |  38.0577 |

```text
================================================================================
French-to-English
================================================================================
Training pairs: 444
Validation pairs: 111
Training batches: 14
Validation batches: 4
```

| Split      |   Percent |   Sentence Pairs |
|:-----------|----------:|-----------------:|
| Training   |        80 |              444 |
| Validation |        20 |              111 |

| Vocabulary                |   Size |
|:--------------------------|-------:|
| French Source Vocabulary  |    994 |
| English Target Vocabulary |    894 |

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique French training source tokens              | 990      |
| Unique English training target tokens             | 890      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV %                           |  38.0577 |
| Validation English OOV %                          |  34.2618 |

## Problem 1: English-to-French Baseline GRU

```python
experiment_results = {}
experiment_results["p1_baseline"] = run_configured_experiment(
    config=EXPERIMENT_CONFIGS["p1_baseline"],
    resources_by_direction=translation_resources,
    common_config=COMMON_CONFIG,
    device=DEVICE
)

display(experiment_results["p1_baseline"]["history_df"])
display(experiment_results["p1_baseline"]["best_epoch_df"])
display(experiment_results["p1_baseline"]["metrics_df"])
display(experiment_results["p1_baseline"]["sample_translations_df"])
```

**Output**

```text
Problem 1 English-to-French Baseline GRU setup
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | Baseline GRU Encoder-Decoder with Fixed Context   |
| Source Language                     | English                                           |
| Target Language                     | French                                            |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
Seq2SeqGRUBaseline(
  (encoder): EncoderGRU(
    (embedding): Embedding(894, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(48, 96, batch_first=True)
  )
  (decoder): DecoderGRUFixedContext(
    (embedding): Embedding(994, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(144, 96, batch_first=True)
    (fc_out): Linear(in_features=96, out_features=994, bias=True)
  )
)
Trainable parameters: 298786
```

```text
Epoch 01/50 | Train Loss: 6.7330 | Val Loss: 6.3689 | Train Tok Acc: 0.0804 | Val Tok Acc: 0.1323 | Loss Gap: -0.3641 | Score: 0.1323 | LR: 0.001000
  New best validation-first score: 0.1323
```

```text
Epoch 02/50 | Train Loss: 5.9941 | Val Loss: 5.9735 | Train Tok Acc: 0.1271 | Val Tok Acc: 0.1323 | Loss Gap: -0.0206 | Score: 0.1323 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 03/50 | Train Loss: 5.6004 | Val Loss: 5.8511 | Train Tok Acc: 0.1409 | Val Tok Acc: 0.1514 | Loss Gap: 0.2507 | Score: 0.1489 | LR: 0.001000
  New best validation-first score: 0.1489
```

```text
Epoch 04/50 | Train Loss: 5.4559 | Val Loss: 5.7417 | Train Tok Acc: 0.1463 | Val Tok Acc: 0.1514 | Loss Gap: 0.2859 | Score: 0.1485 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 05/50 | Train Loss: 5.3459 | Val Loss: 5.6376 | Train Tok Acc: 0.1460 | Val Tok Acc: 0.1514 | Loss Gap: 0.2917 | Score: 0.1485 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 06/50 | Train Loss: 5.2542 | Val Loss: 5.5730 | Train Tok Acc: 0.1469 | Val Tok Acc: 0.1490 | Loss Gap: 0.3188 | Score: 0.1458 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 07/50 | Train Loss: 5.1746 | Val Loss: 5.5156 | Train Tok Acc: 0.1534 | Val Tok Acc: 0.1561 | Loss Gap: 0.3410 | Score: 0.1527 | LR: 0.001000
  New best validation-first score: 0.1527
```

```text
Epoch 08/50 | Train Loss: 5.1001 | Val Loss: 5.4716 | Train Tok Acc: 0.1623 | Val Tok Acc: 0.1633 | Loss Gap: 0.3715 | Score: 0.1596 | LR: 0.001000
  New best validation-first score: 0.1596
```

```text
Epoch 09/50 | Train Loss: 5.0303 | Val Loss: 5.4447 | Train Tok Acc: 0.1663 | Val Tok Acc: 0.1728 | Loss Gap: 0.4144 | Score: 0.1687 | LR: 0.001000
  New best validation-first score: 0.1687
```

```text
Epoch 10/50 | Train Loss: 4.9596 | Val Loss: 5.3790 | Train Tok Acc: 0.1781 | Val Tok Acc: 0.1871 | Loss Gap: 0.4195 | Score: 0.1829 | LR: 0.001000
  New best validation-first score: 0.1829
```

```text
Epoch 11/50 | Train Loss: 4.8797 | Val Loss: 5.3370 | Train Tok Acc: 0.1815 | Val Tok Acc: 0.1871 | Loss Gap: 0.4573 | Score: 0.1826 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 12/50 | Train Loss: 4.8069 | Val Loss: 5.2788 | Train Tok Acc: 0.1849 | Val Tok Acc: 0.1883 | Loss Gap: 0.4719 | Score: 0.1836 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 13/50 | Train Loss: 4.7390 | Val Loss: 5.2343 | Train Tok Acc: 0.1930 | Val Tok Acc: 0.1919 | Loss Gap: 0.4953 | Score: 0.1869 | LR: 0.001000
  New best validation-first score: 0.1869
```

```text
Epoch 14/50 | Train Loss: 4.6674 | Val Loss: 5.1985 | Train Tok Acc: 0.1990 | Val Tok Acc: 0.2062 | Loss Gap: 0.5311 | Score: 0.2009 | LR: 0.001000
  New best validation-first score: 0.2009
```

```text
Epoch 15/50 | Train Loss: 4.6079 | Val Loss: 5.1631 | Train Tok Acc: 0.2030 | Val Tok Acc: 0.2038 | Loss Gap: 0.5551 | Score: 0.1983 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 16/50 | Train Loss: 4.5392 | Val Loss: 5.1204 | Train Tok Acc: 0.2076 | Val Tok Acc: 0.2133 | Loss Gap: 0.5812 | Score: 0.2075 | LR: 0.001000
  New best validation-first score: 0.2075
```

```text
Epoch 17/50 | Train Loss: 4.4771 | Val Loss: 5.0940 | Train Tok Acc: 0.2150 | Val Tok Acc: 0.2157 | Loss Gap: 0.6169 | Score: 0.2096 | LR: 0.001000
  New best validation-first score: 0.2096
```

```text
Epoch 18/50 | Train Loss: 4.4105 | Val Loss: 5.0770 | Train Tok Acc: 0.2159 | Val Tok Acc: 0.2157 | Loss Gap: 0.6665 | Score: 0.2091 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 19/50 | Train Loss: 4.3574 | Val Loss: 5.0443 | Train Tok Acc: 0.2173 | Val Tok Acc: 0.2145 | Loss Gap: 0.6869 | Score: 0.2077 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 20/50 | Train Loss: 4.3058 | Val Loss: 5.0364 | Train Tok Acc: 0.2239 | Val Tok Acc: 0.2181 | Loss Gap: 0.7306 | Score: 0.2108 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 21/50 | Train Loss: 4.2649 | Val Loss: 5.0271 | Train Tok Acc: 0.2244 | Val Tok Acc: 0.2205 | Loss Gap: 0.7622 | Score: 0.2129 | LR: 0.001000
  New best validation-first score: 0.2129
```

```text
Epoch 22/50 | Train Loss: 4.2211 | Val Loss: 4.9822 | Train Tok Acc: 0.2328 | Val Tok Acc: 0.2277 | Loss Gap: 0.7611 | Score: 0.2200 | LR: 0.001000
  New best validation-first score: 0.2200
```

```text
Epoch 23/50 | Train Loss: 4.1517 | Val Loss: 4.9543 | Train Tok Acc: 0.2416 | Val Tok Acc: 0.2312 | Loss Gap: 0.8026 | Score: 0.2232 | LR: 0.001000
  New best validation-first score: 0.2232
```

```text
Epoch 24/50 | Train Loss: 4.1018 | Val Loss: 4.9465 | Train Tok Acc: 0.2439 | Val Tok Acc: 0.2336 | Loss Gap: 0.8447 | Score: 0.2252 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 25/50 | Train Loss: 4.0611 | Val Loss: 4.9169 | Train Tok Acc: 0.2496 | Val Tok Acc: 0.2408 | Loss Gap: 0.8558 | Score: 0.2322 | LR: 0.001000
  New best validation-first score: 0.2322
```

```text
Epoch 26/50 | Train Loss: 4.0229 | Val Loss: 4.9135 | Train Tok Acc: 0.2494 | Val Tok Acc: 0.2396 | Loss Gap: 0.8906 | Score: 0.2307 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 27/50 | Train Loss: 3.9718 | Val Loss: 4.8967 | Train Tok Acc: 0.2542 | Val Tok Acc: 0.2443 | Loss Gap: 0.9249 | Score: 0.2351 | LR: 0.001000
  New best validation-first score: 0.2351
```

```text
Epoch 28/50 | Train Loss: 3.9328 | Val Loss: 4.8762 | Train Tok Acc: 0.2622 | Val Tok Acc: 0.2455 | Loss Gap: 0.9434 | Score: 0.2361 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 29/50 | Train Loss: 3.8907 | Val Loss: 4.8576 | Train Tok Acc: 0.2668 | Val Tok Acc: 0.2479 | Loss Gap: 0.9669 | Score: 0.2382 | LR: 0.001000
  New best validation-first score: 0.2382
```

```text
Epoch 30/50 | Train Loss: 3.8363 | Val Loss: 4.8520 | Train Tok Acc: 0.2708 | Val Tok Acc: 0.2467 | Loss Gap: 1.0158 | Score: 0.2366 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 31/50 | Train Loss: 3.8035 | Val Loss: 4.8314 | Train Tok Acc: 0.2734 | Val Tok Acc: 0.2515 | Loss Gap: 1.0278 | Score: 0.2412 | LR: 0.001000
  New best validation-first score: 0.2412
```

```text
Epoch 32/50 | Train Loss: 3.7679 | Val Loss: 4.8283 | Train Tok Acc: 0.2760 | Val Tok Acc: 0.2479 | Loss Gap: 1.0604 | Score: 0.2373 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 33/50 | Train Loss: 3.7301 | Val Loss: 4.8072 | Train Tok Acc: 0.2774 | Val Tok Acc: 0.2574 | Loss Gap: 1.0771 | Score: 0.2467 | LR: 0.001000
  New best validation-first score: 0.2467
```

```text
Epoch 34/50 | Train Loss: 3.6863 | Val Loss: 4.7990 | Train Tok Acc: 0.2886 | Val Tok Acc: 0.2539 | Loss Gap: 1.1127 | Score: 0.2427 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 35/50 | Train Loss: 3.6533 | Val Loss: 4.7869 | Train Tok Acc: 0.2897 | Val Tok Acc: 0.2622 | Loss Gap: 1.1336 | Score: 0.2509 | LR: 0.001000
  New best validation-first score: 0.2509
```

```text
Epoch 36/50 | Train Loss: 3.6094 | Val Loss: 4.7791 | Train Tok Acc: 0.2943 | Val Tok Acc: 0.2551 | Loss Gap: 1.1697 | Score: 0.2434 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 37/50 | Train Loss: 3.5797 | Val Loss: 4.7631 | Train Tok Acc: 0.2957 | Val Tok Acc: 0.2610 | Loss Gap: 1.1834 | Score: 0.2492 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 38/50 | Train Loss: 3.5580 | Val Loss: 4.7631 | Train Tok Acc: 0.3040 | Val Tok Acc: 0.2574 | Loss Gap: 1.2051 | Score: 0.2454 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 39/50 | Train Loss: 3.5201 | Val Loss: 4.7477 | Train Tok Acc: 0.3052 | Val Tok Acc: 0.2646 | Loss Gap: 1.2276 | Score: 0.2523 | LR: 0.001000
  No useful validation improvement. Patience: 4/4
Early stopping triggered.
Loaded best model from epoch 35 with validation-first score: 0.2509
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.73301 |    6.36895 |               0.080447 |             0.1323   |  -0.364059 |            -0.051854 |          0.1323   |           0.001 |
|       2 |      5.99415 |    5.97351 |               0.127111 |             0.1323   |  -0.020637 |            -0.005189 |          0.1323   |           0.001 |
|       3 |      5.60038 |    5.85112 |               0.140853 |             0.151371 |   0.250741 |            -0.010518 |          0.148863 |           0.001 |
|       4 |      5.45585 |    5.74174 |               0.146293 |             0.151371 |   0.285894 |            -0.005078 |          0.148512 |           0.001 |
|       5 |      5.34588 |    5.63758 |               0.146006 |             0.151371 |   0.291707 |            -0.005364 |          0.148454 |           0.001 |
|       6 |      5.25421 |    5.57303 |               0.146865 |             0.148987 |   0.318821 |            -0.002122 |          0.145799 |           0.001 |
|       7 |      5.17459 |    5.5156  |               0.15345  |             0.156138 |   0.341007 |            -0.002689 |          0.152728 |           0.001 |
|       8 |      5.10013 |    5.4716  |               0.162325 |             0.16329  |   0.371472 |            -0.000965 |          0.159575 |           0.001 |
|       9 |      5.0303  |    5.44468 |               0.166333 |             0.172825 |   0.414376 |            -0.006492 |          0.168681 |           0.001 |
|      10 |      4.95956 |    5.37904 |               0.17807  |             0.187128 |   0.41948  |            -0.009057 |          0.182933 |           0.001 |
|      11 |      4.87969 |    5.33702 |               0.181506 |             0.187128 |   0.457336 |            -0.005622 |          0.182554 |           0.001 |
|      12 |      4.80687 |    5.27877 |               0.184941 |             0.188319 |   0.471901 |            -0.003378 |          0.1836   |           0.001 |
|      13 |      4.739   |    5.23428 |               0.192957 |             0.191895 |   0.495288 |             0.001062 |          0.186942 |           0.001 |
|      14 |      4.66742 |    5.19854 |               0.198969 |             0.206198 |   0.531122 |            -0.007228 |          0.200887 |           0.001 |
|      15 |      4.60792 |    5.16306 |               0.202977 |             0.203814 |   0.555134 |            -0.000837 |          0.198263 |           0.001 |
|      16 |      4.53916 |    5.12035 |               0.207558 |             0.213349 |   0.581189 |            -0.005791 |          0.207537 |           0.001 |
|      17 |      4.47709 |    5.09399 |               0.215001 |             0.215733 |   0.616899 |            -0.000732 |          0.209564 |           0.001 |
|      18 |      4.41045 |    5.07696 |               0.21586  |             0.215733 |   0.666507 |             0.000127 |          0.209068 |           0.001 |
|      19 |      4.35737 |    5.04429 |               0.217292 |             0.214541 |   0.686925 |             0.002751 |          0.207672 |           0.001 |
|      20 |      4.30576 |    5.03636 |               0.223876 |             0.218117 |   0.7306   |             0.00576  |          0.210811 |           0.001 |
|      21 |      4.2649  |    5.02712 |               0.224449 |             0.220501 |   0.762219 |             0.003948 |          0.212878 |           0.001 |
|      22 |      4.22112 |    4.98222 |               0.232751 |             0.227652 |   0.761099 |             0.005099 |          0.220041 |           0.001 |
|      23 |      4.1517  |    4.95432 |               0.241626 |             0.231228 |   0.802616 |             0.010398 |          0.223201 |           0.001 |
|      24 |      4.1018  |    4.94648 |               0.243916 |             0.233611 |   0.84468  |             0.010305 |          0.225165 |           0.001 |
|      25 |      4.06109 |    4.91691 |               0.249642 |             0.240763 |   0.855812 |             0.008879 |          0.232205 |           0.001 |
|      26 |      4.02289 |    4.91353 |               0.249356 |             0.239571 |   0.890639 |             0.009785 |          0.230665 |           0.001 |
|      27 |      3.97182 |    4.89674 |               0.254223 |             0.244338 |   0.924913 |             0.009884 |          0.235089 |           0.001 |
|      28 |      3.93277 |    4.87621 |               0.262239 |             0.24553  |   0.943442 |             0.016708 |          0.236096 |           0.001 |
|      29 |      3.89067 |    4.85761 |               0.266819 |             0.247914 |   0.966937 |             0.018905 |          0.238245 |           0.001 |
|      30 |      3.83629 |    4.85204 |               0.270827 |             0.246722 |   1.01575  |             0.024105 |          0.236565 |           0.001 |
|      31 |      3.80354 |    4.83137 |               0.273404 |             0.25149  |   1.02783  |             0.021914 |          0.241212 |           0.001 |
|      32 |      3.76794 |    4.8283  |               0.275981 |             0.247914 |   1.06036  |             0.028066 |          0.237311 |           0.001 |
|      33 |      3.73012 |    4.80724 |               0.277412 |             0.257449 |   1.07712  |             0.019963 |          0.246678 |           0.001 |
|      34 |      3.68633 |    4.79905 |               0.288577 |             0.253874 |   1.11272  |             0.034703 |          0.242746 |           0.001 |
|      35 |      3.6533  |    4.78687 |               0.289722 |             0.262217 |   1.13357  |             0.027505 |          0.250881 |           0.001 |
|      36 |      3.60939 |    4.77912 |               0.294303 |             0.255066 |   1.16973  |             0.039237 |          0.243368 |           0.001 |
|      37 |      3.57968 |    4.76309 |               0.295734 |             0.261025 |   1.18342  |             0.034709 |          0.249191 |           0.001 |
|      38 |      3.55798 |    4.76307 |               0.304037 |             0.257449 |   1.20509  |             0.046587 |          0.245398 |           0.001 |
|      39 |      3.52012 |    4.7477  |               0.305182 |             0.264601 |   1.22758  |             0.040581 |          0.252325 |           0.001 |

|   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training-Validation Gap |   Validation Perplexity |   Training Token Accuracy at Best Epoch |   Validation Token Accuracy at Best Epoch |   Token Accuracy Gap |   Selection Score |
|-------------:|------------------------------:|-----------------------:|--------------------------:|------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|------------------:|
|           39 |                       3.52012 |                 4.7477 |                   1.22758 |                 115.319 |                                0.305182 |                                  0.264601 |             0.040581 |          0.252325 |

| Model                                    |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 1 English-to-French Baseline GRU |           39 |                       3.52012 |                 4.7477 |                  0.305182 |                    0.264601 |             0.040581 |                           0.252325 |                      0 |         0.031716 |

| English Input                               | Target French                                     | Predicted French                             |   Exact Match |   BLEU-4 |
|:--------------------------------------------|:--------------------------------------------------|:---------------------------------------------|--------------:|---------:|
| we visited the national park                | nous avons visité le parc national                | nous avons visité le parc                    |             0 | 0.818731 |
| she lost her keys                           | elle a perdu ses clés                             | elle a une grande                            |             0 | 0.132322 |
| they built a thick concrete wall for safety | ils ont construit un épais mur en béton pour l... | ils ont construit une nouvelle de la musique |             0 | 0.101528 |
| he finished his book yesterday              | il a fini son livre hier                          | il a une grande maison                       |             0 | 0.093026 |
| the cat is sleeping                         | le chat dort                                      | le pain est très                             |             0 | 0.080343 |

## Problem 1: Plots

```python
plot_configured_experiment_outputs(
    experiment=experiment_results["p1_baseline"],
    title="Problem 1: English-to-French Baseline GRU Training vs Validation Loss",
    show_best_epoch=False
)
```

**Output**

![Output](README_files/output_14_1.png)

```text
Problem 1 Final Validation Metrics
```

| Model                                    |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 1 English-to-French Baseline GRU |           39 |                       3.52012 |                 4.7477 |                  0.305182 |                    0.264601 |             0.040581 |                           0.252325 |                      0 |         0.031716 |

```text
Problem 1 Qualitative Validation Examples
```

| English Input                               | Target French                                     | Predicted French                             |   Exact Match |   BLEU-4 |
|:--------------------------------------------|:--------------------------------------------------|:---------------------------------------------|--------------:|---------:|
| we visited the national park                | nous avons visité le parc national                | nous avons visité le parc                    |             0 | 0.818731 |
| she lost her keys                           | elle a perdu ses clés                             | elle a une grande                            |             0 | 0.132322 |
| they built a thick concrete wall for safety | ils ont construit un épais mur en béton pour l... | ils ont construit une nouvelle de la musique |             0 | 0.101528 |
| he finished his book yesterday              | il a fini son livre hier                          | il a une grande maison                       |             0 | 0.093026 |
| the cat is sleeping                         | le chat dort                                      | le pain est très                             |             0 | 0.080343 |

## Problem 1: Metrics

```python
p1_all_results = display_single_experiment_all_results(
    experiment=experiment_results["p1_baseline"],
    total_pairs=len(pairs),
    common_config=COMMON_CONFIG
)
```

**Output**

```text
================================================================================
PROBLEM 1 ENGLISH-TO-FRENCH BASELINE GRU — ALL METRICS AND RESULTS
================================================================================

1. Dataset and Split Table
```

| Metric                    | Value   |
|:--------------------------|:--------|
| Total Sentence Pairs      | 555     |
| Training Sentence Pairs   | 444     |
| Validation Sentence Pairs | 111     |
| Training Percent          | 80%     |
| Validation Percent        | 20%     |
| Max Sequence Length       | 25      |
| Random Seed               | 42      |

```text
2. Vocabulary Table
```

| Vocabulary                |   Size |
|:--------------------------|-------:|
| English Source Vocabulary |    894 |
| French Target Vocabulary  |    994 |

```text
3. Out-of-Vocabulary Diagnostic Table
```

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique English training source tokens             | 890      |
| Unique French training target tokens              | 990      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV %                          |  34.2618 |
| Validation French OOV %                           |  38.0577 |

```text
4. Hyperparameter and Design Choice Table
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | Baseline GRU Encoder-Decoder with Fixed Context   |
| Source Language                     | English                                           |
| Target Language                     | French                                            |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
5. Model Size Table
```

| Metric                 |   Value |
|:-----------------------|--------:|
| Trainable Parameters   |  298786 |
| Source Vocabulary Size |     894 |
| Target Vocabulary Size |     994 |
| Embedding Size         |      48 |
| Hidden Size            |      96 |
| GRU Layers             |       1 |

```text
6. Full Training History
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.73301 |    6.36895 |               0.080447 |             0.1323   |  -0.364059 |            -0.051854 |          0.1323   |           0.001 |
|       2 |      5.99415 |    5.97351 |               0.127111 |             0.1323   |  -0.020637 |            -0.005189 |          0.1323   |           0.001 |
|       3 |      5.60038 |    5.85112 |               0.140853 |             0.151371 |   0.250741 |            -0.010518 |          0.148863 |           0.001 |
|       4 |      5.45585 |    5.74174 |               0.146293 |             0.151371 |   0.285894 |            -0.005078 |          0.148512 |           0.001 |
|       5 |      5.34588 |    5.63758 |               0.146006 |             0.151371 |   0.291707 |            -0.005364 |          0.148454 |           0.001 |
|       6 |      5.25421 |    5.57303 |               0.146865 |             0.148987 |   0.318821 |            -0.002122 |          0.145799 |           0.001 |
|       7 |      5.17459 |    5.5156  |               0.15345  |             0.156138 |   0.341007 |            -0.002689 |          0.152728 |           0.001 |
|       8 |      5.10013 |    5.4716  |               0.162325 |             0.16329  |   0.371472 |            -0.000965 |          0.159575 |           0.001 |
|       9 |      5.0303  |    5.44468 |               0.166333 |             0.172825 |   0.414376 |            -0.006492 |          0.168681 |           0.001 |
|      10 |      4.95956 |    5.37904 |               0.17807  |             0.187128 |   0.41948  |            -0.009057 |          0.182933 |           0.001 |
|      11 |      4.87969 |    5.33702 |               0.181506 |             0.187128 |   0.457336 |            -0.005622 |          0.182554 |           0.001 |
|      12 |      4.80687 |    5.27877 |               0.184941 |             0.188319 |   0.471901 |            -0.003378 |          0.1836   |           0.001 |
|      13 |      4.739   |    5.23428 |               0.192957 |             0.191895 |   0.495288 |             0.001062 |          0.186942 |           0.001 |
|      14 |      4.66742 |    5.19854 |               0.198969 |             0.206198 |   0.531122 |            -0.007228 |          0.200887 |           0.001 |
|      15 |      4.60792 |    5.16306 |               0.202977 |             0.203814 |   0.555134 |            -0.000837 |          0.198263 |           0.001 |
|      16 |      4.53916 |    5.12035 |               0.207558 |             0.213349 |   0.581189 |            -0.005791 |          0.207537 |           0.001 |
|      17 |      4.47709 |    5.09399 |               0.215001 |             0.215733 |   0.616899 |            -0.000732 |          0.209564 |           0.001 |
|      18 |      4.41045 |    5.07696 |               0.21586  |             0.215733 |   0.666507 |             0.000127 |          0.209068 |           0.001 |
|      19 |      4.35737 |    5.04429 |               0.217292 |             0.214541 |   0.686925 |             0.002751 |          0.207672 |           0.001 |
|      20 |      4.30576 |    5.03636 |               0.223876 |             0.218117 |   0.7306   |             0.00576  |          0.210811 |           0.001 |
|      21 |      4.2649  |    5.02712 |               0.224449 |             0.220501 |   0.762219 |             0.003948 |          0.212878 |           0.001 |
|      22 |      4.22112 |    4.98222 |               0.232751 |             0.227652 |   0.761099 |             0.005099 |          0.220041 |           0.001 |
|      23 |      4.1517  |    4.95432 |               0.241626 |             0.231228 |   0.802616 |             0.010398 |          0.223201 |           0.001 |
|      24 |      4.1018  |    4.94648 |               0.243916 |             0.233611 |   0.84468  |             0.010305 |          0.225165 |           0.001 |
|      25 |      4.06109 |    4.91691 |               0.249642 |             0.240763 |   0.855812 |             0.008879 |          0.232205 |           0.001 |
|      26 |      4.02289 |    4.91353 |               0.249356 |             0.239571 |   0.890639 |             0.009785 |          0.230665 |           0.001 |
|      27 |      3.97182 |    4.89674 |               0.254223 |             0.244338 |   0.924913 |             0.009884 |          0.235089 |           0.001 |
|      28 |      3.93277 |    4.87621 |               0.262239 |             0.24553  |   0.943442 |             0.016708 |          0.236096 |           0.001 |
|      29 |      3.89067 |    4.85761 |               0.266819 |             0.247914 |   0.966937 |             0.018905 |          0.238245 |           0.001 |
|      30 |      3.83629 |    4.85204 |               0.270827 |             0.246722 |   1.01575  |             0.024105 |          0.236565 |           0.001 |
|      31 |      3.80354 |    4.83137 |               0.273404 |             0.25149  |   1.02783  |             0.021914 |          0.241212 |           0.001 |
|      32 |      3.76794 |    4.8283  |               0.275981 |             0.247914 |   1.06036  |             0.028066 |          0.237311 |           0.001 |
|      33 |      3.73012 |    4.80724 |               0.277412 |             0.257449 |   1.07712  |             0.019963 |          0.246678 |           0.001 |
|      34 |      3.68633 |    4.79905 |               0.288577 |             0.253874 |   1.11272  |             0.034703 |          0.242746 |           0.001 |
|      35 |      3.6533  |    4.78687 |               0.289722 |             0.262217 |   1.13357  |             0.027505 |          0.250881 |           0.001 |
|      36 |      3.60939 |    4.77912 |               0.294303 |             0.255066 |   1.16973  |             0.039237 |          0.243368 |           0.001 |
|      37 |      3.57968 |    4.76309 |               0.295734 |             0.261025 |   1.18342  |             0.034709 |          0.249191 |           0.001 |
|      38 |      3.55798 |    4.76307 |               0.304037 |             0.257449 |   1.20509  |             0.046587 |          0.245398 |           0.001 |
|      39 |      3.52012 |    4.7477  |               0.305182 |             0.264601 |   1.22758  |             0.040581 |          0.252325 |           0.001 |

```text
7. Best Epoch and Final Epoch Table
```

| Metric                                  |      Value |
|:----------------------------------------|-----------:|
| Epochs Completed                        |  39        |
| Best Selected Epoch                     |  39        |
| Training Loss at Best Epoch             |   3.52012  |
| Validation Loss at Best Epoch           |   4.7477   |
| Training-Validation Gap at Best Epoch   |   1.22758  |
| Validation Perplexity at Best Epoch     | 115.319    |
| Training Token Accuracy at Best Epoch   |   0.305182 |
| Validation Token Accuracy at Best Epoch |   0.264601 |
| Token Accuracy Gap at Best Epoch        |   0.040581 |
| Validation-First Selection Score        |   0.252325 |
| Final Epoch                             |  39        |
| Final Training Loss                     |   3.52012  |
| Final Validation Loss                   |   4.7477   |
| Final Training-Validation Gap           |   1.22758  |
| Final Validation Perplexity             | 115.319    |
| Final Training Token Accuracy           |   0.305182 |
| Final Validation Token Accuracy         |   0.264601 |
| Final Token Accuracy Gap                |   0.040581 |
| Final Selection Score                   |   0.252325 |

```text
8. Final Translation Metrics
```

| Model                                    |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 1 English-to-French Baseline GRU |           39 |                       3.52012 |                 4.7477 |                  0.305182 |                    0.264601 |             0.040581 |                           0.252325 |                      0 |         0.031716 |

```text
9. Exact Match Count Table
```

| Metric                |   Value |
|:----------------------|--------:|
| Validation Examples   |     111 |
| Exact Match Count     |       0 |
| Non-Exact Match Count |     111 |
| Exact Match Accuracy  |       0 |

```text
10. BLEU-4 Distribution Table
```

| Metric                    |    Value |
|:--------------------------|---------:|
| Average BLEU-4            | 0.031716 |
| Median BLEU-4             | 0.027776 |
| Minimum BLEU-4            | 0        |
| Maximum BLEU-4            | 0.818731 |
| Standard Deviation BLEU-4 | 0.080061 |
| BLEU-4 >= 0.25 Count      | 1        |
| BLEU-4 >= 0.50 Count      | 1        |
| BLEU-4 >= 0.75 Count      | 1        |

```text
11. Top 10 Validation Examples by BLEU-4
```

| English Input                               | Target French                                     | Predicted French                             |   Exact Match |   BLEU-4 |
|:--------------------------------------------|:--------------------------------------------------|:---------------------------------------------|--------------:|---------:|
| we visited the national park                | nous avons visité le parc national                | nous avons visité le parc                    |             0 | 0.818731 |
| she lost her keys                           | elle a perdu ses clés                             | elle a une grande                            |             0 | 0.132322 |
| they built a thick concrete wall for safety | ils ont construit un épais mur en béton pour l... | ils ont construit une nouvelle de la musique |             0 | 0.101528 |
| he finished his book yesterday              | il a fini son livre hier                          | il a une grande maison                       |             0 | 0.093026 |
| the cat is sleeping                         | le chat dort                                      | le pain est très                             |             0 | 0.080343 |
| the water is cold                           | l'eau est froide                                  | le pain est très                             |             0 | 0.080343 |
| the cat meows loudly                        | le chat miaule bruyamment                         | le pain est très                             |             0 | 0.080343 |
| she visits her grandparents                 | elle visite ses grands-parents                    | elle porte un roman                          |             0 | 0.080343 |
| he finished his test early                  | il a fini son test en avance                      | il a une grande maison                       |             0 | 0.076163 |
| where is the station ?                      | où est la gare ?                                  | le pain est très                             |             0 | 0.062571 |

```text
12. Lowest 10 Validation Examples by BLEU-4
```

| English Input                                     | Target French                                     | Predicted French                             |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:---------------------------------------------|--------------:|---------:|
| she won a tennis match                            | elle a gagné un match de tennis                   | ils ont construit une nouvelle               |             0 |        0 |
| we watch a movie together                         | nous regardons un film ensemble                   | ils ont construit une grande                 |             0 |        0 |
| she wears a beautiful silver bracelet on her w... | elle porte un beau bracelet en argent au poignet  | ils ont construit une nouvelle de la musique |             0 |        0 |
| he answers all the questions                      | il répond à toutes les questions                  | nous avons visité le parc                    |             0 |        0 |
| he drives a black sedan                           | il conduit une berline noire                      | ils ont construit un groupe                  |             0 |        0 |
| she works full-time at the general hospital       | elle travaille à plein temps à l'hôpital général  | ils ont construit une nouvelle de la maison  |             0 |        0 |
| how are you ?                                     | comment ça va ?                                   | nous avons visité le parc                    |             0 |        0 |
| he enjoys reading books                           | il aime lire des livres                           | nous avons visité le parc                    |             0 |        0 |
| the library is a quiet place                      | la bibliothèque est un endroit calme              | nous avons visité le parc                    |             0 |        0 |
| the pure water in the mountain stream is refre... | l'eau pure du ruisseau de montagne est rafraîc... | nous avons traversé la chimie dans le parc   |             0 |        0 |

```text
13. Exact Match Examples
No exact match examples were found for Problem 1.

14. Full Validation Translation Results
```

| Unnamed: 0   | English Input                                     | Target French                                     | Predicted French                             | Exact Match   | BLEU-4   |
|:-------------|:--------------------------------------------------|:--------------------------------------------------|:---------------------------------------------|:--------------|:---------|
| 0            | she won a tennis match                            | elle a gagné un match de tennis                   | ils ont construit une nouvelle               | 0             | 0.000000 |
| 1            | the organic market opens at dawn on saturdays     | le marché biologique ouvre à l'aube le samedi     | nous avons traversé la musique dans le parc  | 0             | 0.027776 |
| 2            | we watch a movie together                         | nous regardons un film ensemble                   | ils ont construit une grande                 | 0             | 0.000000 |
| 3            | the bread at this bakery is always crunchy        | le pain de cette boulangerie est toujours crou... | nous avons visité le parc dans le parc       | 0             | 0.027776 |
| 4            | we dance at the wedding                           | nous dansons au mariage                           | nous avons visité le parc                    | 0             | 0.053728 |
| ...          | ...                                               | ...                                               | ...                                          | ...           | ...      |
| 106          | they play soccer every weekend                    | ils jouent au football chaque week-end            | ils ont construit une grande                 | 0             | 0.043989 |
| 107          | i want a hot cup of coffee                        | je veux une tasse de café chaud                   | ils ont construit une nouvelle de la maison  | 0             | 0.033032 |
| 108          | the book is on the table                          | le livre est sur la table                         | nous avons visité le parc                    | 0             | 0.043989 |
| 109          | he finished his science project ahead of schedule | il a terminé son projet scientifique en avance... | ils ont construit une nouvelle de la musique | 0             | 0.000000 |
| 110          | we need to buy a dozen fresh eggs                 | nous devons acheter une douzaine d' ufs frais     | ils ont construit une nouvelle de la musique | 0             | 0.027776 |

```text
15. Compact Final Table for Report Table
```

| Problem   | Direction         | Architecture                                    |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:----------|:------------------|:------------------------------------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 1 | English-to-French | Baseline GRU Encoder-Decoder with Fixed Context |           39 |                 4.7477 |                       3.52012 |                   1.22758 |                    0.264601 |                  0.305182 |             0.040581 |                           0.252325 |                      0 |         0.031716 |                 298786 |

## Problem 2: English-to-French GRU with Attention

```python
experiment_results["p2_attention"] = run_configured_experiment(
    config=EXPERIMENT_CONFIGS["p2_attention"],
    resources_by_direction=translation_resources,
    common_config=COMMON_CONFIG,
    device=DEVICE
)

display(experiment_results["p2_attention"]["history_df"])
display(experiment_results["p2_attention"]["best_epoch_df"])
display(experiment_results["p2_attention"]["metrics_df"])
display(experiment_results["p2_attention"]["sample_translations_df"])
```

**Output**

```text
Problem 2 English-to-French GRU with Attention setup
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | GRU Encoder-Decoder with Attention                |
| Attention Type                      | Bahdanau Additive Attention                       |
| Source Language                     | English                                           |
| Target Language                     | French                                            |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
Seq2SeqGRUAttention(
  (encoder): EncoderGRU(
    (embedding): Embedding(894, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(48, 96, batch_first=True)
  )
  (decoder): DecoderGRUAttention(
    (embedding): Embedding(994, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (attention): BahdanauAttention(
      (W_query): Linear(in_features=96, out_features=96, bias=True)
      (W_keys): Linear(in_features=96, out_features=96, bias=True)
      (v): Linear(in_features=96, out_features=1, bias=True)
    )
    (gru): GRU(144, 96, batch_first=True)
    (fc_out): Linear(in_features=240, out_features=994, bias=True)
  )
)
Trainable parameters: 460643
```

```text
Epoch 01/50 | Train Loss: 6.7713 | Val Loss: 6.5544 | Train Tok Acc: 0.0286 | Val Tok Acc: 0.1061 | Loss Gap: -0.2168 | Score: 0.1061 | LR: 0.001000
  New best validation-first score: 0.1061
```

```text
Epoch 02/50 | Train Loss: 5.8954 | Val Loss: 5.8410 | Train Tok Acc: 0.1388 | Val Tok Acc: 0.1490 | Loss Gap: -0.0544 | Score: 0.1490 | LR: 0.001000
  New best validation-first score: 0.1490
```

```text
Epoch 03/50 | Train Loss: 5.3105 | Val Loss: 5.6412 | Train Tok Acc: 0.1560 | Val Tok Acc: 0.1633 | Loss Gap: 0.3307 | Score: 0.1600 | LR: 0.001000
  New best validation-first score: 0.1600
```

```text
Epoch 04/50 | Train Loss: 5.0722 | Val Loss: 5.4921 | Train Tok Acc: 0.1597 | Val Tok Acc: 0.1740 | Loss Gap: 0.4199 | Score: 0.1698 | LR: 0.001000
  New best validation-first score: 0.1698
```

```text
Epoch 05/50 | Train Loss: 4.8950 | Val Loss: 5.3722 | Train Tok Acc: 0.1815 | Val Tok Acc: 0.2002 | Loss Gap: 0.4771 | Score: 0.1955 | LR: 0.001000
  New best validation-first score: 0.1955
```

```text
Epoch 06/50 | Train Loss: 4.7195 | Val Loss: 5.2858 | Train Tok Acc: 0.1970 | Val Tok Acc: 0.2229 | Loss Gap: 0.5663 | Score: 0.2172 | LR: 0.001000
  New best validation-first score: 0.2172
```

```text
Epoch 07/50 | Train Loss: 4.5604 | Val Loss: 5.2102 | Train Tok Acc: 0.2216 | Val Tok Acc: 0.2408 | Loss Gap: 0.6498 | Score: 0.2343 | LR: 0.001000
  New best validation-first score: 0.2343
```

```text
Epoch 08/50 | Train Loss: 4.3992 | Val Loss: 5.1119 | Train Tok Acc: 0.2316 | Val Tok Acc: 0.2574 | Loss Gap: 0.7127 | Score: 0.2503 | LR: 0.001000
  New best validation-first score: 0.2503
```

```text
Epoch 09/50 | Train Loss: 4.2459 | Val Loss: 5.0236 | Train Tok Acc: 0.2554 | Val Tok Acc: 0.2789 | Loss Gap: 0.7777 | Score: 0.2711 | LR: 0.001000
  New best validation-first score: 0.2711
```

```text
Epoch 10/50 | Train Loss: 4.0873 | Val Loss: 4.9686 | Train Tok Acc: 0.2740 | Val Tok Acc: 0.2908 | Loss Gap: 0.8813 | Score: 0.2820 | LR: 0.001000
  New best validation-first score: 0.2820
```

```text
Epoch 11/50 | Train Loss: 3.9471 | Val Loss: 4.9113 | Train Tok Acc: 0.2851 | Val Tok Acc: 0.2920 | Loss Gap: 0.9642 | Score: 0.2824 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 12/50 | Train Loss: 3.8017 | Val Loss: 4.8431 | Train Tok Acc: 0.3046 | Val Tok Acc: 0.2956 | Loss Gap: 1.0414 | Score: 0.2852 | LR: 0.001000
  New best validation-first score: 0.2852
```

```text
Epoch 13/50 | Train Loss: 3.6835 | Val Loss: 4.7872 | Train Tok Acc: 0.3149 | Val Tok Acc: 0.2968 | Loss Gap: 1.1037 | Score: 0.2857 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 14/50 | Train Loss: 3.5398 | Val Loss: 4.7476 | Train Tok Acc: 0.3398 | Val Tok Acc: 0.3051 | Loss Gap: 1.2078 | Score: 0.2930 | LR: 0.001000
  New best validation-first score: 0.2930
```

```text
Epoch 15/50 | Train Loss: 3.4443 | Val Loss: 4.7208 | Train Tok Acc: 0.3392 | Val Tok Acc: 0.3135 | Loss Gap: 1.2765 | Score: 0.3007 | LR: 0.001000
  New best validation-first score: 0.3007
```

```text
Epoch 16/50 | Train Loss: 3.3417 | Val Loss: 4.6836 | Train Tok Acc: 0.3527 | Val Tok Acc: 0.3182 | Loss Gap: 1.3419 | Score: 0.3048 | LR: 0.001000
  New best validation-first score: 0.3048
```

```text
Epoch 17/50 | Train Loss: 3.2442 | Val Loss: 4.6475 | Train Tok Acc: 0.3685 | Val Tok Acc: 0.3254 | Loss Gap: 1.4033 | Score: 0.3114 | LR: 0.001000
  New best validation-first score: 0.3114
```

```text
Epoch 18/50 | Train Loss: 3.1658 | Val Loss: 4.6311 | Train Tok Acc: 0.3710 | Val Tok Acc: 0.3266 | Loss Gap: 1.4653 | Score: 0.3119 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 19/50 | Train Loss: 3.0722 | Val Loss: 4.5933 | Train Tok Acc: 0.3919 | Val Tok Acc: 0.3278 | Loss Gap: 1.5212 | Score: 0.3126 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 20/50 | Train Loss: 2.9981 | Val Loss: 4.5919 | Train Tok Acc: 0.4008 | Val Tok Acc: 0.3278 | Loss Gap: 1.5938 | Score: 0.3118 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 21/50 | Train Loss: 2.9096 | Val Loss: 4.5768 | Train Tok Acc: 0.4143 | Val Tok Acc: 0.3409 | Loss Gap: 1.6672 | Score: 0.3242 | LR: 0.001000
  New best validation-first score: 0.3242
```

```text
Epoch 22/50 | Train Loss: 2.8545 | Val Loss: 4.5557 | Train Tok Acc: 0.4246 | Val Tok Acc: 0.3468 | Loss Gap: 1.7012 | Score: 0.3298 | LR: 0.001000
  New best validation-first score: 0.3298
```

```text
Epoch 23/50 | Train Loss: 2.7778 | Val Loss: 4.5490 | Train Tok Acc: 0.4363 | Val Tok Acc: 0.3480 | Loss Gap: 1.7712 | Score: 0.3303 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 24/50 | Train Loss: 2.7168 | Val Loss: 4.5226 | Train Tok Acc: 0.4460 | Val Tok Acc: 0.3623 | Loss Gap: 1.8058 | Score: 0.3443 | LR: 0.001000
  New best validation-first score: 0.3443
```

```text
Epoch 25/50 | Train Loss: 2.6551 | Val Loss: 4.5196 | Train Tok Acc: 0.4621 | Val Tok Acc: 0.3516 | Loss Gap: 1.8645 | Score: 0.3330 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 26/50 | Train Loss: 2.6037 | Val Loss: 4.5181 | Train Tok Acc: 0.4738 | Val Tok Acc: 0.3611 | Loss Gap: 1.9144 | Score: 0.3420 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 27/50 | Train Loss: 2.5509 | Val Loss: 4.5031 | Train Tok Acc: 0.4833 | Val Tok Acc: 0.3588 | Loss Gap: 1.9522 | Score: 0.3392 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 28/50 | Train Loss: 2.5209 | Val Loss: 4.5005 | Train Tok Acc: 0.4901 | Val Tok Acc: 0.3564 | Loss Gap: 1.9796 | Score: 0.3366 | LR: 0.001000
  No useful validation improvement. Patience: 4/4
Early stopping triggered.
Loaded best model from epoch 24 with validation-first score: 0.3443
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.77127 |    6.55443 |               0.028629 |             0.106079 |  -0.216845 |            -0.07745  |          0.106079 |           0.001 |
|       2 |      5.89542 |    5.84098 |               0.138849 |             0.148987 |  -0.054436 |            -0.010138 |          0.148987 |           0.001 |
|       3 |      5.31051 |    5.6412  |               0.156026 |             0.16329  |   0.330698 |            -0.007263 |          0.159983 |           0.001 |
|       4 |      5.07225 |    5.49213 |               0.159748 |             0.174017 |   0.41988  |            -0.014269 |          0.169818 |           0.001 |
|       5 |      4.89502 |    5.37216 |               0.181506 |             0.200238 |   0.477137 |            -0.018733 |          0.195467 |           0.001 |
|       6 |      4.71948 |    5.28577 |               0.196965 |             0.222884 |   0.566288 |            -0.025919 |          0.217222 |           0.001 |
|       7 |      4.56038 |    5.21022 |               0.221586 |             0.240763 |   0.64984  |            -0.019177 |          0.234264 |           0.001 |
|       8 |      4.39923 |    5.11193 |               0.231606 |             0.257449 |   0.712699 |            -0.025843 |          0.250322 |           0.001 |
|       9 |      4.24587 |    5.0236  |               0.255368 |             0.278903 |   0.777721 |            -0.023536 |          0.271126 |           0.001 |
|      10 |      4.08732 |    4.96859 |               0.273977 |             0.290822 |   0.881265 |            -0.016846 |          0.28201  |           0.001 |
|      11 |      3.9471  |    4.91128 |               0.285142 |             0.292014 |   0.964179 |            -0.006873 |          0.282373 |           0.001 |
|      12 |      3.80173 |    4.84309 |               0.304609 |             0.29559  |   1.04136  |             0.009019 |          0.285176 |           0.001 |
|      13 |      3.6835  |    4.78723 |               0.314916 |             0.296782 |   1.10373  |             0.018134 |          0.285745 |           0.001 |
|      14 |      3.5398  |    4.74758 |               0.339823 |             0.305125 |   1.20778  |             0.034697 |          0.293047 |           0.001 |
|      15 |      3.44428 |    4.72082 |               0.33925  |             0.313468 |   1.27655  |             0.025782 |          0.300703 |           0.001 |
|      16 |      3.34171 |    4.6836  |               0.352705 |             0.318236 |   1.34189  |             0.034469 |          0.304817 |           0.001 |
|      17 |      3.24422 |    4.64749 |               0.368451 |             0.325387 |   1.40327  |             0.043064 |          0.311355 |           0.001 |
|      18 |      3.1658  |    4.63113 |               0.371028 |             0.326579 |   1.46534  |             0.044449 |          0.311926 |           0.001 |
|      19 |      3.07216 |    4.59331 |               0.391927 |             0.327771 |   1.52115  |             0.064156 |          0.31256  |           0.001 |
|      20 |      2.99806 |    4.59186 |               0.400802 |             0.327771 |   1.5938   |             0.07303  |          0.311833 |           0.001 |
|      21 |      2.90957 |    4.57675 |               0.414257 |             0.340882 |   1.66718  |             0.073375 |          0.32421  |           0.001 |
|      22 |      2.85454 |    4.55571 |               0.424563 |             0.346841 |   1.70117  |             0.077722 |          0.32983  |           0.001 |
|      23 |      2.77776 |    4.54901 |               0.436301 |             0.348033 |   1.77125  |             0.088268 |          0.330321 |           0.001 |
|      24 |      2.71684 |    4.52264 |               0.446035 |             0.362336 |   1.80579  |             0.083699 |          0.344278 |           0.001 |
|      25 |      2.65514 |    4.51962 |               0.462067 |             0.351609 |   1.86448  |             0.110458 |          0.332964 |           0.001 |
|      26 |      2.60367 |    4.51808 |               0.473805 |             0.361144 |   1.91441  |             0.112661 |          0.342    |           0.001 |
|      27 |      2.55087 |    4.50309 |               0.483252 |             0.35876  |   1.95222  |             0.124492 |          0.339238 |           0.001 |
|      28 |      2.52093 |    4.50052 |               0.490123 |             0.356377 |   1.97959  |             0.133746 |          0.336581 |           0.001 |

|   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training-Validation Gap |   Validation Perplexity |   Training Token Accuracy at Best Epoch |   Validation Token Accuracy at Best Epoch |   Token Accuracy Gap |   Selection Score |
|-------------:|------------------------------:|-----------------------:|--------------------------:|------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|------------------:|
|           24 |                       2.71684 |                4.52264 |                   1.80579 |                  92.078 |                                0.446035 |                                  0.362336 |             0.083699 |          0.344278 |

| Model                                          |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 2 English-to-French GRU with Attention |           24 |                       2.71684 |                4.52264 |                  0.446035 |                    0.362336 |             0.083699 |                           0.344278 |                      0 |          0.07046 |

| English Input                                     | Target French                                   | Predicted French                     |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:------------------------------------------------|:-------------------------------------|--------------:|---------:|
| we visited the national park                      | nous avons visité le parc national              | nous avons visité le parc            |             0 | 0.818731 |
| she bought a beautiful painting                   | elle a acheté un beau tableau                   | elle a acheté un tournoi             |             0 | 0.547518 |
| we are planning a trip to rome                    | nous planifions un voyage à rome                | nous planifions un voyage de mariage |             0 | 0.508133 |
| we are planning a trip to london                  | nous planifions un voyage à londres             | nous planifions un voyage de mariage |             0 | 0.508133 |
| he speaks six international languages complete... | il parle couramment six langues internationales | il parle couramment cinq langues     |             0 | 0.233947 |

## Problem 2: Plots and Comparison

```python
plot_configured_experiment_outputs(
    experiment=experiment_results["p2_attention"],
    title="Problem 2: English-to-French GRU with Attention Training vs Validation Loss",
    show_best_epoch=True
)

p1_vs_p2_comparison = display_configured_comparison(
    experiment_results=experiment_results,
    experiment_keys=["p1_baseline", "p2_attention"],
    title="Problem 1 vs Problem 2 Comparison"
)

p2_improvement_table = display_configured_improvement(
    baseline_experiment=experiment_results["p1_baseline"],
    candidate_experiment=experiment_results["p2_attention"],
    title="Problem 2 Direct Improvement Table over Problem 1"
)

display_configured_attention_examples(experiment_results["p2_attention"], n=2)
```

**Output**

![Output](README_files/output_20_2.png)

```text
Best selected epoch: 24
Validation loss at selected epoch: 4.5226
Problem 2 Final Validation Metrics
```

| Model                                          |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 2 English-to-French GRU with Attention |           24 |                       2.71684 |                4.52264 |                  0.446035 |                    0.362336 |             0.083699 |                           0.344278 |                      0 |          0.07046 |

```text
Problem 2 Qualitative Validation Examples
```

| English Input                                     | Target French                                   | Predicted French                     |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:------------------------------------------------|:-------------------------------------|--------------:|---------:|
| we visited the national park                      | nous avons visité le parc national              | nous avons visité le parc            |             0 | 0.818731 |
| she bought a beautiful painting                   | elle a acheté un beau tableau                   | elle a acheté un tournoi             |             0 | 0.547518 |
| we are planning a trip to rome                    | nous planifions un voyage à rome                | nous planifions un voyage de mariage |             0 | 0.508133 |
| we are planning a trip to london                  | nous planifions un voyage à londres             | nous planifions un voyage de mariage |             0 | 0.508133 |
| he speaks six international languages complete... | il parle couramment six langues internationales | il parle couramment cinq langues     |             0 | 0.233947 |

```text
Problem 1 vs Problem 2 Comparison
```

| Problem   | Model                       | Direction         |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:----------|:----------------------------|:------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 1 | Baseline GRU                | English-to-French |           39 |                4.7477  |                       3.52012 |                   1.22758 |                    0.264601 |                  0.305182 |             0.040581 |                           0.252325 |                      0 |         0.031716 |                 298786 |
| Problem 2 | GRU with Bahdanau Attention | English-to-French |           24 |                4.52264 |                       2.71684 |                   1.80579 |                    0.362336 |                  0.446035 |             0.083699 |                           0.344278 |                      0 |         0.07046  |                 460643 |

```text
Problem 2 Direct Improvement Table over Problem 1
```

| Metric                       |   Problem 1 English-to-French Baseline GRU |   Problem 2 English-to-French GRU with Attention |   Problem 2 English-to-French GRU with Attention - Problem 1 English-to-French Baseline GRU | Improved?   |
|:-----------------------------|-------------------------------------------:|-------------------------------------------------:|--------------------------------------------------------------------------------------------:|:------------|
| Best Validation Loss         |                                   4.7477   |                                         4.52264  |                                                                                   -0.225063 | Yes         |
| Validation Token Accuracy    |                                   0.264601 |                                         0.362336 |                                                                                    0.097735 | Yes         |
| Training-Validation Loss Gap |                                   1.22758  |                                         1.80579  |                                                                                    0.578216 | No          |
| Exact Match Accuracy         |                                   0        |                                         0        |                                                                                    0        | No          |
| Average BLEU-4               |                                   0.031716 |                                         0.07046  |                                                                                    0.038744 | Yes         |

```text
Problem 2 Attention Map Example 1
English Input: we visited the national park
Target French: nous avons visité le parc national
Predicted French: nous avons visité le parc
Exact Match: 0
BLEU-4: 0.8187307530779819
```

![Output](README_files/output_20_3.png)

```text
Problem 2 Attention Map Example 2
English Input: she bought a beautiful painting
Target French: elle a acheté un beau tableau
Predicted French: elle a acheté un tournoi
Exact Match: 0
BLEU-4: 0.5475182535069453
```

![Output](README_files/output_20_4.png)

## Problem 2: Metrics

```python
p2_all_results = display_single_experiment_all_results(
    experiment=experiment_results["p2_attention"],
    total_pairs=len(pairs),
    common_config=COMMON_CONFIG
)

print("\n16. Problem 1 vs Problem 2 Compact Comparison")
display(p1_vs_p2_comparison)

print("\n17. Direct Improvement Table over Problem 1")
display(p2_improvement_table)
```

**Output**

```text
================================================================================
PROBLEM 2 ENGLISH-TO-FRENCH GRU WITH ATTENTION — ALL METRICS AND RESULTS
================================================================================

1. Dataset and Split Table
```

| Metric                    | Value   |
|:--------------------------|:--------|
| Total Sentence Pairs      | 555     |
| Training Sentence Pairs   | 444     |
| Validation Sentence Pairs | 111     |
| Training Percent          | 80%     |
| Validation Percent        | 20%     |
| Max Sequence Length       | 25      |
| Random Seed               | 42      |

```text
2. Vocabulary Table
```

| Vocabulary                |   Size |
|:--------------------------|-------:|
| English Source Vocabulary |    894 |
| French Target Vocabulary  |    994 |

```text
3. Out-of-Vocabulary Diagnostic Table
```

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique English training source tokens             | 890      |
| Unique French training target tokens              | 990      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV %                          |  34.2618 |
| Validation French OOV %                           |  38.0577 |

```text
4. Hyperparameter and Design Choice Table
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | GRU Encoder-Decoder with Attention                |
| Attention Type                      | Bahdanau Additive Attention                       |
| Source Language                     | English                                           |
| Target Language                     | French                                            |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
5. Model Size Table
```

| Metric                 |   Value |
|:-----------------------|--------:|
| Trainable Parameters   |  460643 |
| Source Vocabulary Size |     894 |
| Target Vocabulary Size |     994 |
| Embedding Size         |      48 |
| Hidden Size            |      96 |
| GRU Layers             |       1 |

```text
6. Full Training History
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.77127 |    6.55443 |               0.028629 |             0.106079 |  -0.216845 |            -0.07745  |          0.106079 |           0.001 |
|       2 |      5.89542 |    5.84098 |               0.138849 |             0.148987 |  -0.054436 |            -0.010138 |          0.148987 |           0.001 |
|       3 |      5.31051 |    5.6412  |               0.156026 |             0.16329  |   0.330698 |            -0.007263 |          0.159983 |           0.001 |
|       4 |      5.07225 |    5.49213 |               0.159748 |             0.174017 |   0.41988  |            -0.014269 |          0.169818 |           0.001 |
|       5 |      4.89502 |    5.37216 |               0.181506 |             0.200238 |   0.477137 |            -0.018733 |          0.195467 |           0.001 |
|       6 |      4.71948 |    5.28577 |               0.196965 |             0.222884 |   0.566288 |            -0.025919 |          0.217222 |           0.001 |
|       7 |      4.56038 |    5.21022 |               0.221586 |             0.240763 |   0.64984  |            -0.019177 |          0.234264 |           0.001 |
|       8 |      4.39923 |    5.11193 |               0.231606 |             0.257449 |   0.712699 |            -0.025843 |          0.250322 |           0.001 |
|       9 |      4.24587 |    5.0236  |               0.255368 |             0.278903 |   0.777721 |            -0.023536 |          0.271126 |           0.001 |
|      10 |      4.08732 |    4.96859 |               0.273977 |             0.290822 |   0.881265 |            -0.016846 |          0.28201  |           0.001 |
|      11 |      3.9471  |    4.91128 |               0.285142 |             0.292014 |   0.964179 |            -0.006873 |          0.282373 |           0.001 |
|      12 |      3.80173 |    4.84309 |               0.304609 |             0.29559  |   1.04136  |             0.009019 |          0.285176 |           0.001 |
|      13 |      3.6835  |    4.78723 |               0.314916 |             0.296782 |   1.10373  |             0.018134 |          0.285745 |           0.001 |
|      14 |      3.5398  |    4.74758 |               0.339823 |             0.305125 |   1.20778  |             0.034697 |          0.293047 |           0.001 |
|      15 |      3.44428 |    4.72082 |               0.33925  |             0.313468 |   1.27655  |             0.025782 |          0.300703 |           0.001 |
|      16 |      3.34171 |    4.6836  |               0.352705 |             0.318236 |   1.34189  |             0.034469 |          0.304817 |           0.001 |
|      17 |      3.24422 |    4.64749 |               0.368451 |             0.325387 |   1.40327  |             0.043064 |          0.311355 |           0.001 |
|      18 |      3.1658  |    4.63113 |               0.371028 |             0.326579 |   1.46534  |             0.044449 |          0.311926 |           0.001 |
|      19 |      3.07216 |    4.59331 |               0.391927 |             0.327771 |   1.52115  |             0.064156 |          0.31256  |           0.001 |
|      20 |      2.99806 |    4.59186 |               0.400802 |             0.327771 |   1.5938   |             0.07303  |          0.311833 |           0.001 |
|      21 |      2.90957 |    4.57675 |               0.414257 |             0.340882 |   1.66718  |             0.073375 |          0.32421  |           0.001 |
|      22 |      2.85454 |    4.55571 |               0.424563 |             0.346841 |   1.70117  |             0.077722 |          0.32983  |           0.001 |
|      23 |      2.77776 |    4.54901 |               0.436301 |             0.348033 |   1.77125  |             0.088268 |          0.330321 |           0.001 |
|      24 |      2.71684 |    4.52264 |               0.446035 |             0.362336 |   1.80579  |             0.083699 |          0.344278 |           0.001 |
|      25 |      2.65514 |    4.51962 |               0.462067 |             0.351609 |   1.86448  |             0.110458 |          0.332964 |           0.001 |
|      26 |      2.60367 |    4.51808 |               0.473805 |             0.361144 |   1.91441  |             0.112661 |          0.342    |           0.001 |
|      27 |      2.55087 |    4.50309 |               0.483252 |             0.35876  |   1.95222  |             0.124492 |          0.339238 |           0.001 |
|      28 |      2.52093 |    4.50052 |               0.490123 |             0.356377 |   1.97959  |             0.133746 |          0.336581 |           0.001 |

```text
7. Best Epoch and Final Epoch Table
```

| Metric                                  |     Value |
|:----------------------------------------|----------:|
| Epochs Completed                        | 28        |
| Best Selected Epoch                     | 24        |
| Training Loss at Best Epoch             |  2.71684  |
| Validation Loss at Best Epoch           |  4.52264  |
| Training-Validation Gap at Best Epoch   |  1.80579  |
| Validation Perplexity at Best Epoch     | 92.078    |
| Training Token Accuracy at Best Epoch   |  0.446035 |
| Validation Token Accuracy at Best Epoch |  0.362336 |
| Token Accuracy Gap at Best Epoch        |  0.083699 |
| Validation-First Selection Score        |  0.344278 |
| Final Epoch                             | 28        |
| Final Training Loss                     |  2.52093  |
| Final Validation Loss                   |  4.50052  |
| Final Training-Validation Gap           |  1.97959  |
| Final Validation Perplexity             | 90.0643   |
| Final Training Token Accuracy           |  0.490123 |
| Final Validation Token Accuracy         |  0.356377 |
| Final Token Accuracy Gap                |  0.133746 |
| Final Selection Score                   |  0.336581 |

```text
8. Final Translation Metrics
```

| Model                                          |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:-----------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 2 English-to-French GRU with Attention |           24 |                       2.71684 |                4.52264 |                  0.446035 |                    0.362336 |             0.083699 |                           0.344278 |                      0 |          0.07046 |

```text
9. Exact Match Count Table
```

| Metric                |   Value |
|:----------------------|--------:|
| Validation Examples   |     111 |
| Exact Match Count     |       0 |
| Non-Exact Match Count |     111 |
| Exact Match Accuracy  |       0 |

```text
10. BLEU-4 Distribution Table
```

| Metric                    |    Value |
|:--------------------------|---------:|
| Average BLEU-4            | 0.07046  |
| Median BLEU-4             | 0.041799 |
| Minimum BLEU-4            | 0        |
| Maximum BLEU-4            | 0.818731 |
| Standard Deviation BLEU-4 | 0.112829 |
| BLEU-4 >= 0.25 Count      | 4        |
| BLEU-4 >= 0.50 Count      | 4        |
| BLEU-4 >= 0.75 Count      | 1        |

```text
11. Top 10 Validation Examples by BLEU-4
```

| English Input                                     | Target French                                     | Predicted French                       |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:---------------------------------------|--------------:|---------:|
| we visited the national park                      | nous avons visité le parc national                | nous avons visité le parc              |             0 | 0.818731 |
| she bought a beautiful painting                   | elle a acheté un beau tableau                     | elle a acheté un tournoi               |             0 | 0.547518 |
| we are planning a trip to rome                    | nous planifions un voyage à rome                  | nous planifions un voyage de mariage   |             0 | 0.508133 |
| we are planning a trip to london                  | nous planifions un voyage à londres               | nous planifions un voyage de mariage   |             0 | 0.508133 |
| he speaks six international languages complete... | il parle couramment six langues internationales   | il parle couramment cinq langues       |             0 | 0.233947 |
| the cat climbs the tall tree                      | le chat grimpe sur le grand arbre                 | le koala grimpe sur le parc            |             0 | 0.183787 |
| i want a hot cup of coffee                        | je veux une tasse de café chaud                   | je veux une nouvelle paire             |             0 | 0.178248 |
| we visited the ancient amphitheater during our... | nous avons visité l'amphithéâtre antique penda... | nous avons visité le monument national |             0 | 0.144776 |
| the summer is very hot here                       | l'été est très chaud ici                          | le thé est très                        |             0 | 0.132322 |
| the winter is very cold here                      | l'hiver est très froid ici                        | le musée est très                      |             0 | 0.132322 |

```text
12. Lowest 10 Validation Examples by BLEU-4
```

| English Input                                     | Target French                                     | Predicted French                            |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:--------------------------------------------|--------------:|---------:|
| the pure water in the mountain stream is refre... | l'eau pure du ruisseau de montagne est rafraîc... | les oiseaux dans le parc                    |             0 |        0 |
| the beautiful spring flowers are finally blooming | les belles fleurs du printemps s'épanouissent ... | le koala grimpe sur le parc                 |             0 |        0 |
| how are you ?                                     | comment ça va ?                                   | ils parlent souvent                         |             0 |        0 |
| the new software update fixed several critical... | la nouvelle mise à jour logicielle a corrigé p... | le train arrive frais sent merveilleusement |             0 |        0 |
| the keys are missing from my coat pocket          | les clés manquent dans la poche de mon manteau    | le pain est trop sucré                      |             0 |        0 |
| the door has a wooden handle                      | la porte a une poignée en bois                    | le thé est un lieu                          |             0 |        0 |
| i love this song                                  | j'aime cette chanson                              | nous avons visité le parc                   |             0 |        0 |
| the project takes a lot of time                   | le projet prend beaucoup de temps                 | la soupe a une grande ville                 |             0 |        0 |
| i accidentally left my umbrella inside the taxi   | j'ai accidentellement laissé mon parapluie dan... | ils ont construit une grande ville          |             0 |        0 |
| the theater performance received a standing ov... | la performance théâtrale a reçu une ovation de... | le thé est un lieu                          |             0 |        0 |

```text
13. Exact Match Examples
No exact match examples were found for Problem 2.

14. Full Validation Translation Results
```

| Unnamed: 0   | English Input                                     | Target French                                     | Predicted French                        | Exact Match   | BLEU-4   |
|:-------------|:--------------------------------------------------|:--------------------------------------------------|:----------------------------------------|:--------------|:---------|
| 0            | she won a tennis match                            | elle a gagné un match de tennis                   | elle a acheté un tournoi                | 0             | 0.084288 |
| 1            | the organic market opens at dawn on saturdays     | le marché biologique ouvre à l'aube le samedi     | le pain est très                        | 0             | 0.029556 |
| 2            | we watch a movie together                         | nous regardons un film ensemble                   | nous avons célébré une grande maison    | 0             | 0.040825 |
| 3            | the bread at this bakery is always crunchy        | le pain de cette boulangerie est toujours crou... | le pain est très                        | 0             | 0.069172 |
| 4            | we dance at the wedding                           | nous dansons au mariage                           | nous avons visité le parc               | 0             | 0.053728 |
| ...          | ...                                               | ...                                               | ...                                     | ...           | ...      |
| 106          | they play soccer every weekend                    | ils jouent au football chaque week-end            | ils parlent souvent à                   | 0             | 0.048730 |
| 107          | i want a hot cup of coffee                        | je veux une tasse de café chaud                   | je veux une nouvelle paire              | 0             | 0.178248 |
| 108          | the book is on the table                          | le livre est sur la table                         | le pain est très                        | 0             | 0.057951 |
| 109          | he finished his science project ahead of schedule | il a terminé son projet scientifique en avance... | il a décidé de la musique de jazz       | 0             | 0.040371 |
| 110          | we need to buy a dozen fresh eggs                 | nous devons acheter une douzaine d' ufs frais     | nous avons célébré un voyage de mariage | 0             | 0.028634 |

```text
15. Compact Final Table for Report Table
```

| Problem   | Direction         | Architecture                       |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:----------|:------------------|:-----------------------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 2 | English-to-French | GRU Encoder-Decoder with Attention |           24 |                4.52264 |                       2.71684 |                   1.80579 |                    0.362336 |                  0.446035 |             0.083699 |                           0.344278 |                      0 |          0.07046 |                 460643 |

```text
16. Problem 1 vs Problem 2 Compact Comparison
```

| Problem   | Model                       | Direction         |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:----------|:----------------------------|:------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 1 | Baseline GRU                | English-to-French |           39 |                4.7477  |                       3.52012 |                   1.22758 |                    0.264601 |                  0.305182 |             0.040581 |                           0.252325 |                      0 |         0.031716 |                 298786 |
| Problem 2 | GRU with Bahdanau Attention | English-to-French |           24 |                4.52264 |                       2.71684 |                   1.80579 |                    0.362336 |                  0.446035 |             0.083699 |                           0.344278 |                      0 |         0.07046  |                 460643 |

```text
17. Direct Improvement Table over Problem 1
```

| Metric                       |   Problem 1 English-to-French Baseline GRU |   Problem 2 English-to-French GRU with Attention |   Problem 2 English-to-French GRU with Attention - Problem 1 English-to-French Baseline GRU | Improved?   |
|:-----------------------------|-------------------------------------------:|-------------------------------------------------:|--------------------------------------------------------------------------------------------:|:------------|
| Best Validation Loss         |                                   4.7477   |                                         4.52264  |                                                                                   -0.225063 | Yes         |
| Validation Token Accuracy    |                                   0.264601 |                                         0.362336 |                                                                                    0.097735 | Yes         |
| Training-Validation Loss Gap |                                   1.22758  |                                         1.80579  |                                                                                    0.578216 | No          |
| Exact Match Accuracy         |                                   0        |                                         0        |                                                                                    0        | No          |
| Average BLEU-4               |                                   0.031716 |                                         0.07046  |                                                                                    0.038744 | Yes         |

## Problem 3: Reverse Direction Setup

```python
p3_resource = translation_resources["French-to-English"]

print("Problem 3 preview: French source -> English target")
display(p3_resource["preview_df"])

print("Problem 3 split table")
display(p3_resource["split_table"])

print("Problem 3 vocabulary table")
display(p3_resource["vocab_table"])

print("Problem 3 OOV diagnostic table")
display(p3_resource["diagnostic_df"])
```

**Output**

```text
Problem 3 preview: French source -> English target
```

| French                         | English                    |
|:-------------------------------|:---------------------------|
| j'ai froid                     | i am cold                  |
| tu es fatigué                  | you are tired              |
| il a faim                      | he is hungry               |
| elle est heureuse              | she is happy               |
| nous sommes amis               | we are friends             |
| ils sont étudiants             | they are students          |
| le chat dort                   | the cat is sleeping        |
| le soleil brille               | the sun is shining         |
| nous aimons la musique         | we love music              |
| elle parle français couramment | she speaks french fluently |

```text
Problem 3 split table
```

| Split      |   Percent |   Sentence Pairs |
|:-----------|----------:|-----------------:|
| Training   |        80 |              444 |
| Validation |        20 |              111 |

```text
Problem 3 vocabulary table
```

| Vocabulary                |   Size |
|:--------------------------|-------:|
| French Source Vocabulary  |    994 |
| English Target Vocabulary |    894 |

```text
Problem 3 OOV diagnostic table
```

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique French training source tokens              | 990      |
| Unique English training target tokens             | 890      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV %                           |  38.0577 |
| Validation English OOV %                          |  34.2618 |

## Problem 3: French-to-English Models

```python
for experiment_key in ["p3_baseline", "p3_attention"]:
    experiment_results[experiment_key] = run_configured_experiment(
        config=EXPERIMENT_CONFIGS[experiment_key],
        resources_by_direction=translation_resources,
        common_config=COMMON_CONFIG,
        device=DEVICE
    )

    display(experiment_results[experiment_key]["history_df"])
    display(experiment_results[experiment_key]["best_epoch_df"])
    display(experiment_results[experiment_key]["metrics_df"])
    display(experiment_results[experiment_key]["sample_translations_df"])
```

**Output**

```text
Problem 3A French-to-English Baseline GRU setup
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | Baseline GRU Encoder-Decoder with Fixed Context   |
| Source Language                     | French                                            |
| Target Language                     | English                                           |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
Seq2SeqGRUBaseline(
  (encoder): EncoderGRU(
    (embedding): Embedding(994, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(48, 96, batch_first=True)
  )
  (decoder): DecoderGRUFixedContext(
    (embedding): Embedding(894, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(144, 96, batch_first=True)
    (fc_out): Linear(in_features=96, out_features=894, bias=True)
  )
)
Trainable parameters: 289086
```

```text
Epoch 01/50 | Train Loss: 6.5312 | Val Loss: 6.1124 | Train Tok Acc: 0.1166 | Val Tok Acc: 0.1825 | Loss Gap: -0.4188 | Score: 0.1825 | LR: 0.001000
  New best validation-first score: 0.1825
```

```text
Epoch 02/50 | Train Loss: 5.7137 | Val Loss: 5.7201 | Train Tok Acc: 0.1691 | Val Tok Acc: 0.1825 | Loss Gap: 0.0063 | Score: 0.1824 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 03/50 | Train Loss: 5.3507 | Val Loss: 5.5785 | Train Tok Acc: 0.1712 | Val Tok Acc: 0.1850 | Loss Gap: 0.2278 | Score: 0.1827 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 04/50 | Train Loss: 5.1867 | Val Loss: 5.4471 | Train Tok Acc: 0.1712 | Val Tok Acc: 0.1900 | Loss Gap: 0.2604 | Score: 0.1874 | LR: 0.001000
  New best validation-first score: 0.1874
```

```text
Epoch 05/50 | Train Loss: 5.0658 | Val Loss: 5.3539 | Train Tok Acc: 0.1816 | Val Tok Acc: 0.1963 | Loss Gap: 0.2880 | Score: 0.1934 | LR: 0.001000
  New best validation-first score: 0.1934
```

```text
Epoch 06/50 | Train Loss: 4.9609 | Val Loss: 5.2906 | Train Tok Acc: 0.1883 | Val Tok Acc: 0.2037 | Loss Gap: 0.3297 | Score: 0.2005 | LR: 0.001000
  New best validation-first score: 0.2005
```

```text
Epoch 07/50 | Train Loss: 4.8672 | Val Loss: 5.2281 | Train Tok Acc: 0.1953 | Val Tok Acc: 0.2150 | Loss Gap: 0.3609 | Score: 0.2114 | LR: 0.001000
  New best validation-first score: 0.2114
```

```text
Epoch 08/50 | Train Loss: 4.7809 | Val Loss: 5.1759 | Train Tok Acc: 0.1996 | Val Tok Acc: 0.2300 | Loss Gap: 0.3950 | Score: 0.2261 | LR: 0.001000
  New best validation-first score: 0.2261
```

```text
Epoch 09/50 | Train Loss: 4.6919 | Val Loss: 5.1112 | Train Tok Acc: 0.2069 | Val Tok Acc: 0.2375 | Loss Gap: 0.4193 | Score: 0.2333 | LR: 0.001000
  New best validation-first score: 0.2333
```

```text
Epoch 10/50 | Train Loss: 4.6137 | Val Loss: 5.0768 | Train Tok Acc: 0.2243 | Val Tok Acc: 0.2412 | Loss Gap: 0.4631 | Score: 0.2366 | LR: 0.001000
  New best validation-first score: 0.2366
```

```text
Epoch 11/50 | Train Loss: 4.5354 | Val Loss: 5.0411 | Train Tok Acc: 0.2264 | Val Tok Acc: 0.2462 | Loss Gap: 0.5057 | Score: 0.2412 | LR: 0.001000
  New best validation-first score: 0.2412
```

```text
Epoch 12/50 | Train Loss: 4.4595 | Val Loss: 4.9960 | Train Tok Acc: 0.2356 | Val Tok Acc: 0.2475 | Loss Gap: 0.5365 | Score: 0.2421 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 13/50 | Train Loss: 4.3957 | Val Loss: 4.9738 | Train Tok Acc: 0.2411 | Val Tok Acc: 0.2437 | Loss Gap: 0.5781 | Score: 0.2380 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 14/50 | Train Loss: 4.3348 | Val Loss: 4.9369 | Train Tok Acc: 0.2429 | Val Tok Acc: 0.2462 | Loss Gap: 0.6021 | Score: 0.2402 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 15/50 | Train Loss: 4.2666 | Val Loss: 4.9031 | Train Tok Acc: 0.2499 | Val Tok Acc: 0.2487 | Loss Gap: 0.6364 | Score: 0.2424 | LR: 0.001000
  No useful validation improvement. Patience: 4/4
Early stopping triggered.
Loaded best model from epoch 11 with validation-first score: 0.2412
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.53125 |    6.1124  |               0.11657  |              0.1825  |  -0.418841 |            -0.06593  |          0.1825   |           0.001 |
|       2 |      5.71371 |    5.72006 |               0.169057 |              0.1825  |   0.006349 |            -0.013443 |          0.182437 |           0.001 |
|       3 |      5.35071 |    5.57853 |               0.171193 |              0.185   |   0.227819 |            -0.013807 |          0.182722 |           0.001 |
|       4 |      5.18666 |    5.44711 |               0.171193 |              0.19    |   0.260444 |            -0.018807 |          0.187396 |           0.001 |
|       5 |      5.06583 |    5.35387 |               0.181569 |              0.19625 |   0.288046 |            -0.014681 |          0.19337  |           0.001 |
|       6 |      4.96089 |    5.29057 |               0.188282 |              0.20375 |   0.329681 |            -0.015468 |          0.200453 |           0.001 |
|       7 |      4.86725 |    5.2281  |               0.195301 |              0.215   |   0.360854 |            -0.019699 |          0.211391 |           0.001 |
|       8 |      4.78086 |    5.17586 |               0.199573 |              0.23    |   0.394997 |            -0.030427 |          0.22605  |           0.001 |
|       9 |      4.6919  |    5.1112  |               0.206897 |              0.2375  |   0.419303 |            -0.030603 |          0.233307 |           0.001 |
|      10 |      4.61372 |    5.07677 |               0.224291 |              0.24125 |   0.463051 |            -0.016959 |          0.236619 |           0.001 |
|      11 |      4.5354  |    5.04106 |               0.226427 |              0.24625 |   0.505658 |            -0.019823 |          0.241193 |           0.001 |
|      12 |      4.4595  |    4.99598 |               0.235581 |              0.2475  |   0.536485 |            -0.011919 |          0.242135 |           0.001 |
|      13 |      4.39567 |    4.97376 |               0.241074 |              0.24375 |   0.578095 |            -0.002676 |          0.237969 |           0.001 |
|      14 |      4.33483 |    4.93689 |               0.242905 |              0.24625 |   0.60206  |            -0.003345 |          0.240229 |           0.001 |
|      15 |      4.26664 |    4.90306 |               0.249924 |              0.24875 |   0.63642  |             0.001174 |          0.242386 |           0.001 |

|   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training-Validation Gap |   Validation Perplexity |   Training Token Accuracy at Best Epoch |   Validation Token Accuracy at Best Epoch |   Token Accuracy Gap |   Selection Score |
|-------------:|------------------------------:|-----------------------:|--------------------------:|------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|------------------:|
|           15 |                       4.26664 |                4.90306 |                   0.63642 |                 134.702 |                                0.249924 |                                   0.24875 |             0.001174 |          0.242386 |

| Model                                     |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3A French-to-English Baseline GRU |           15 |                       4.26664 |                4.90306 |                  0.249924 |                     0.24875 |             0.001174 |                           0.242386 |                      0 |         0.031008 |

| French Input                              | Target English                   | Predicted English   |   Exact Match |   BLEU-4 |
|:------------------------------------------|:---------------------------------|:--------------------|--------------:|---------:|
| il a faim                                 | he is hungry                     | she is to           |             0 | 0.113622 |
| la bibliothèque est un endroit calme      | the library is a quiet place     | she is a new        |             0 | 0.103052 |
| elle va régulièrement à la salle de sport | she goes to the gym regularly    | we are to to the    |             0 | 0.093026 |
| nous planifions un voyage à londres       | we are planning a trip to london | we are to the       |             0 | 0.088819 |
| nous planifions un voyage à rome          | we are planning a trip to rome   | we are to the       |             0 | 0.088819 |

```text
Problem 3B French-to-English GRU with Attention setup
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | GRU Encoder-Decoder with Attention                |
| Attention Type                      | Bahdanau Additive Attention                       |
| Source Language                     | French                                            |
| Target Language                     | English                                           |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
Seq2SeqGRUAttention(
  (encoder): EncoderGRU(
    (embedding): Embedding(994, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (gru): GRU(48, 96, batch_first=True)
  )
  (decoder): DecoderGRUAttention(
    (embedding): Embedding(894, 48, padding_idx=0)
    (dropout): Dropout(p=0.35, inplace=False)
    (attention): BahdanauAttention(
      (W_query): Linear(in_features=96, out_features=96, bias=True)
      (W_keys): Linear(in_features=96, out_features=96, bias=True)
      (v): Linear(in_features=96, out_features=1, bias=True)
    )
    (gru): GRU(144, 96, batch_first=True)
    (fc_out): Linear(in_features=240, out_features=894, bias=True)
  )
)
Trainable parameters: 436543
```

```text
Epoch 01/50 | Train Loss: 6.6398 | Val Loss: 6.3179 | Train Tok Acc: 0.0305 | Val Tok Acc: 0.1625 | Loss Gap: -0.3219 | Score: 0.1625 | LR: 0.001000
  New best validation-first score: 0.1625
```

```text
Epoch 02/50 | Train Loss: 5.7152 | Val Loss: 5.6176 | Train Tok Acc: 0.1611 | Val Tok Acc: 0.1825 | Loss Gap: -0.0975 | Score: 0.1825 | LR: 0.001000
  New best validation-first score: 0.1825
```

```text
Epoch 03/50 | Train Loss: 5.1674 | Val Loss: 5.4380 | Train Tok Acc: 0.1675 | Val Tok Acc: 0.1825 | Loss Gap: 0.2706 | Score: 0.1798 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 04/50 | Train Loss: 4.9449 | Val Loss: 5.2977 | Train Tok Acc: 0.1706 | Val Tok Acc: 0.1900 | Loss Gap: 0.3528 | Score: 0.1865 | LR: 0.001000
  New best validation-first score: 0.1865
```

```text
Epoch 05/50 | Train Loss: 4.7651 | Val Loss: 5.1897 | Train Tok Acc: 0.1877 | Val Tok Acc: 0.2125 | Loss Gap: 0.4246 | Score: 0.2083 | LR: 0.001000
  New best validation-first score: 0.2083
```

```text
Epoch 06/50 | Train Loss: 4.5957 | Val Loss: 5.0862 | Train Tok Acc: 0.2042 | Val Tok Acc: 0.2288 | Loss Gap: 0.4906 | Score: 0.2238 | LR: 0.001000
  New best validation-first score: 0.2238
```

```text
Epoch 07/50 | Train Loss: 4.4389 | Val Loss: 5.0150 | Train Tok Acc: 0.2222 | Val Tok Acc: 0.2462 | Loss Gap: 0.5761 | Score: 0.2405 | LR: 0.001000
  New best validation-first score: 0.2405
```

```text
Epoch 08/50 | Train Loss: 4.2832 | Val Loss: 4.9413 | Train Tok Acc: 0.2475 | Val Tok Acc: 0.2550 | Loss Gap: 0.6581 | Score: 0.2484 | LR: 0.001000
  New best validation-first score: 0.2484
```

```text
Epoch 09/50 | Train Loss: 4.1210 | Val Loss: 4.8681 | Train Tok Acc: 0.2658 | Val Tok Acc: 0.2662 | Loss Gap: 0.7471 | Score: 0.2588 | LR: 0.001000
  New best validation-first score: 0.2588
```

```text
Epoch 10/50 | Train Loss: 3.9851 | Val Loss: 4.8215 | Train Tok Acc: 0.2887 | Val Tok Acc: 0.2888 | Loss Gap: 0.8364 | Score: 0.2804 | LR: 0.001000
  New best validation-first score: 0.2804
```

```text
Epoch 11/50 | Train Loss: 3.8398 | Val Loss: 4.7633 | Train Tok Acc: 0.3082 | Val Tok Acc: 0.2838 | Loss Gap: 0.9235 | Score: 0.2745 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 12/50 | Train Loss: 3.7069 | Val Loss: 4.7003 | Train Tok Acc: 0.3277 | Val Tok Acc: 0.3025 | Loss Gap: 0.9934 | Score: 0.2926 | LR: 0.001000
  New best validation-first score: 0.2926
```

```text
Epoch 13/50 | Train Loss: 3.5725 | Val Loss: 4.6671 | Train Tok Acc: 0.3409 | Val Tok Acc: 0.3137 | Loss Gap: 1.0945 | Score: 0.3028 | LR: 0.001000
  New best validation-first score: 0.3028
```

```text
Epoch 14/50 | Train Loss: 3.4639 | Val Loss: 4.6200 | Train Tok Acc: 0.3518 | Val Tok Acc: 0.3250 | Loss Gap: 1.1560 | Score: 0.3134 | LR: 0.001000
  New best validation-first score: 0.3134
```

```text
Epoch 15/50 | Train Loss: 3.3448 | Val Loss: 4.5766 | Train Tok Acc: 0.3714 | Val Tok Acc: 0.3438 | Loss Gap: 1.2318 | Score: 0.3314 | LR: 0.001000
  New best validation-first score: 0.3314
```

```text
Epoch 16/50 | Train Loss: 3.2361 | Val Loss: 4.5374 | Train Tok Acc: 0.3943 | Val Tok Acc: 0.3463 | Loss Gap: 1.3013 | Score: 0.3332 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 17/50 | Train Loss: 3.1400 | Val Loss: 4.5121 | Train Tok Acc: 0.3976 | Val Tok Acc: 0.3588 | Loss Gap: 1.3721 | Score: 0.3450 | LR: 0.001000
  New best validation-first score: 0.3450
```

```text
Epoch 18/50 | Train Loss: 3.0494 | Val Loss: 4.4968 | Train Tok Acc: 0.4171 | Val Tok Acc: 0.3663 | Loss Gap: 1.4474 | Score: 0.3518 | LR: 0.001000
  New best validation-first score: 0.3518
```

```text
Epoch 19/50 | Train Loss: 2.9667 | Val Loss: 4.4725 | Train Tok Acc: 0.4242 | Val Tok Acc: 0.3750 | Loss Gap: 1.5058 | Score: 0.3599 | LR: 0.001000
  New best validation-first score: 0.3599
```

```text
Epoch 20/50 | Train Loss: 2.8828 | Val Loss: 4.4356 | Train Tok Acc: 0.4376 | Val Tok Acc: 0.3912 | Loss Gap: 1.5528 | Score: 0.3757 | LR: 0.001000
  New best validation-first score: 0.3757
```

```text
Epoch 21/50 | Train Loss: 2.7968 | Val Loss: 4.4384 | Train Tok Acc: 0.4559 | Val Tok Acc: 0.3837 | Loss Gap: 1.6416 | Score: 0.3673 | LR: 0.001000
  No useful validation improvement. Patience: 1/4
```

```text
Epoch 22/50 | Train Loss: 2.7359 | Val Loss: 4.4108 | Train Tok Acc: 0.4620 | Val Tok Acc: 0.3875 | Loss Gap: 1.6749 | Score: 0.3708 | LR: 0.001000
  No useful validation improvement. Patience: 2/4
```

```text
Epoch 23/50 | Train Loss: 2.6597 | Val Loss: 4.4088 | Train Tok Acc: 0.4693 | Val Tok Acc: 0.3862 | Loss Gap: 1.7491 | Score: 0.3688 | LR: 0.001000
  No useful validation improvement. Patience: 3/4
```

```text
Epoch 24/50 | Train Loss: 2.5959 | Val Loss: 4.3840 | Train Tok Acc: 0.5029 | Val Tok Acc: 0.3937 | Loss Gap: 1.7882 | Score: 0.3759 | LR: 0.001000
  No useful validation improvement. Patience: 4/4
Early stopping triggered.
Loaded best model from epoch 20 with validation-first score: 0.3757
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.63984 |    6.3179  |               0.030516 |              0.1625  |  -0.321935 |            -0.131984 |          0.1625   |           0.001 |
|       2 |      5.71518 |    5.61764 |               0.161123 |              0.1825  |  -0.097544 |            -0.021377 |          0.1825   |           0.001 |
|       3 |      5.16737 |    5.43796 |               0.167531 |              0.1825  |   0.270589 |            -0.014969 |          0.179794 |           0.001 |
|       4 |      4.94488 |    5.29772 |               0.170583 |              0.19    |   0.352846 |            -0.019417 |          0.186472 |           0.001 |
|       5 |      4.76506 |    5.18966 |               0.187672 |              0.2125  |   0.424597 |            -0.024828 |          0.208254 |           0.001 |
|       6 |      4.59567 |    5.08623 |               0.20415  |              0.22875 |   0.490568 |            -0.0246   |          0.223844 |           0.001 |
|       7 |      4.43889 |    5.01496 |               0.222154 |              0.24625 |   0.576069 |            -0.024096 |          0.240489 |           0.001 |
|       8 |      4.28322 |    4.94129 |               0.247482 |              0.255   |   0.658078 |            -0.007518 |          0.248419 |           0.001 |
|       9 |      4.12102 |    4.86809 |               0.265792 |              0.26625 |   0.747069 |            -0.000458 |          0.258779 |           0.001 |
|      10 |      3.98514 |    4.82151 |               0.288679 |              0.28875 |   0.836378 |            -7.1e-05  |          0.280386 |           0.001 |
|      11 |      3.83984 |    4.76335 |               0.308209 |              0.28375 |   0.92351  |             0.024459 |          0.274515 |           0.001 |
|      12 |      3.70686 |    4.70031 |               0.327739 |              0.3025  |   0.993446 |             0.025239 |          0.292566 |           0.001 |
|      13 |      3.57254 |    4.66709 |               0.340861 |              0.31375 |   1.09455  |             0.027111 |          0.302805 |           0.001 |
|      14 |      3.46393 |    4.61995 |               0.351846 |              0.325   |   1.15602  |             0.026846 |          0.31344  |           0.001 |
|      15 |      3.34477 |    4.5766  |               0.371376 |              0.34375 |   1.23183  |             0.027626 |          0.331432 |           0.001 |
|      16 |      3.2361  |    4.5374  |               0.394263 |              0.34625 |   1.3013   |             0.048013 |          0.333237 |           0.001 |
|      17 |      3.14003 |    4.51208 |               0.39762  |              0.35875 |   1.37206  |             0.03887  |          0.345029 |           0.001 |
|      18 |      3.04944 |    4.49681 |               0.41715  |              0.36625 |   1.44737  |             0.0509   |          0.351776 |           0.001 |
|      19 |      2.96668 |    4.47248 |               0.424168 |              0.375   |   1.5058   |             0.049168 |          0.359942 |           0.001 |
|      20 |      2.88282 |    4.43562 |               0.437595 |              0.39125 |   1.5528   |             0.046345 |          0.375722 |           0.001 |
|      21 |      2.79677 |    4.43839 |               0.455905 |              0.38375 |   1.64162  |             0.072155 |          0.367334 |           0.001 |
|      22 |      2.73589 |    4.41083 |               0.462008 |              0.3875  |   1.67494  |             0.074508 |          0.370751 |           0.001 |
|      23 |      2.65969 |    4.40876 |               0.469332 |              0.38625 |   1.74907  |             0.083082 |          0.368759 |           0.001 |
|      24 |      2.59587 |    4.38403 |               0.502899 |              0.39375 |   1.78816  |             0.109149 |          0.375868 |           0.001 |

|   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training-Validation Gap |   Validation Perplexity |   Training Token Accuracy at Best Epoch |   Validation Token Accuracy at Best Epoch |   Token Accuracy Gap |   Selection Score |
|-------------:|------------------------------:|-----------------------:|--------------------------:|------------------------:|----------------------------------------:|------------------------------------------:|---------------------:|------------------:|
|           24 |                       2.59587 |                4.38403 |                   1.78816 |                 80.1604 |                                0.502899 |                                   0.39375 |             0.109149 |          0.375868 |

| Model                                           |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3B French-to-English GRU with Attention |           24 |                       2.59587 |                4.38403 |                  0.502899 |                     0.39375 |             0.109149 |                           0.375868 |                      0 |         0.063423 |

| French Input                                      | Target English                                    | Predicted English                     |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:--------------------------------------|--------------:|---------:|
| ils parlent souvent des efforts de conservatio... | they often talk about environmental conservati... | they often talk about in the distance |             0 | 0.411134 |
| j'aime vraiment marcher dans la forêt dense       | i truly enjoy walking in the dense forest         | i enjoy walking in the park           |             0 | 0.384982 |
| il conduit une berline noire                      | he drives a black sedan                           | he drives a beautiful                 |             0 | 0.309679 |
| nous devons acheter du pain                       | we need to buy some bread                         | we are going to buy some              |             0 | 0.217119 |
| je veux une tasse de café chaud                   | i want a hot cup of coffee                        | i want a new pair of                  |             0 | 0.183787 |

## Problem 3: Plots and Comparison

```python
plot_configured_experiment_outputs(
    experiment=experiment_results["p3_baseline"],
    title="Problem 3A: French-to-English Baseline GRU Training vs Validation Loss",
    show_best_epoch=True
)

plot_configured_experiment_outputs(
    experiment=experiment_results["p3_attention"],
    title="Problem 3B: French-to-English GRU with Attention Training vs Validation Loss",
    show_best_epoch=True
)

p3_comparison = display_configured_comparison(
    experiment_results=experiment_results,
    experiment_keys=["p3_baseline", "p3_attention"],
    title="Problem 3 Baseline vs Attention Comparison"
)

display_configured_attention_examples(experiment_results["p3_attention"], n=2)
```

**Output**

![Output](README_files/output_28_5.png)

```text
Best selected epoch: 15
Validation loss at selected epoch: 4.9031
Problem 3A Final Validation Metrics
```

| Model                                     |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3A French-to-English Baseline GRU |           15 |                       4.26664 |                4.90306 |                  0.249924 |                     0.24875 |             0.001174 |                           0.242386 |                      0 |         0.031008 |

```text
Problem 3A Qualitative Validation Examples
```

| French Input                              | Target English                   | Predicted English   |   Exact Match |   BLEU-4 |
|:------------------------------------------|:---------------------------------|:--------------------|--------------:|---------:|
| il a faim                                 | he is hungry                     | she is to           |             0 | 0.113622 |
| la bibliothèque est un endroit calme      | the library is a quiet place     | she is a new        |             0 | 0.103052 |
| elle va régulièrement à la salle de sport | she goes to the gym regularly    | we are to to the    |             0 | 0.093026 |
| nous planifions un voyage à londres       | we are planning a trip to london | we are to the       |             0 | 0.088819 |
| nous planifions un voyage à rome          | we are planning a trip to rome   | we are to the       |             0 | 0.088819 |

![Output](README_files/output_28_6.png)

```text
Best selected epoch: 24
Validation loss at selected epoch: 4.3840
Problem 3B Final Validation Metrics
```

| Model                                           |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3B French-to-English GRU with Attention |           24 |                       2.59587 |                4.38403 |                  0.502899 |                     0.39375 |             0.109149 |                           0.375868 |                      0 |         0.063423 |

```text
Problem 3B Qualitative Validation Examples
```

| French Input                                      | Target English                                    | Predicted English                     |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:--------------------------------------|--------------:|---------:|
| ils parlent souvent des efforts de conservatio... | they often talk about environmental conservati... | they often talk about in the distance |             0 | 0.411134 |
| j'aime vraiment marcher dans la forêt dense       | i truly enjoy walking in the dense forest         | i enjoy walking in the park           |             0 | 0.384982 |
| il conduit une berline noire                      | he drives a black sedan                           | he drives a beautiful                 |             0 | 0.309679 |
| nous devons acheter du pain                       | we need to buy some bread                         | we are going to buy some              |             0 | 0.217119 |
| je veux une tasse de café chaud                   | i want a hot cup of coffee                        | i want a new pair of                  |             0 | 0.183787 |

```text
Problem 3 Baseline vs Attention Comparison
```

| Problem    | Model                       | Direction         |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:-----------|:----------------------------|:------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 3A | Baseline GRU                | French-to-English |           15 |                4.90306 |                       4.26664 |                   0.63642 |                     0.24875 |                  0.249924 |             0.001174 |                           0.242386 |                      0 |         0.031008 |                 289086 |
| Problem 3B | GRU with Bahdanau Attention | French-to-English |           24 |                4.38403 |                       2.59587 |                   1.78816 |                     0.39375 |                  0.502899 |             0.109149 |                           0.375868 |                      0 |         0.063423 |                 436543 |

```text
Problem 3B Attention Map Example 1
French Input: ils parlent souvent des efforts de conservation de l'environnement
Target English: they often talk about environmental conservation efforts
Predicted English: they often talk about in the distance
Exact Match: 0
BLEU-4: 0.41113361690051975
```

![Output](README_files/output_28_7.png)

```text
Problem 3B Attention Map Example 2
French Input: j'aime vraiment marcher dans la forêt dense
Target English: i truly enjoy walking in the dense forest
Predicted English: i enjoy walking in the park
Exact Match: 0
BLEU-4: 0.38498150077635496
```

![Output](README_files/output_28_8.png)

## Problem 3: Metrics and Final Comparison

```python
p3_baseline_all_results = display_single_experiment_all_results(
    experiment=experiment_results["p3_baseline"],
    total_pairs=len(pairs),
    common_config=COMMON_CONFIG
)

p3_attention_all_results = display_single_experiment_all_results(
    experiment=experiment_results["p3_attention"],
    total_pairs=len(pairs),
    common_config=COMMON_CONFIG
)

print("\nProblem 3 Baseline vs Attention Comparison")
display(p3_comparison)

final_all_models_comparison = display_configured_comparison(
    experiment_results=experiment_results,
    experiment_keys=["p1_baseline", "p2_attention", "p3_baseline", "p3_attention"],
    title="Final Four-Model Comparison"
)

direction_comparison = make_direction_level_comparison(final_all_models_comparison)
print("\nDirection-Level Average Comparison")
display(direction_comparison)
```

**Output**

```text
================================================================================
PROBLEM 3A FRENCH-TO-ENGLISH BASELINE GRU — ALL METRICS AND RESULTS
================================================================================

1. Dataset and Split Table
```

| Metric                    | Value   |
|:--------------------------|:--------|
| Total Sentence Pairs      | 555     |
| Training Sentence Pairs   | 444     |
| Validation Sentence Pairs | 111     |
| Training Percent          | 80%     |
| Validation Percent        | 20%     |
| Max Sequence Length       | 25      |
| Random Seed               | 42      |

```text
2. Vocabulary Table
```

| Vocabulary                |   Size |
|:--------------------------|-------:|
| French Source Vocabulary  |    994 |
| English Target Vocabulary |    894 |

```text
3. Out-of-Vocabulary Diagnostic Table
```

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique French training source tokens              | 990      |
| Unique English training target tokens             | 890      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV %                           |  38.0577 |
| Validation English OOV %                          |  34.2618 |

```text
4. Hyperparameter and Design Choice Table
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | Baseline GRU Encoder-Decoder with Fixed Context   |
| Source Language                     | French                                            |
| Target Language                     | English                                           |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
5. Model Size Table
```

| Metric                 |   Value |
|:-----------------------|--------:|
| Trainable Parameters   |  289086 |
| Source Vocabulary Size |     994 |
| Target Vocabulary Size |     894 |
| Embedding Size         |      48 |
| Hidden Size            |      96 |
| GRU Layers             |       1 |

```text
6. Full Training History
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.53125 |    6.1124  |               0.11657  |              0.1825  |  -0.418841 |            -0.06593  |          0.1825   |           0.001 |
|       2 |      5.71371 |    5.72006 |               0.169057 |              0.1825  |   0.006349 |            -0.013443 |          0.182437 |           0.001 |
|       3 |      5.35071 |    5.57853 |               0.171193 |              0.185   |   0.227819 |            -0.013807 |          0.182722 |           0.001 |
|       4 |      5.18666 |    5.44711 |               0.171193 |              0.19    |   0.260444 |            -0.018807 |          0.187396 |           0.001 |
|       5 |      5.06583 |    5.35387 |               0.181569 |              0.19625 |   0.288046 |            -0.014681 |          0.19337  |           0.001 |
|       6 |      4.96089 |    5.29057 |               0.188282 |              0.20375 |   0.329681 |            -0.015468 |          0.200453 |           0.001 |
|       7 |      4.86725 |    5.2281  |               0.195301 |              0.215   |   0.360854 |            -0.019699 |          0.211391 |           0.001 |
|       8 |      4.78086 |    5.17586 |               0.199573 |              0.23    |   0.394997 |            -0.030427 |          0.22605  |           0.001 |
|       9 |      4.6919  |    5.1112  |               0.206897 |              0.2375  |   0.419303 |            -0.030603 |          0.233307 |           0.001 |
|      10 |      4.61372 |    5.07677 |               0.224291 |              0.24125 |   0.463051 |            -0.016959 |          0.236619 |           0.001 |
|      11 |      4.5354  |    5.04106 |               0.226427 |              0.24625 |   0.505658 |            -0.019823 |          0.241193 |           0.001 |
|      12 |      4.4595  |    4.99598 |               0.235581 |              0.2475  |   0.536485 |            -0.011919 |          0.242135 |           0.001 |
|      13 |      4.39567 |    4.97376 |               0.241074 |              0.24375 |   0.578095 |            -0.002676 |          0.237969 |           0.001 |
|      14 |      4.33483 |    4.93689 |               0.242905 |              0.24625 |   0.60206  |            -0.003345 |          0.240229 |           0.001 |
|      15 |      4.26664 |    4.90306 |               0.249924 |              0.24875 |   0.63642  |             0.001174 |          0.242386 |           0.001 |

```text
7. Best Epoch and Final Epoch Table
```

| Metric                                  |      Value |
|:----------------------------------------|-----------:|
| Epochs Completed                        |  15        |
| Best Selected Epoch                     |  15        |
| Training Loss at Best Epoch             |   4.26664  |
| Validation Loss at Best Epoch           |   4.90306  |
| Training-Validation Gap at Best Epoch   |   0.63642  |
| Validation Perplexity at Best Epoch     | 134.702    |
| Training Token Accuracy at Best Epoch   |   0.249924 |
| Validation Token Accuracy at Best Epoch |   0.24875  |
| Token Accuracy Gap at Best Epoch        |   0.001174 |
| Validation-First Selection Score        |   0.242386 |
| Final Epoch                             |  15        |
| Final Training Loss                     |   4.26664  |
| Final Validation Loss                   |   4.90306  |
| Final Training-Validation Gap           |   0.63642  |
| Final Validation Perplexity             | 134.702    |
| Final Training Token Accuracy           |   0.249924 |
| Final Validation Token Accuracy         |   0.24875  |
| Final Token Accuracy Gap                |   0.001174 |
| Final Selection Score                   |   0.242386 |

```text
8. Final Translation Metrics
```

| Model                                     |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3A French-to-English Baseline GRU |           15 |                       4.26664 |                4.90306 |                  0.249924 |                     0.24875 |             0.001174 |                           0.242386 |                      0 |         0.031008 |

```text
9. Exact Match Count Table
```

| Metric                |   Value |
|:----------------------|--------:|
| Validation Examples   |     111 |
| Exact Match Count     |       0 |
| Non-Exact Match Count |     111 |
| Exact Match Accuracy  |       0 |

```text
10. BLEU-4 Distribution Table
```

| Metric                    |    Value |
|:--------------------------|---------:|
| Average BLEU-4            | 0.031008 |
| Median BLEU-4             | 0.029487 |
| Minimum BLEU-4            | 0        |
| Maximum BLEU-4            | 0.113622 |
| Standard Deviation BLEU-4 | 0.029723 |
| BLEU-4 >= 0.25 Count      | 0        |
| BLEU-4 >= 0.50 Count      | 0        |
| BLEU-4 >= 0.75 Count      | 0        |

```text
11. Top 10 Validation Examples by BLEU-4
```

| French Input                              | Target English                    | Predicted English   |   Exact Match |   BLEU-4 |
|:------------------------------------------|:----------------------------------|:--------------------|--------------:|---------:|
| il a faim                                 | he is hungry                      | she is to           |             0 | 0.113622 |
| la bibliothèque est un endroit calme      | the library is a quiet place      | she is a new        |             0 | 0.103052 |
| elle va régulièrement à la salle de sport | she goes to the gym regularly     | we are to to the    |             0 | 0.093026 |
| nous planifions un voyage à londres       | we are planning a trip to london  | we are to the       |             0 | 0.088819 |
| nous planifions un voyage à rome          | we are planning a trip to rome    | we are to the       |             0 | 0.088819 |
| elle visite ses grands-parents            | she visits her grandparents       | she is a            |             0 | 0.081414 |
| le chat dort                              | the cat is sleeping               | she is to           |             0 | 0.081414 |
| elle parle français couramment            | she speaks french fluently        | she is a            |             0 | 0.081414 |
| nous construisons un château de sable     | we build a sandcastle             | we are to the       |             0 | 0.080343 |
| ils vont à la plage chaque été            | they go to the beach every summer | we are to to the    |             0 | 0.076163 |

```text
12. Lowest 10 Validation Examples by BLEU-4
```

| French Input                                      | Target English                                    | Predicted English   |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:--------------------|--------------:|---------:|
| elle a gagné un match de tennis                   | she won a tennis match                            | we are to the       |             0 |        0 |
| l'eau est froide                                  | the water is cold                                 | we are to           |             0 |        0 |
| elle porte un beau bracelet en argent au poignet  | she wears a beautiful silver bracelet on her w... | we are to to the    |             0 |        0 |
| ils ont construit un épais mur en béton pour l... | they built a thick concrete wall for safety       | we are to to the    |             0 |        0 |
| il conduit une berline noire                      | he drives a black sedan                           | we are to the       |             0 |        0 |
| elle travaille dans une entreprise de logiciels   | she works at a software company                   | we are to the       |             0 |        0 |
| il chante dans le ch ur                           | he sings in the choir                             | we are a new        |             0 |        0 |
| il a fini son livre hier                          | he finished his book yesterday                    | we are to the       |             0 |        0 |
| il a fini son test en avance                      | he finished his test early                        | we are to to the    |             0 |        0 |
| comment ça va ?                                   | how are you ?                                     | she is a new        |             0 |        0 |

```text
13. Exact Match Examples
No exact match examples were found for Problem 3A.

14. Full Validation Translation Results
```

| Unnamed: 0   | French Input                                      | Target English                                    | Predicted English   | Exact Match   | BLEU-4   |
|:-------------|:--------------------------------------------------|:--------------------------------------------------|:--------------------|:--------------|:---------|
| 0            | elle a gagné un match de tennis                   | she won a tennis match                            | we are to the       | 0             | 0.000000 |
| 1            | le marché biologique ouvre à l'aube le samedi     | the organic market opens at dawn on saturdays     | we are to to the    | 0             | 0.029487 |
| 2            | nous regardons un film ensemble                   | we watch a movie together                         | we are to the       | 0             | 0.062571 |
| 3            | le pain de cette boulangerie est toujours crou... | the bread at this bakery is always crunchy        | we are to to the    | 0             | 0.029487 |
| 4            | nous dansons au mariage                           | we dance at the wedding                           | we are to           | 0             | 0.058335 |
| ...          | ...                                               | ...                                               | ...                 | ...           | ...      |
| 106          | ils jouent au football chaque week-end            | they play soccer every weekend                    | we are to the       | 0             | 0.000000 |
| 107          | je veux une tasse de café chaud                   | i want a hot cup of coffee                        | we are to to the    | 0             | 0.000000 |
| 108          | le livre est sur la table                         | the book is on the table                          | we are to the       | 0             | 0.048730 |
| 109          | il a terminé son projet scientifique en avance... | he finished his science project ahead of schedule | we are to to the    | 0             | 0.000000 |
| 110          | nous devons acheter une douzaine d' ufs frais     | we need to buy a dozen fresh eggs                 | we are to to the    | 0             | 0.035066 |

```text
15. Compact Final Table for Report Table
```

| Problem    | Direction         | Architecture                                    |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:-----------|:------------------|:------------------------------------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 3A | French-to-English | Baseline GRU Encoder-Decoder with Fixed Context |           15 |                4.90306 |                       4.26664 |                   0.63642 |                     0.24875 |                  0.249924 |             0.001174 |                           0.242386 |                      0 |         0.031008 |                 289086 |

```text
================================================================================
PROBLEM 3B FRENCH-TO-ENGLISH GRU WITH ATTENTION — ALL METRICS AND RESULTS
================================================================================

1. Dataset and Split Table
```

| Metric                    | Value   |
|:--------------------------|:--------|
| Total Sentence Pairs      | 555     |
| Training Sentence Pairs   | 444     |
| Validation Sentence Pairs | 111     |
| Training Percent          | 80%     |
| Validation Percent        | 20%     |
| Max Sequence Length       | 25      |
| Random Seed               | 42      |

```text
2. Vocabulary Table
```

| Vocabulary                |   Size |
|:--------------------------|-------:|
| French Source Vocabulary  |    994 |
| English Target Vocabulary |    894 |

```text
3. Out-of-Vocabulary Diagnostic Table
```

| Check                                             |    Value |
|:--------------------------------------------------|---------:|
| Total sentence pairs                              | 555      |
| Training sentence pairs                           | 444      |
| Validation sentence pairs                         | 111      |
| Unique French training source tokens              | 990      |
| Unique English training target tokens             | 890      |
| Validation French OOV tokens compared to training | 145      |
| Validation English OOV tokens compared to trai... | 123      |
| Validation French OOV %                           |  38.0577 |
| Validation English OOV %                          |  34.2618 |

```text
4. Hyperparameter and Design Choice Table
```

| Choice                              | Value                                             |
|:------------------------------------|:--------------------------------------------------|
| Architecture                        | GRU Encoder-Decoder with Attention                |
| Attention Type                      | Bahdanau Additive Attention                       |
| Source Language                     | French                                            |
| Target Language                     | English                                           |
| Embedding Size                      | 48                                                |
| Hidden Size                         | 96                                                |
| GRU Layers                          | 1                                                 |
| Dropout                             | 0.35                                              |
| Batch Size                          | 32                                                |
| Epochs                              | 50                                                |
| Learning Rate                       | 0.001                                             |
| Learning-Rate Scheduler             | ReduceLROnPlateau                                 |
| Optimizer                           | Adam                                              |
| Weight Decay                        | 0.001                                             |
| Loss Function                       | Cross-Entropy Loss with <pad> ignored and labe... |
| Label Smoothing                     | 0.05                                              |
| Gradient Clipping                   | 1.0                                               |
| Early Stopping Patience             | 4                                                 |
| Model Selection Metric              | validation token accuracy minus train-validati... |
| Selection Min Delta                 | 0.002                                             |
| Overfit Penalty Weight              | 0.01                                              |
| Decoding Strategy                   | beam                                              |
| Beam Width                          | 3                                                 |
| Length Penalty Alpha                | 0.7                                               |
| Minimum Decode Length               | 1                                                 |
| Mask Special Tokens During Decoding | True                                              |
| Max Sequence Length                 | 25                                                |

```text
5. Model Size Table
```

| Metric                 |   Value |
|:-----------------------|--------:|
| Trainable Parameters   |  436543 |
| Source Vocabulary Size |     994 |
| Target Vocabulary Size |     894 |
| Embedding Size         |      48 |
| Hidden Size            |      96 |
| GRU Layers             |       1 |

```text
6. Full Training History
```

|   epoch |   train_loss |   val_loss |   train_token_accuracy |   val_token_accuracy |   loss_gap |   token_accuracy_gap |   selection_score |   learning_rate |
|--------:|-------------:|-----------:|-----------------------:|---------------------:|-----------:|---------------------:|------------------:|----------------:|
|       1 |      6.63984 |    6.3179  |               0.030516 |              0.1625  |  -0.321935 |            -0.131984 |          0.1625   |           0.001 |
|       2 |      5.71518 |    5.61764 |               0.161123 |              0.1825  |  -0.097544 |            -0.021377 |          0.1825   |           0.001 |
|       3 |      5.16737 |    5.43796 |               0.167531 |              0.1825  |   0.270589 |            -0.014969 |          0.179794 |           0.001 |
|       4 |      4.94488 |    5.29772 |               0.170583 |              0.19    |   0.352846 |            -0.019417 |          0.186472 |           0.001 |
|       5 |      4.76506 |    5.18966 |               0.187672 |              0.2125  |   0.424597 |            -0.024828 |          0.208254 |           0.001 |
|       6 |      4.59567 |    5.08623 |               0.20415  |              0.22875 |   0.490568 |            -0.0246   |          0.223844 |           0.001 |
|       7 |      4.43889 |    5.01496 |               0.222154 |              0.24625 |   0.576069 |            -0.024096 |          0.240489 |           0.001 |
|       8 |      4.28322 |    4.94129 |               0.247482 |              0.255   |   0.658078 |            -0.007518 |          0.248419 |           0.001 |
|       9 |      4.12102 |    4.86809 |               0.265792 |              0.26625 |   0.747069 |            -0.000458 |          0.258779 |           0.001 |
|      10 |      3.98514 |    4.82151 |               0.288679 |              0.28875 |   0.836378 |            -7.1e-05  |          0.280386 |           0.001 |
|      11 |      3.83984 |    4.76335 |               0.308209 |              0.28375 |   0.92351  |             0.024459 |          0.274515 |           0.001 |
|      12 |      3.70686 |    4.70031 |               0.327739 |              0.3025  |   0.993446 |             0.025239 |          0.292566 |           0.001 |
|      13 |      3.57254 |    4.66709 |               0.340861 |              0.31375 |   1.09455  |             0.027111 |          0.302805 |           0.001 |
|      14 |      3.46393 |    4.61995 |               0.351846 |              0.325   |   1.15602  |             0.026846 |          0.31344  |           0.001 |
|      15 |      3.34477 |    4.5766  |               0.371376 |              0.34375 |   1.23183  |             0.027626 |          0.331432 |           0.001 |
|      16 |      3.2361  |    4.5374  |               0.394263 |              0.34625 |   1.3013   |             0.048013 |          0.333237 |           0.001 |
|      17 |      3.14003 |    4.51208 |               0.39762  |              0.35875 |   1.37206  |             0.03887  |          0.345029 |           0.001 |
|      18 |      3.04944 |    4.49681 |               0.41715  |              0.36625 |   1.44737  |             0.0509   |          0.351776 |           0.001 |
|      19 |      2.96668 |    4.47248 |               0.424168 |              0.375   |   1.5058   |             0.049168 |          0.359942 |           0.001 |
|      20 |      2.88282 |    4.43562 |               0.437595 |              0.39125 |   1.5528   |             0.046345 |          0.375722 |           0.001 |
|      21 |      2.79677 |    4.43839 |               0.455905 |              0.38375 |   1.64162  |             0.072155 |          0.367334 |           0.001 |
|      22 |      2.73589 |    4.41083 |               0.462008 |              0.3875  |   1.67494  |             0.074508 |          0.370751 |           0.001 |
|      23 |      2.65969 |    4.40876 |               0.469332 |              0.38625 |   1.74907  |             0.083082 |          0.368759 |           0.001 |
|      24 |      2.59587 |    4.38403 |               0.502899 |              0.39375 |   1.78816  |             0.109149 |          0.375868 |           0.001 |

```text
7. Best Epoch and Final Epoch Table
```

| Metric                                  |     Value |
|:----------------------------------------|----------:|
| Epochs Completed                        | 24        |
| Best Selected Epoch                     | 24        |
| Training Loss at Best Epoch             |  2.59587  |
| Validation Loss at Best Epoch           |  4.38403  |
| Training-Validation Gap at Best Epoch   |  1.78816  |
| Validation Perplexity at Best Epoch     | 80.1604   |
| Training Token Accuracy at Best Epoch   |  0.502899 |
| Validation Token Accuracy at Best Epoch |  0.39375  |
| Token Accuracy Gap at Best Epoch        |  0.109149 |
| Validation-First Selection Score        |  0.375868 |
| Final Epoch                             | 24        |
| Final Training Loss                     |  2.59587  |
| Final Validation Loss                   |  4.38403  |
| Final Training-Validation Gap           |  1.78816  |
| Final Validation Perplexity             | 80.1604   |
| Final Training Token Accuracy           |  0.502899 |
| Final Validation Token Accuracy         |  0.39375  |
| Final Token Accuracy Gap                |  0.109149 |
| Final Selection Score                   |  0.375868 |

```text
8. Final Translation Metrics
```

| Model                                           |   Best Epoch |   Training Loss at Best Epoch |   Best Validation Loss |   Training Token Accuracy |   Validation Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------------------------------------|-------------:|------------------------------:|-----------------------:|--------------------------:|----------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|
| Problem 3B French-to-English GRU with Attention |           24 |                       2.59587 |                4.38403 |                  0.502899 |                     0.39375 |             0.109149 |                           0.375868 |                      0 |         0.063423 |

```text
9. Exact Match Count Table
```

| Metric                |   Value |
|:----------------------|--------:|
| Validation Examples   |     111 |
| Exact Match Count     |       0 |
| Non-Exact Match Count |     111 |
| Exact Match Accuracy  |       0 |

```text
10. BLEU-4 Distribution Table
```

| Metric                    |    Value |
|:--------------------------|---------:|
| Average BLEU-4            | 0.063423 |
| Median BLEU-4             | 0.04873  |
| Minimum BLEU-4            | 0        |
| Maximum BLEU-4            | 0.411134 |
| Standard Deviation BLEU-4 | 0.062314 |
| BLEU-4 >= 0.25 Count      | 3        |
| BLEU-4 >= 0.50 Count      | 0        |
| BLEU-4 >= 0.75 Count      | 0        |

```text
11. Top 10 Validation Examples by BLEU-4
```

| French Input                                      | Target English                                    | Predicted English                     |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:--------------------------------------|--------------:|---------:|
| ils parlent souvent des efforts de conservatio... | they often talk about environmental conservati... | they often talk about in the distance |             0 | 0.411134 |
| j'aime vraiment marcher dans la forêt dense       | i truly enjoy walking in the dense forest         | i enjoy walking in the park           |             0 | 0.384982 |
| il conduit une berline noire                      | he drives a black sedan                           | he drives a beautiful                 |             0 | 0.309679 |
| nous devons acheter du pain                       | we need to buy some bread                         | we are going to buy some              |             0 | 0.217119 |
| je veux une tasse de café chaud                   | i want a hot cup of coffee                        | i want a new pair of                  |             0 | 0.183787 |
| je veux un verre froid de café glacé              | i want a cold glass of iced coffee                | i want a new pair of                  |             0 | 0.155572 |
| ils ont construit un épais mur en béton pour l... | they built a thick concrete wall for safety       | they built a new pair of              |             0 | 0.144776 |
| vous avez une vue spectaculaire depuis votre b... | you have a spectacular view from your balcony     | you have a new pair of shoes          |             0 | 0.141718 |
| elle porte un beau bracelet en argent au poignet  | she wears a beautiful silver bracelet on her w... | she wears a traditional sunday        |             0 | 0.119483 |
| ils font de la randonnée dans la forêt            | they hike in the forest                           | they share their in the pool          |             0 | 0.095544 |

```text
12. Lowest 10 Validation Examples by BLEU-4
```

| French Input                                      | Target English                                    | Predicted English            |   Exact Match |   BLEU-4 |
|:--------------------------------------------------|:--------------------------------------------------|:-----------------------------|--------------:|---------:|
| il chante dans le ch ur                           | he sings in the choir                             | i look forward to drink      |             0 | 0        |
| comment ça va ?                                   | how are you ?                                     | i look forward               |             0 | 0        |
| l'hiver est très froid ici                        | the winter is very cold here                      | i look forward to drink      |             0 | 0        |
| j'aime cette chanson                              | i love this song                                  | we are going                 |             0 | 0        |
| l'été est très chaud ici                          | the summer is very hot here                       | i look forward to            |             0 | 0        |
| j'ai froid                                        | i am cold                                         | he cleans a beautiful        |             0 | 0        |
| l'enseignant parle lentement                      | the teacher speaks slowly                         | she is looking               |             0 | 0        |
| quelle heure est-il ?                             | what time is it ?                                 | i look forward               |             0 | 0        |
| le fleuriste du coin vend des roses fraîches      | the flower shop on the corner sells fresh roses   | the bread is too sweet       |             0 | 0.024142 |
| l'eau pure du ruisseau de montagne est rafraîc... | the pure water in the mountain stream is refre... | i look forward to the garden |             0 | 0.024762 |

```text
13. Exact Match Examples
No exact match examples were found for Problem 3B.

14. Full Validation Translation Results
```

| Unnamed: 0   | French Input                                      | Target English                                    | Predicted English                 | Exact Match   | BLEU-4   |
|:-------------|:--------------------------------------------------|:--------------------------------------------------|:----------------------------------|:--------------|:---------|
| 0            | elle a gagné un match de tennis                   | she won a tennis match                            | she wears a traditional           | 0             | 0.074410 |
| 1            | le marché biologique ouvre à l'aube le samedi     | the organic market opens at dawn on saturdays     | the sun is too sweet              | 0             | 0.029487 |
| 2            | nous regardons un film ensemble                   | we watch a movie together                         | we are planning a beautiful       | 0             | 0.063894 |
| 3            | le pain de cette boulangerie est toujours crou... | the bread at this bakery is always crunchy        | the bread is too is very          | 0             | 0.068460 |
| 4            | nous dansons au mariage                           | we dance at the wedding                           | we are going to buy               | 0             | 0.053728 |
| ...          | ...                                               | ...                                               | ...                               | ...           | ...      |
| 106          | ils jouent au football chaque week-end            | they play soccer every weekend                    | they feed in the park             | 0             | 0.053728 |
| 107          | je veux une tasse de café chaud                   | i want a hot cup of coffee                        | i want a new pair of              | 0             | 0.183787 |
| 108          | le livre est sur la table                         | the book is on the table                          | the bread is too sweet            | 0             | 0.052312 |
| 109          | il a terminé son projet scientifique en avance... | he finished his science project ahead of schedule | he drives a beautiful garden      | 0             | 0.029487 |
| 110          | nous devons acheter une douzaine d' ufs frais     | we need to buy a dozen fresh eggs                 | we are planning a big in the park | 0             | 0.033032 |

```text
15. Compact Final Table for Report Table
```

| Problem    | Direction         | Architecture                       |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:-----------|:------------------|:-----------------------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 3B | French-to-English | GRU Encoder-Decoder with Attention |           24 |                4.38403 |                       2.59587 |                   1.78816 |                     0.39375 |                  0.502899 |             0.109149 |                           0.375868 |                      0 |         0.063423 |                 436543 |

```text
Problem 3 Baseline vs Attention Comparison
```

| Problem    | Model                       | Direction         |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:-----------|:----------------------------|:------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 3A | Baseline GRU                | French-to-English |           15 |                4.90306 |                       4.26664 |                   0.63642 |                     0.24875 |                  0.249924 |             0.001174 |                           0.242386 |                      0 |         0.031008 |                 289086 |
| Problem 3B | GRU with Bahdanau Attention | French-to-English |           24 |                4.38403 |                       2.59587 |                   1.78816 |                     0.39375 |                  0.502899 |             0.109149 |                           0.375868 |                      0 |         0.063423 |                 436543 |

```text
Final Four-Model Comparison
```

| Problem    | Model                       | Direction         |   Best Epoch |   Best Validation Loss |   Training Loss at Best Epoch |   Training-Validation Gap |   Validation Token Accuracy |   Training Token Accuracy |   Token Accuracy Gap |   Validation-First Selection Score |   Exact Match Accuracy |   Average BLEU-4 |   Trainable Parameters |
|:-----------|:----------------------------|:------------------|-------------:|-----------------------:|------------------------------:|--------------------------:|----------------------------:|--------------------------:|---------------------:|-----------------------------------:|-----------------------:|-----------------:|-----------------------:|
| Problem 1  | Baseline GRU                | English-to-French |           39 |                4.7477  |                       3.52012 |                   1.22758 |                    0.264601 |                  0.305182 |             0.040581 |                           0.252325 |                      0 |         0.031716 |                 298786 |
| Problem 2  | GRU with Bahdanau Attention | English-to-French |           24 |                4.52264 |                       2.71684 |                   1.80579 |                    0.362336 |                  0.446035 |             0.083699 |                           0.344278 |                      0 |         0.07046  |                 460643 |
| Problem 3A | Baseline GRU                | French-to-English |           15 |                4.90306 |                       4.26664 |                   0.63642 |                    0.24875  |                  0.249924 |             0.001174 |                           0.242386 |                      0 |         0.031008 |                 289086 |
| Problem 3B | GRU with Bahdanau Attention | French-to-English |           24 |                4.38403 |                       2.59587 |                   1.78816 |                    0.39375  |                  0.502899 |             0.109149 |                           0.375868 |                      0 |         0.063423 |                 436543 |

```text
Direction-Level Average Comparison
```

| Direction         |   Best Validation Loss |   Validation Token Accuracy |   Exact Match Accuracy |   Average BLEU-4 |
|:------------------|-----------------------:|----------------------------:|-----------------------:|-----------------:|
| English-to-French |                4.63517 |                    0.313468 |                      0 |         0.051088 |
| French-to-English |                4.64355 |                    0.32125  |                      0 |         0.047216 |

## Custom Sentence Tests

```python
CUSTOM_TRANSLATION_TEST_PAIRS = [
    {
        "English Source": "i am cold",
        "French Reference": "j'ai froid",
    },
    {
        "English Source": "he is a teacher",
        "French Reference": "il est professeur",
    },
    {
        "English Source": "she likes music",
        "French Reference": "elle aime la musique",
    },
    {
        "English Source": "we are going home",
        "French Reference": "nous rentrons à la maison",
    },
    {
        "English Source": "they are playing outside",
        "French Reference": "ils jouent dehors",
    },
    {
        "English Source": "the dog is sleeping",
        "French Reference": "le chien dort",
    },
    {
        "English Source": "i want a cup of coffee",
        "French Reference": "je veux une tasse de café",
    },
    {
        "English Source": "please close the door",
        "French Reference": "ferme la porte s'il te plaît",
    },
    {
        "English Source": "the book is on the table",
        "French Reference": "le livre est sur la table",
    },
    {
        "English Source": "tom is reading a newspaper",
        "French Reference": "tom lit un journal",
    },
]


def get_oov_tokens(sentence, vocab):
    return [token for token in tokenize(sentence) if token not in vocab.token_to_idx]


def translate_one_custom_sentence(experiment, source_sentence):
    config = experiment["config"]
    resource = experiment["resource"]
    model = experiment["model"]

    decode_kwargs = dict(
        model=model,
        source_sentence=source_sentence,
        src_vocab=resource["src_vocab"],
        tgt_vocab=resource["tgt_vocab"],
        max_len=COMMON_CONFIG["max_len"],
        device=DEVICE,
        decoding_strategy=config.get("decode_strategy", "greedy"),
        beam_width=config.get("beam_width", 1),
        length_penalty_alpha=config.get("length_penalty_alpha", 0.0),
        mask_special_tokens=config.get("mask_special_tokens", True),
        min_decode_len=config.get("min_decode_len", 1),
    )

    if config["architecture_type"] == "attention":
        pred_tokens, _, _ = translate_attention(**decode_kwargs)
    else:
        pred_tokens = translate_baseline(**decode_kwargs)

    return pred_tokens


def run_fixed_custom_sentence_tests(custom_pairs):
    required_experiments = ["p1_baseline", "p2_attention", "p3_baseline", "p3_attention"]
    missing_experiments = [key for key in required_experiments if key not in experiment_results]

    if missing_experiments:
        raise RuntimeError(
            "Run the Problem 1, Problem 2, and Problem 3 training cells before this cell. "
            f"Missing experiment_results keys: {missing_experiments}"
        )

    experiment_order = [
        ("Problem 1", "p1_baseline"),
        ("Problem 2", "p2_attention"),
        ("Problem 3A", "p3_baseline"),
        ("Problem 3B", "p3_attention"),
    ]

    records = []

    for pair_id, pair in enumerate(custom_pairs, start=1):
        for problem_label, experiment_key in experiment_order:
            experiment = experiment_results[experiment_key]
            config = experiment["config"]
            resource = experiment["resource"]

            if config["direction"] == "English-to-French":
                source_sentence = pair["English Source"]
                reference_sentence = pair["French Reference"]
            else:
                source_sentence = pair["French Reference"]
                reference_sentence = pair["English Source"]

            pred_tokens = translate_one_custom_sentence(
                experiment=experiment,
                source_sentence=source_sentence,
            )
            reference_tokens = tokenize(reference_sentence)
            source_oov_tokens = get_oov_tokens(source_sentence, resource["src_vocab"])

            records.append({
                "Test ID": pair_id,
                "Problem": problem_label,
                "Model": config["model_label"],
                "Direction": config["direction"],
                "Source Sentence": source_sentence,
                "Reference Translation": tokens_to_sentence(reference_tokens),
                "Model Prediction": tokens_to_sentence(pred_tokens),
                "Exact Match": compute_exact_match(pred_tokens, reference_tokens),
                "BLEU-4": compute_bleu4(pred_tokens, reference_tokens),
                "Source OOV Count": len(source_oov_tokens),
                "Source OOV Tokens": ", ".join(source_oov_tokens) if source_oov_tokens else "None",
            })

    return pd.DataFrame(records)


custom_sentence_results = run_fixed_custom_sentence_tests(CUSTOM_TRANSLATION_TEST_PAIRS)

print("Custom sentence predictions for Problem 1 through Problem 3")
display(custom_sentence_results)

custom_sentence_metrics = (
    custom_sentence_results
    .groupby(["Problem", "Model", "Direction"], as_index=False)
    .agg(
        Mean_BLEU_4=("BLEU-4", "mean"),
        Exact_Match_Rate=("Exact Match", "mean"),
        Mean_Source_OOV_Count=("Source OOV Count", "mean"),
    )
    .sort_values(["Direction", "Problem"])
)

print("Table across the custom sentences")
display(custom_sentence_metrics)
```

**Output**

```text
Custom sentence predictions for Problem 1 through Problem 3
```

|   Test ID | Problem    | Model                       | Direction         | Source Sentence              | Reference Translation        | Model Prediction                    |   Exact Match |   BLEU-4 |   Source OOV Count | Source OOV Tokens   |
|----------:|:-----------|:----------------------------|:------------------|:-----------------------------|:-----------------------------|:------------------------------------|--------------:|---------:|-------------------:|:--------------------|
|         1 | Problem 1  | Baseline GRU                | English-to-French | i am cold                    | j'ai froid                   | nous avons visité                   |             0 | 0        |                  0 |                     |
|         1 | Problem 2  | GRU with Bahdanau Attention | English-to-French | i am cold                    | j'ai froid                   | je lis                              |             0 | 0        |                  0 |                     |
|         1 | Problem 3A | Baseline GRU                | French-to-English | j'ai froid                   | i am cold                    | she is to                           |             0 | 0        |                  0 |                     |
|         1 | Problem 3B | GRU with Bahdanau Attention | French-to-English | j'ai froid                   | i am cold                    | he cleans a beautiful               |             0 | 0        |                  0 |                     |
|         2 | Problem 1  | Baseline GRU                | English-to-French | he is a teacher              | il est professeur            | nous avons visité                   |             0 | 0        |                  0 |                     |
|         2 | Problem 2  | GRU with Bahdanau Attention | English-to-French | he is a teacher              | il est professeur            | il écrit des articles               |             0 | 0.080343 |                  0 |                     |
|         2 | Problem 3A | Baseline GRU                | French-to-English | il est professeur            | he is a teacher              | we are to                           |             0 | 0        |                  0 |                     |
|         2 | Problem 3B | GRU with Bahdanau Attention | French-to-English | il est professeur            | he is a teacher              | he speaks the phone                 |             0 | 0.080343 |                  0 |                     |
|         3 | Problem 1  | Baseline GRU                | English-to-French | she likes music              | elle aime la musique         | ils parlent souvent                 |             0 | 0        |                  0 |                     |
|         3 | Problem 2  | GRU with Bahdanau Attention | English-to-French | she likes music              | elle aime la musique         | elle traduit des contrats           |             0 | 0.080343 |                  0 |                     |
|         3 | Problem 3A | Baseline GRU                | French-to-English | elle aime la musique         | she likes music              | she is to                           |             0 | 0.113622 |                  0 |                     |
|         3 | Problem 3B | GRU with Bahdanau Attention | French-to-English | elle aime la musique         | she likes music              | she paints at                       |             0 | 0.113622 |                  0 |                     |
|         4 | Problem 1  | Baseline GRU                | English-to-French | we are going home            | nous rentrons à la maison    | nous avons visité le parc           |             0 | 0.053728 |                  0 |                     |
|         4 | Problem 2  | GRU with Bahdanau Attention | English-to-French | we are going home            | nous rentrons à la maison    | nous avons célébré leur             |             0 | 0.062571 |                  0 |                     |
|         4 | Problem 3A | Baseline GRU                | French-to-English | nous rentrons à la maison    | we are going home            | we are to the                       |             0 | 0.169904 |                  1 | rentrons            |
|         4 | Problem 3B | GRU with Bahdanau Attention | French-to-English | nous rentrons à la maison    | we are going home            | we are going to buy                 |             0 | 0.265915 |                  1 | rentrons            |
|         5 | Problem 1  | Baseline GRU                | English-to-French | they are playing outside     | ils jouent dehors            | nous avons visité le parc           |             0 | 0        |                  1 | outside             |
|         5 | Problem 2  | GRU with Bahdanau Attention | English-to-French | they are playing outside     | ils jouent dehors            | ils parlent souvent à               |             0 | 0.080343 |                  1 | outside             |
|         5 | Problem 3A | Baseline GRU                | French-to-English | ils jouent dehors            | they are playing outside     | she is to                           |             0 | 0        |                  1 | dehors              |
|         5 | Problem 3B | GRU with Bahdanau Attention | French-to-English | ils jouent dehors            | they are playing outside     | they speak spanish                  |             0 | 0.081414 |                  1 | dehors              |
|         6 | Problem 1  | Baseline GRU                | English-to-French | the dog is sleeping          | le chien dort                | le pain est très                    |             0 | 0.080343 |                  0 |                     |
|         6 | Problem 2  | GRU with Bahdanau Attention | English-to-French | the dog is sleeping          | le chien dort                | le pain est très                    |             0 | 0.080343 |                  0 |                     |
|         6 | Problem 3A | Baseline GRU                | French-to-English | le chien dort                | the dog is sleeping          | she is to                           |             0 | 0.081414 |                  0 |                     |
|         6 | Problem 3B | GRU with Bahdanau Attention | French-to-English | le chien dort                | the dog is sleeping          | the sun is melting                  |             0 | 0.095544 |                  0 |                     |
|         7 | Problem 1  | Baseline GRU                | English-to-French | i want a cup of coffee       | je veux une tasse de café    | ils ont construit une grande maison |             0 | 0.040825 |                  0 |                     |
|         7 | Problem 2  | GRU with Bahdanau Attention | English-to-French | i want a cup of coffee       | je veux une tasse de café    | je veux une lettre                  |             0 | 0.241178 |                  0 |                     |
|         7 | Problem 3A | Baseline GRU                | French-to-English | je veux une tasse de café    | i want a cup of coffee       | we are to to the                    |             0 | 0        |                  0 |                     |
|         7 | Problem 3B | GRU with Bahdanau Attention | French-to-English | je veux une tasse de café    | i want a cup of coffee       | i want a new pair of                |             0 | 0.217119 |                  0 |                     |
|         8 | Problem 1  | Baseline GRU                | English-to-French | please close the door        | ferme la porte s'il te plaît | nous avons visité le parc           |             0 | 0        |                  0 |                     |
|         8 | Problem 2  | GRU with Bahdanau Attention | English-to-French | please close the door        | ferme la porte s'il te plaît | ils nourrissent les pigeons         |             0 | 0        |                  0 |                     |
|         8 | Problem 3A | Baseline GRU                | French-to-English | ferme la porte s'il te plaît | please close the door        | we are a new                        |             0 | 0        |                  2 | ferme, te           |
|         8 | Problem 3B | GRU with Bahdanau Attention | French-to-English | ferme la porte s'il te plaît | please close the door        | i look forward to the park          |             0 | 0.040825 |                  2 | ferme, te           |
|         9 | Problem 1  | Baseline GRU                | English-to-French | the book is on the table     | le livre est sur la table    | nous avons visité le parc           |             0 | 0.043989 |                  1 | book                |
|         9 | Problem 2  | GRU with Bahdanau Attention | English-to-French | the book is on the table     | le livre est sur la table    | le pain est très                    |             0 | 0.057951 |                  1 | book                |
|         9 | Problem 3A | Baseline GRU                | French-to-English | le livre est sur la table    | the book is on the table     | we are to the                       |             0 | 0.04873  |                  1 | livre               |
|         9 | Problem 3B | GRU with Bahdanau Attention | French-to-English | le livre est sur la table    | the book is on the table     | the bread is too sweet              |             0 | 0.052312 |                  1 | livre               |
|        10 | Problem 1  | Baseline GRU                | English-to-French | tom is reading a newspaper   | tom lit un journal           | nous avons visité le parc           |             0 | 0        |                  1 | tom                 |
|        10 | Problem 2  | GRU with Bahdanau Attention | English-to-French | tom is reading a newspaper   | tom lit un journal           | ils ont construit une belle         |             0 | 0        |                  1 | tom                 |
|        10 | Problem 3A | Baseline GRU                | French-to-English | tom lit un journal           | tom is reading a newspaper   | she is a                            |             0 | 0.069373 |                  1 | tom                 |
|        10 | Problem 3B | GRU with Bahdanau Attention | French-to-English | tom lit un journal           | tom is reading a newspaper   | i want a beautiful                  |             0 | 0.062571 |                  1 | tom                 |

```text
Table across the custom sentences
```

| Problem    | Model                       | Direction         |   Mean_BLEU_4 |   Exact_Match_Rate |   Mean_Source_OOV_Count |
|:-----------|:----------------------------|:------------------|--------------:|-------------------:|------------------------:|
| Problem 1  | Baseline GRU                | English-to-French |      0.021889 |                  0 |                     0.3 |
| Problem 2  | GRU with Bahdanau Attention | English-to-French |      0.068307 |                  0 |                     0.3 |
| Problem 3A | Baseline GRU                | French-to-English |      0.048304 |                  0 |                     0.6 |
| Problem 3B | GRU with Bahdanau Attention | French-to-English |      0.100966 |                  0 |                     0.6 |
