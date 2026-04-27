# Condensed: Fine-Tune W2V2-Bert for low-resource ASR with 🤗 Transformers

Summary: This tutorial provides a complete implementation for fine-tuning Wav2Vec2-BERT (`facebook/w2v-bert-2.0`) on low-resource ASR using CTC, demonstrated on Mongolian Common Voice data. It covers building a character-level CTC tokenizer from dataset transcriptions, text normalization, vocabulary cleaning, audio preprocessing with `SeamlessM4TFeatureExtractor` (16kHz resampling), and a custom `DataCollatorCTCWithPadding` for dynamic batching. Key model configurations include `add_adapter=True`, disabled dropout, gradient checkpointing, and CTC-specific parameters. Training uses HF `Trainer` with WER evaluation. Scaling tips address CTC token duration (10–35ms), warmup ratios (5–15%), AdamW β₂ tuning, and low-frequency character removal from vocabularies.

*This is a condensed version that preserves essential implementation details and context.*

# Fine-Tune W2V2-Bert for Low-Resource ASR with 🤗 Transformers

## Key Concepts

**Wav2Vec2-BERT** is a 580M-parameter model pre-trained on **4.5M hours** of unlabeled audio in **143+ languages**. It uses **CTC (Connectionist Temporal Classification)** for ASR fine-tuning, predicting transcriptions in a single pass.

**Why W2V2-BERT over Whisper for low-resource languages:**
- Whisper fails on resource-poor languages (>100% WER for Mongolian)
- W2V2-BERT achieves **similar WER** to Whisper-large-v3 after fine-tuning
- **10-30x faster** inference, **2.5x more resource-efficient**
- Easily adaptable to any alphabet; requires little labeled data

## Setup

```bash
pip install datasets transformers torchaudio jiwer accelerate -U
```

```python
from huggingface_hub import notebook_login
notebook_login()
```

> **Best Practice:** Upload checkpoints directly to the 🤗 Hub during training for version control, Tensorboard logs, and model cards.

## Architecture Overview

W2V2-BERT requires two components:
- **Feature extractor** (`SeamlessM4TFeatureExtractor`): processes speech signal to model input format (shared with SeamlessM4T v1/v2)
- **Tokenizer** (`Wav2Vec2CTCTokenizer`): decodes predicted output classes to transcription text

## Creating the CTC Tokenizer

### Data Loading

Merge train+validation splits for small datasets; use test for validation:

```python
from datasets import load_dataset, Audio

common_voice_train = load_dataset("mozilla-foundation/common_voice_16_0", "mn", split="train+validation", use_auth_token=True)
common_voice_test = load_dataset("mozilla-foundation/common_voice_16_0", "mn", split="test", use_auth_token=True)
```

Remove unnecessary columns to keep only audio and transcription:

```python
common_voice_train = common_voice_train.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
common_voice_test = common_voice_test.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"])
```

### Text Normalization

Remove special characters that don't correspond to acoustic sound units (punctuation hurts CTC without a language model):

```python
import re
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\"\%\'\"\�\'\»\«]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)
```

### Building Vocabulary from Characters

CTC classifies speech chunks into letters. Extract all distinct characters from train and test sets:

```python
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
```

> **Important:** Use `batched=True, batch_size=-1` to process all transcriptions at once for vocabulary extraction.

### Cleaning Redundant Characters

**Best Practice:** CTC benefits from reduced vocabulary size — remove redundant characters (e.g., Latin characters when targeting Mongolian Cyrillic):

```python
def remove_latin_characters(batch):
    batch["sentence"] = re.sub(r'[a-z]+', '', batch["sentence"])
    return batch
```

> **Tip:** Dataset cleaning is iterative. Inspect the vocabulary carefully and consult native speakers when possible to identify characters that should be removed.

### Finalizing the Vocabulary

After cleaning, the vocabulary contains only Mongolian Cyrillic characters plus space. Key special tokens to add:

```python
# Replace space with visible delimiter token
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

# Add unknown token and CTC blank/padding token
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
# Final vocabulary size: 37 tokens → output dimension of the linear CTC layer
```

> **Critical:** The `[PAD]` token serves as CTC's "blank token", a core component of the CTC algorithm. The `" "` (space) token must be kept so the model learns word boundaries.

### Creating and Saving the Tokenizer

```python
import json
with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

from transformers import Wav2Vec2CTCTokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

repo_name = "w2v-bert-2.0-mongolian-colab-CV16.0"
tokenizer.push_to_hub(repo_name)
```

### Feature Extractor and Processor

The `SeamlessM4TFeatureExtractor` converts raw audio to log-mel spectrograms. Unlike the tokenizer, it doesn't need to be learned from data — load directly from the pretrained checkpoint:

```python
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

# Wrap tokenizer + feature extractor into a single processor
processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
processor.push_to_hub(repo_name)
```

> **Best Practice:** Upload the tokenizer and processor to the 🤗 Hub for reuse with the fine-tuned model.

## Preprocessing Data

### Audio Resampling

**Critical:** The sampling rate of fine-tuning data must match the pre-training rate. W2V2-BERT expects **16kHz** input.

```python
from datasets import Audio

common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
```

> **Warning:** Mismatched sampling rates will severely degrade performance — the same signal at different rates has very different distributions.

### Processing Pipeline

The processor handles: (1) Log-Mel feature extraction from audio, (2) encoding transcriptions to label IDs.

```python
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["input_length"] = len(batch["input_features"])
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)
```

## Training Setup

Four key components needed:

1. **Data collator** — Dynamic padding (pad to longest sample in batch, not overall longest) is essential due to W2V2-BERT's large input-to-output length ratio
2. **Evaluation metric** — Word Error Rate (WER)
3. **Pre-trained checkpoint** — Load and configure for training
4. **Training configuration**

### Data Collator

Custom collator pads inputs and labels separately (different modalities/lengths). Labels are padded with `-100` to ignore in loss computation:

```python
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
```

### Evaluation Metric (WER)

```python
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # Do NOT group tokens for labels — otherwise "hello" becomes "helo"
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
```

### Model Configuration

**Key decisions:** All dropout disabled (small trainable subset, not prone to overfitting), gradient checkpointing for GPU memory, `add_adapter=True`, CTC loss reduction set to "mean".

```python
from transformers import Wav2Vec2BertForCTC

model = Wav2Vec2BertForCTC.from_pretrained(
    "facebook/w2v-bert-2.0",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.0,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    add_adapter=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)
```

> **Warning:** These hyperparameters are tuned for this specific dataset. Adapt them for other languages/datasets.

### Training Arguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=repo_name,
    group_by_length=True,          # groups similar-length samples → reduces padding waste
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=10,
    gradient_checkpointing=True,
    fp16=True,
    save_steps=600,
    eval_steps=300,
    logging_steps=300,
    learning_rate=5e-5,
    warmup_steps=500,
    save_total_limit=2,
    push_to_hub=True,
)
```

### Trainer Initialization

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    tokenizer=processor.feature_extractor,
)
```

> **CTC Notes:** The blank token (`[PAD]`) allows predicting repeated characters (e.g., "hello" → `[PAD] "h" "e" "e" "l" "l" [PAD] "l" "o" "o" [PAD]`). Consecutive identical tokens are grouped during decoding, but **not** when decoding labels (`group_tokens=False`).

### Training Results

```python
trainer.train()
```

| Step | Training Loss | Validation Loss | WER |
|:---:|:---:|:---:|:---:|
| 300 | 1.7127 | 0.6477 | 0.5179 |
| 600 | 0.3493 | 0.6158 | 0.4420 |
| 900 | 0.1805 | 0.5251 | 0.3673 |
| 1200 | 0.0754 | 0.5288 | 0.3240 |

Final WER of **32.4%** is comparable to Whisper-large-v3 fine-tuned on the same data (**33.3% WER**), demonstrating near state-of-the-art performance on low-resource languages.

```python
trainer.push_to_hub()
```

### Evaluation / Inference

```python
model = Wav2Vec2BertForCTC.from_pretrained(repo_name).to("cuda")
processor = Wav2Vec2BertProcessor.from_pretrained(repo_name)

sample = common_voice_test[0]
input_features = torch.tensor(sample["input_features"]).to("cuda").unsqueeze(0)

with torch.no_grad():
    logits = model(input_features).logits

pred_ids = torch.argmax(logits, dim=-1)[0]
print(processor.decode(pred_ids))
```

> **Tip:** Using a [language model](https://huggingface.co/blog/wav2vec2-with-ngram) for decoding would further improve performance.

## Scaling-Up Tips (from HF experts)


...(truncated)