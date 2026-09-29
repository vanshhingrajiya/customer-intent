import pandas as pd
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

# 1. Load data using pandas directly from the local CSVs
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Determine columns
text_col = 'text' if 'text' in train_df.columns else train_df.columns[0]
label_col = 'category' if 'category' in train_df.columns else ('label' if 'label' in train_df.columns else train_df.columns[1])

# 2. Encode string labels into numerical IDs
le = LabelEncoder()
train_df['label_id'] = le.fit_transform(train_df[label_col])
test_df['label_id'] = le.transform(test_df[label_col])

label_names = le.classes_.tolist()
num_labels = len(label_names)

print(f"Found {num_labels} unique intent categories.")

# 3. Load Tokenizer
print("Loading tokenizer...")
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Create a custom PyTorch Dataset
class BankingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        # Tokenize all texts upfront
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BankingDataset(train_df[text_col].tolist(), train_df['label_id'].tolist())
test_dataset = BankingDataset(test_df[text_col].tolist(), test_df['label_id'].tolist())

# 5. Initialize Model
print("Initializing model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 6. Configure Trainer
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1, # Set to 1 epoch for local testing speed
    weight_decay=0.01,
    eval_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 7. Train and Evaluate
print('Training...')
trainer.train()

print('Evaluating...')
eval_results = trainer.evaluate()

# 8. Save artifacts needed for the UI
print('Saving weights to .pth, metrics, and labels...')
with open('metrics.json', 'w') as f:
    json.dump(eval_results, f)

torch.save(model.state_dict(), 'banking77_distilbert.pth')

with open('labels.json', 'w') as f:
    json.dump(label_names, f)

print('Done! Files banking77_distilbert.pth, metrics.json, and labels.json have been generated.')
