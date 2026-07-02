"""
result:




"""
import re
import random
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from itertools import chain
from functools import partial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

np.random.seed(2)
random.seed(2)
torch.manual_seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load data
def parse_linear_file(filepath):
    """Returns list of (doc_id, sentence) tuples."""
    with open(filepath) as f:
        content = f.read()
    docs = re.split(r'<doc date=.*?>', content)[1:]
    result = []
    for doc_id, doc_text in enumerate(docs):
        sentences = re.split(r'(?<=[.!?])\s+', doc_text.strip())
        for s in sentences:
            s = s.strip()
            if s:
                result.append((doc_id, s))
    return result

# tokens
def tokenizer(text):
    text = text.lower()
    return re.findall(r"\b\w+\b", text)

class TextDataset(Dataset):
    def __init__(self, samples, vocab, tokenizer):
        self.texts = [t for t, l in samples]
        self.labels = [l for t, l in samples]
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        ids = [self.vocab.get(tok, self.vocab["<UNK>"]) for tok in tokens]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch, pad_idx):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    labels = torch.stack(labels)
    return padded, labels, lengths

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, pad_idx=0, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, padded_seqs, lengths):
        embedded = self.embedding(padded_seqs)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        out = self.dropout(h_n[-1])
        return self.fc(out).squeeze(1)

# train
if __name__ = '__main__':
  
  files = {
      "alzheimer": "./data/class1/linear.txt",
      "control":   "./data/class2/linear.txt"
  }
  
  all_samples = []  # (doc_id, sentence, label)
  for label_name, filepath in files.items():
      label = 1 if label_name == "alzheimer" else 0
      for doc_id, sentence in parse_linear_file(filepath):
          all_samples.append((f"{label_name}_{doc_id}", sentence, label))
  
  print(f"Total sentences: {len(all_samples)}")
  
  # preprocess
  sent_to_labels = {}
  for doc_id, sent, label in all_samples:
      sent_to_labels.setdefault(sent, set()).add(label)
  
  ambiguous = {s for s, labels in sent_to_labels.items() if len(labels) > 1}
  all_samples = [(d, s, l) for d, s, l in all_samples if s not in ambiguous]
  print(f"After removing cross-class duplicates: {len(all_samples)}")
  
  all_samples = [(d, s, l) for d, s, l in all_samples if len(tokenizer(s)) > 0]
  print(f"After removing empty-token sentences: {len(all_samples)}")
  
  # split data
  doc_ids = [d for d, s, l in all_samples]
  texts   = [s for d, s, l in all_samples]
  labels  = [l for d, s, l in all_samples]
  
  gss1 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
  train_idx, temp_idx = next(gss1.split(texts, labels, groups=doc_ids))
  
  temp_doc_ids = [doc_ids[i] for i in temp_idx]
  temp_texts   = [texts[i] for i in temp_idx]
  temp_labels  = [labels[i] for i in temp_idx]
  
  gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
  val_idx, test_idx = next(gss2.split(temp_texts, temp_labels, groups=temp_doc_ids))
  
  data = {
      "train": [(texts[i], labels[i]) for i in train_idx],
      "val":   [(temp_texts[i], temp_labels[i]) for i in val_idx],
      "test":  [(temp_texts[i], temp_labels[i]) for i in test_idx],
  }
  
  for split in data:
      n_pos = sum(l for _, l in data[split])
      print(f"{split}: {len(data[split])} sentences, {n_pos} positive ({n_pos/len(data[split]):.2%})")
  
  # sanity check: no doc_id shared across splits
  train_docs = {doc_ids[i] for i in train_idx}
  val_docs   = {temp_doc_ids[i] for i in val_idx}
  test_docs  = {temp_doc_ids[i] for i in test_idx}
  assert not (train_docs & val_docs) and not (train_docs & test_docs) and not (val_docs & test_docs)
  print("No document leakage across splits: confirmed")
  
  # build vocabs
  train_tokens = [tokenizer(t) for t, l in data["train"]]
  counter = Counter(chain.from_iterable(train_tokens))
  
  vocab = {"<PAD>": 0, "<UNK>": 1}
  for tok, freq in counter.most_common():
      vocab[tok] = len(vocab)
  
  print(f"Vocab size: {len(vocab)}")
  
  # Dataset/loader
  train_dataset = TextDataset(data["train"], vocab, tokenizer)
  val_dataset   = TextDataset(data["val"], vocab, tokenizer)
  test_dataset  = TextDataset(data["test"], vocab, tokenizer)
  
  collate = partial(collate_fn, pad_idx=vocab["<PAD>"])
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)
  val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate)
  test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)
  
  # model architecture
  model = RNNClassifier(vocab_size=len(vocab), pad_idx=vocab["<PAD>"]).to(device)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val_loss = float("inf")
    patience, patience_counter = 5, 0
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for padded, lab, lengths in train_loader:
            padded, lab = padded.to(device), lab.to(device)
            optimizer.zero_grad()
            logits = model(padded, lengths)
            loss = criterion(logits, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for padded, lab, lengths in val_loader:
                padded, lab = padded.to(device), lab.to(device)
                logits = model(padded, lengths)
                val_loss += criterion(logits, lab).item()
        val_loss /= len(val_loader)
    
        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for padded, lab, lengths in test_loader:
            padded, lab = padded.to(device), lab.to(device)
            logits = model(padded, lengths)
            probs = torch.sigmoid(logits)
            all_preds.extend((probs > 0.5).float().cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    print(f"Accuracy:  {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_labels, all_preds):.4f}")
    print(f"F1:        {f1_score(all_labels, all_preds):.4f}")
    print(f"AUC:       {roc_auc_score(all_labels, all_probs):.4f}")
