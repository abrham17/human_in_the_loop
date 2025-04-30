import torch
import pickle
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import Counter

MAX_WORDS = 5000
MAX_LEN = 100



class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = word_tokenize(text.lower())
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens[:self.max_len]]
        padded = token_ids + [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        return torch.tensor(padded, dtype=torch.int64), torch.tensor(label, dtype=torch.int64)


def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    label_mapping = {'Neutral': 0, 'Positive': 1, 'Negative': 2}
    df['sentiment'] = df['sentiment'].map(label_mapping)
    df = df.dropna(subset=['sentiment'])
    df = df.reset_index(drop=True)
    df['sentence'] = df['sentence'].str.lower().str.replace('[^\w\s]', '', regex=True)
    return df

def build_vocab(texts, max_words=MAX_WORDS):
    all_tokens = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({token: idx + 2 for idx, (token, _) in enumerate(token_counts.most_common(len(token_counts)))})
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    return vocab
