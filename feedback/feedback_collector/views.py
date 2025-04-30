from django.shortcuts import render, redirect
from django.http import HttpResponse
import os
import pickle
from torch.utils.data import DataLoader
import pandas as pd
import torch
from torch import nn, optim
from torch.nn.functional import softmax
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64
from nltk.tokenize import word_tokenize

from .data_preprocessing import SentimentDataset, load_and_preprocess_data, build_vocab
from .lstmmodel import SentimentLSTM

EMBEDDING_DIM = 100
MAX_LEN = 100
CONFIDENCE_THRESHOLD = 0.7
EPOCHS = 23
BATCH_SIZE = 1
LEARNING_RATE = 0.01
MODEL_PATH = 'sentiment_model.pth'
VOCAB_PATH = 'vocab.pkl'
ORIG_CORPUS = 'feedback_collector/hello.csv'
FEEDBACK_CSV = 'feedback_data.csv'


def plot_metrics(loss_before, acc_before, loss_after=None, acc_after=None):
    epochs_range = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss_before, label='Before Feedback')
    if loss_after:
        plt.plot(epochs_range, loss_after, label='After Feedback')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc_before, label='Before Feedback')
    if acc_after:
        plt.plot(epochs_range, acc_after, label='After Feedback')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64


def train_model(model, train_loader, device, vocab, epochs=EPOCHS):
    acc_per_epoch = []
    loss_per_epoch = [] 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    low_conf_samples = []
    Index = 0
    rev_vocab = {idx: token for token, idx in vocab.items()}
    pad_id = vocab.get('<pad>')

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if confs[0].item() < CONFIDENCE_THRESHOLD:
                token_ids = texts[0].cpu().tolist()
                words = [rev_vocab.get(tok, '<unk>') for tok in token_ids if tok != pad_id]
                sentence = ' '.join(words)
                low_conf_samples.append({
                    'index': Index+1,
                    'sentence': sentence,
                    'current_label': labels[0].item(),
                    'predicted': preds[0].item(),
                    'confidence': round(confs[0].item(), 2)
                })
                Index += 1

        acc = correct / total if total > 0 else 0
        acc_per_epoch.append(acc)
        loss_per_epoch.append(total_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")
    return low_conf_samples, loss_per_epoch, acc_per_epoch


def init_model_and_vocab(device):
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        model = SentimentLSTM(len(vocab), EMBEDDING_DIM, 64, 3).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        df = load_and_preprocess_data(ORIG_CORPUS)
        dataset = SentimentDataset(df['sentence'].values, df['sentiment'].values, vocab, MAX_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        low_conf, loss, acc = train_model(model, loader, device, vocab)
    else:
        df = load_and_preprocess_data(ORIG_CORPUS)
        vocab = build_vocab(df['sentence'].values)
        model = SentimentLSTM(len(vocab), EMBEDDING_DIM, 64, 3).to(device)
        dataset = SentimentDataset(df['sentence'].values, df['sentiment'].values, vocab, MAX_LEN)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        low_conf, loss, acc = train_model(model, loader, device, vocab)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab, f)
    return model, vocab, low_conf, loss, acc


def save_feedback(data, filename=FEEDBACK_CSV):
    df = pd.DataFrame(data)
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=header, index=False)


def retrain_with_feedback(device, vocab):
    orig = load_and_preprocess_data(ORIG_CORPUS)
    fb = pd.read_csv(FEEDBACK_CSV)
    df = pd.concat([orig, fb[['sentence', 'sentiment']]], ignore_index=True)
    dataset = SentimentDataset(df['sentence'].values, df['sentiment'].values, vocab, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SentimentLSTM(len(vocab), EMBEDDING_DIM, 64, 3).to(device)
    _, loss_after, acc_after = train_model(model, loader, device, vocab)
    torch.save(model.state_dict(), MODEL_PATH)
    return loss_after, acc_after


def main(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, low_conf, loss_before, acc_before = init_model_and_vocab(device)
    plot_data = None

    if request.method == 'POST':
        corrected = []
        for key, value in request.POST.items():
            if key.startswith('label_'):
                idx = int(key.split('_')[1])
                if 0 <= idx < len(low_conf):
                    corrected.append({
                        'sentence': low_conf[idx]['sentence'],
                        'sentiment': int(value)
                    })

        if corrected:
            save_feedback(corrected)
            loss_after, acc_after = retrain_with_feedback(device, vocab)
            plot_data = plot_metrics(loss_before, acc_before, loss_after, acc_after)
            return render(request, 'feedback_collecter_form.html', {
                'samples': [],
                'plot_data': plot_data,
                'message': 'Model retrained successfully with your feedback.'
            })

    plot_data = plot_metrics(loss_before, acc_before)
    return render(request, 'feedback_collecter_form.html', {
        'samples': low_conf,
        'plot_data': plot_data
    })
