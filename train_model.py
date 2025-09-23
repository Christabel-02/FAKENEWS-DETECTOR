# train_model.py
# Trains both: Text Model (BERT) + Image Model (CNN)

import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ======================
# Configurations
# ======================
MODEL_NAME = "bert-base-uncased"
TEXT_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
MAX_LEN = 128

SAVE_TEXT_DIR = "models/text_model"
SAVE_IMAGE_PATH = "models/image_model.pth"
os.makedirs("models", exist_ok=True)

# ======================
# TEXT DATASET
# ======================
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# ======================
# TEXT TRAINING FUNCTIONS
# ======================
def train_text_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training Text"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(dataloader), acc

def eval_text_epoch(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    eval_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Text"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return eval_loss / len(dataloader), acc, classification_report(all_labels, all_preds)

# ======================
# IMAGE TRAINING FUNCTIONS
# ======================
def train_image_model(data_dir, save_path, device, epochs=2, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Fake/Real

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Image Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Image Train Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Image Val Acc: {val_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Image model saved at {save_path}")

# ======================
# MAIN SCRIPT
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Train Text Model ---------
    df = pd.read_csv("data/news_dataset.csv")  # needs 'text' and 'label'
    texts = df["text"].values
    labels = df["label"].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=TEXT_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TEXT_BATCH_SIZE)

    text_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    text_model.to(device)

    optimizer = AdamW(text_model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    for epoch in range(EPOCHS):
        print(f"\nText Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_text_epoch(text_model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_report = eval_text_epoch(text_model, val_loader, device)

        print(f"Text Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Text Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(val_report)

    text_model.save_pretrained(SAVE_TEXT_DIR)
    tokenizer.save_pretrained(SAVE_TEXT_DIR)
    print(f"Text model saved at {SAVE_TEXT_DIR}")

    # --------- Train Image Model ---------
    # Expect folder structure:
    # data/images/train/fake, data/images/train/real
    # data/images/val/fake, data/images/val/real
    image_data_dir = "data/images"
    if os.path.exists(image_data_dir):
        train_image_model(image_data_dir, SAVE_IMAGE_PATH, device, epochs=EPOCHS, batch_size=IMAGE_BATCH_SIZE)
    else:
        print("Image dataset not found, skipping image model training.")

if __name__ == "__main__":
    main()
