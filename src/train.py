import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from dataset import AffectNetDataset  
from model import EmotionEfficientNetB4 

# -------------------------------
# PARAMETERS
# -------------------------------
csv_file = "../Data/labels.csv"  
img_dir = "../Data/Train"        
batch_size = 24
num_epochs = 15
lr = 0.0001
weight_decay = 1e-4
num_classes = 7  # contempt retiré
dropout_prob = 0.5
img_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patience_earlystop = 3

# -------------------------------
# TRANSFORMS
# -------------------------------
normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    normalize
])

val_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize
])

# -------------------------------
# MIXUP FUNCTIONS
# -------------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------------
# MAIN TRAINING
# -------------------------------
if __name__ == "__main__":
    # -------------------------------
    # STRATIFIED SPLIT
    # -------------------------------
    df = pd.read_csv(csv_file)
    df = df[df["label"] != "contempt"]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    train_dataset = AffectNetDataset(df=train_df, img_dir=img_dir, transform=train_tf)
    val_dataset   = AffectNetDataset(df=val_df, img_dir=img_dir, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------
    # MODEL, LOSS, OPTIMIZER, SCHEDULER
    # -------------------------------
    model = EmotionEfficientNetB4(num_classes=num_classes, dropout_prob=dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    scaler = torch.cuda.amp.GradScaler()
    best_val_acc = 0
    epochs_no_improve = 0

    # -------------------------------
    # TRAINING LOOP
    # -------------------------------
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            images, labels = images.to(device), labels.to(device)

            # MixUp
            images, targets_a, targets_b, lam = mixup_data(images, labels)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # -------------------------------
        # VALIDATION
        # -------------------------------
        model.eval()
        val_loss = 0
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}% | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        # -------------------------------
        # EarlyStopping
        # -------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "efficientnetb4_emotion.pth")
            print("💾 Best model saved!")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_earlystop:
                print("⏹ Early stopping triggered!")
                break

    print("✅ Training finished. Best Val Acc:", best_val_acc)
