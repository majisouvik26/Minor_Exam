import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset
import wandb
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from huggingface_hub import HfApi, hf_hub_download

CLASS_NAMES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class STL10Dataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_model(num_classes=10):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
    return total_loss / len(loader), correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hf_repo", type=str, default="stl10-resnet18")
    parser.add_argument("--wandb_project", type=str, default="stl10-minor-exam")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and only run eval/plots from saved model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset("Chiranjeev007/STL-10_Subset", keep_in_memory=True)
    train_ds = STL10Dataset(dataset["train"], transform=get_train_transforms())
    test_ds = STL10Dataset(dataset["test"], transform=get_eval_transforms())
    val_ds = STL10Dataset(dataset["validation"], transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = create_model().to(device)
    best_val_acc = 0.0
    best_state_path = "best_stl10_resnet18.pt"

    if not args.skip_train:
        wandb.init(project=args.wandb_project, config=vars(args))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_state_path)
        wandb.finish()

    model.load_state_dict(torch.load(best_state_path, map_location=device))
    wandb.init(project=args.wandb_project, name="eval_and_plots", config=vars(args))

    api = HfApi()
    repo_id = "souvikmaji22/" + args.hf_repo
    try:
        api.create_repo(repo_id, private=False, exist_ok=True)
    except Exception:
        pass
    api.upload_file(path_in_repo="pytorch_model.bin", path_or_fileobj=best_state_path, repo_id=repo_id, repo_type="model")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_images_raw = []
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(device)
            out = model(images_gpu)
            probs = torch.softmax(out, dim=1)
            _, preds = out.max(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            for i in range(images.size(0)):
                img = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = img.clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()
                all_images_raw.append(img)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = accuracy_score(all_labels, all_preds)
    wandb.log({"test_accuracy": test_acc})

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()

    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    for i in range(len(all_labels)):
        class_total[all_labels[i]] += 1
        if all_preds[i] == all_labels[i]:
            class_correct[all_labels[i]] += 1
    class_acc = class_correct / np.maximum(class_total, 1)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(CLASS_NAMES, class_acc, color="steelblue")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    wandb.log({"class_wise_accuracy": wandb.Image(fig2)})
    plt.close()

    correct_indices = [i for i in range(len(all_labels)) if all_preds[i] == all_labels[i]]
    incorrect_indices = [i for i in range(len(all_labels)) if all_preds[i] != all_labels[i]]
    selected_correct = correct_indices[:10]
    selected_incorrect = incorrect_indices[:10]
    selected_indices = selected_correct + selected_incorrect

    table_data = []
    for idx in selected_indices:
        img = (all_images_raw[idx] * 255).astype(np.uint8)
        table_data.append([
            wandb.Image(img),
            CLASS_NAMES[all_preds[idx]],
            CLASS_NAMES[all_labels[idx]]
        ])
    wandb_table = wandb.Table(columns=["image", "predicted", "actual"], data=table_data)
    wandb.log({"test_samples_20": wandb_table})

    model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", repo_type="model")
    loaded_state = torch.load(model_path, map_location=device)
    model_load = create_model().to(device)
    model_load.load_state_dict(loaded_state, strict=True)
    model_load.eval()
    eval_correct = 0
    eval_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model_load(images)
            _, preds = out.max(1)
            eval_total += labels.size(0)
            eval_correct += preds.eq(labels).sum().item()
    loaded_acc = eval_correct / eval_total
    wandb.log({"loaded_model_test_accuracy": loaded_acc})
    wandb.finish()

if __name__ == "__main__":
    main()
