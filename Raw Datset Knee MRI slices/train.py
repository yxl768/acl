import argparse
import collections
import os
import random
import sys
import typing
from pathlib import Path
import numpy as np

typing.OrderedDict = collections.OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class ACLDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_transform=None, mask_transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        image_files = set(f.name for f in self.img_dir.iterdir() if f.is_file())
        mask_files = set(f.name for f in self.mask_dir.iterdir() if f.is_file())
        self.images = sorted(list(image_files.intersection(mask_files)))
        if len(self.images) == 0:
            raise FileNotFoundError(
                "没找到对应的图片和掩码文件，请检查路径"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.images[idx]
        mask_path = self.mask_dir / self.images[idx]

        # ===== screenshot start: 数据加载 =====
        # 数据加载
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        # ===== screenshot end: 数据加载 =====

        # ===== screenshot start: 数据增强 =====
        # 数据增强
        if self.image_transform:
            image = self.image_transform(image)
        # ===== screenshot end: 数据增强 =====

        # ===== screenshot start: 掩码预处理 =====
        # 掩码预处理
        if self.mask_transform:
            mask = self.mask_transform(mask)
        # ===== screenshot end: 掩码预处理 =====

        # ===== screenshot start: 二值化 =====
        # 二值化
        mask = (mask > 0.5).float()
        # ===== screenshot end: 二值化 =====
        return image, mask

# ===== screenshot start: 模型设计 =====
# 模型设计
# ===== screenshot start: 模型架构 =====
# 模型架构
# ===== screenshot end: 模型架构 =====
# ===== screenshot start: 输入设计 =====
# 输入设计
# ===== screenshot end: 输入设计 =====
# ===== screenshot start: 输出设计 =====
# 输出设计
# ===== screenshot end: 输出设计 =====
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class StandardUNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=16, dropout=0.1):
        super().__init__()
        # ===== screenshot start: 超参数设置 =====
        # 超参数设置
        # ===== screenshot end: 超参数设置 =====
        self.inc = ConvBlock(in_channels, base_filters, dropout)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters, base_filters * 2, dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters * 2, base_filters * 4, dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters * 4, base_filters * 8, dropout))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(base_filters * 8, base_filters * 16, dropout))

        self.up1 = UpBlock(base_filters * 16 + base_filters * 8, base_filters * 8, dropout)
        self.up2 = UpBlock(base_filters * 8 + base_filters * 4, base_filters * 4, dropout)
        self.up3 = UpBlock(base_filters * 4 + base_filters * 2, base_filters * 2, dropout)
        self.up4 = UpBlock(base_filters * 2 + base_filters, base_filters, dropout)

        self.out_conv = nn.Conv2d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.out_conv(x))

def get_model():
    return StandardUNet()

# ===== screenshot end: 模型设计 =====

def parse_args():
    parser = argparse.ArgumentParser(description="ACL MRI 单通道分割训练脚本")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument("--img_dir", type=str, default=str(base_dir / "images"))
    parser.add_argument("--mask_dir", type=str, default=str(base_dir / "masks"))
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    return parser.parse_args()

def fix_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(image_size: int, augment: bool = False):
    # 数据预处理
    train_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    eval_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
    image_transform = transforms.Compose(train_transforms if augment else eval_transforms)

    mask_transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
    ])
    return image_transform, mask_transform

# ===== screenshot end: 数据预处理 =====

# ===== screenshot start: 评价指标体系选型逻辑 =====
# 评价指标体系选型逻辑
# ===== screenshot end: 评价指标体系选型逻辑 =====

def get_binary_metrics(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    preds = (preds > 0.5).float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    tn = ((1 - preds) * (1 - targets)).sum(dim=(1, 2, 3))

    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    dice = ((2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)).mean().item()
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    f1 = (2.0 * precision * recall / (precision + recall + eps))
    accuracy = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean().item()
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "f1": f1,
        "accuracy": accuracy,
    }


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)
        dice_score = ((2.0 * intersection + self.eps) / (union + self.eps)).mean()
        return 1.0 - dice_score

# ===== screenshot start: 优化器设计 =====
# 优化器设计
# ===== screenshot end: 优化器设计 =====
# ===== screenshot start: 损失函数设计 =====
# 损失函数设计
class ComboLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.bce_weight * self.bce(preds, targets) + self.dice_weight * self.dice(preds, targets)
# ===== screenshot end: 损失函数设计 =====

# ===== screenshot start: 数据集划分 =====
# 数据集划分
# ===== screenshot end: 数据集划分 =====
def split_dataset(dataset: Dataset, val_split: float, test_split: float, seed: int):
    if val_split + test_split >= 1.0:
        raise ValueError("val_split 和 test_split 之和必须小于 1.0")

    total = len(dataset)
    test_size = max(1, int(total * test_split)) if test_split > 0 else 0
    val_size = max(1, int(total * val_split)) if val_split > 0 else 0
    train_size = total - val_size - test_size
    if train_size < 1:
        raise ValueError("训练集样本过少，请减小 val_split/test_split 或增加数据量")

    lengths = [train_size, val_size, test_size] if test_size > 0 else [train_size, val_size]
    splits = random_split(
        dataset,
        lengths,
        generator=torch.Generator().manual_seed(seed),
    )
    if test_size > 0:
        train_dataset, val_dataset, test_dataset = splits
    else:
        train_dataset, val_dataset = splits
        test_dataset = None
    return train_dataset, val_dataset, test_dataset

def limit_dataset_subset(dataset: Dataset, max_len: int):
    if len(dataset) <= max_len:
        return dataset
    if hasattr(dataset, "indices"):
        base_ds = dataset.dataset
        indices = dataset.indices[:max_len]
    else:
        base_ds = dataset
        indices = list(range(max_len))
    return Subset(base_ds, indices)


def limit_datasets(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
    train_dataset = limit_dataset_subset(train_dataset, 1500)
    val_dataset = limit_dataset_subset(val_dataset, 300)
    if test_dataset is not None:
        test_dataset = limit_dataset_subset(test_dataset, 200)
    return train_dataset, val_dataset, test_dataset

def plot_history(history: dict, save_path: str):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_loss"], label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_metric_curves(history: dict, save_path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(history["train_iou"], label="Train IoU", marker="o")
    plt.plot(history["val_iou"], label="Val IoU", marker="o")
    plt.plot(history["train_f1"], label="Train F1", marker="o")
    plt.plot(history["val_f1"], label="Val F1", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("IoU / F1 Score Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(confusion: np.ndarray, save_path: str):
    labels = ["Background", "ACL"]
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, int(confusion[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close()


def save_metrics_csv(history: dict, csv_path: str):
    import csv
    headers = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "train_precision",
        "train_recall",
        "train_f1",
        "train_accuracy",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_accuracy",
        "test_loss",
        "test_dice",
        "test_iou",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_accuracy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for epoch in range(len(history["train_loss"])):
            writer.writerow([
                epoch + 1,
                history["train_loss"][epoch],
                history.get("train_dice", [None])[epoch] if history.get("train_dice") else None,
                history.get("train_iou", [None])[epoch] if history.get("train_iou") else None,
                history.get("train_precision", [None])[epoch] if history.get("train_precision") else None,
                history.get("train_recall", [None])[epoch] if history.get("train_recall") else None,
                history.get("train_f1", [None])[epoch] if history.get("train_f1") else None,
                history.get("train_accuracy", [None])[epoch] if history.get("train_accuracy") else None,
                history.get("val_loss", [None])[epoch] if history.get("val_loss") else None,
                history.get("val_dice", [None])[epoch] if history.get("val_dice") else None,
                history.get("val_iou", [None])[epoch] if history.get("val_iou") else None,
                history.get("val_precision", [None])[epoch] if history.get("val_precision") else None,
                history.get("val_recall", [None])[epoch] if history.get("val_recall") else None,
                history.get("val_f1", [None])[epoch] if history.get("val_f1") else None,
                history.get("val_accuracy", [None])[epoch] if history.get("val_accuracy") else None,
                history.get("test_loss", [None])[epoch] if history.get("test_loss") else None,
                history.get("test_dice", [None])[epoch] if history.get("test_dice") else None,
                history.get("test_iou", [None])[epoch] if history.get("test_iou") else None,
                history.get("test_precision", [None])[epoch] if history.get("test_precision") else None,
                history.get("test_recall", [None])[epoch] if history.get("test_recall") else None,
                history.get("test_f1", [None])[epoch] if history.get("test_f1") else None,
                history.get("test_accuracy", [None])[epoch] if history.get("test_accuracy") else None,
            ])


def tensor_to_image(tensor: torch.Tensor):
    tensor = tensor.squeeze(0).cpu().numpy()
    tensor = (tensor * 0.5 + 0.5).clip(0, 1)
    return (tensor * 255).astype("uint8")


def save_sample_predictions(model, dataset: Dataset, device: torch.device, output_dir: str, num_samples: int = 4):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    with torch.no_grad():
        for idx_num, idx in enumerate(indices, start=1):
            image, mask = dataset[idx]
            input_image = image.unsqueeze(0).to(device)
            pred = model(input_image)
            pred = (pred > 0.5).float().cpu()

            image_np = tensor_to_image(image)
            mask_np = mask.squeeze(0).cpu().numpy().astype("uint8") * 255
            pred_np = pred.squeeze(0).squeeze(0).numpy().astype("uint8") * 255

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(image_np, cmap="gray")
            axes[0].set_title("Input")
            axes[0].axis("off")
            axes[1].imshow(mask_np, cmap="gray")
            axes[1].set_title("GT")
            axes[1].axis("off")
            axes[2].imshow(pred_np, cmap="gray")
            axes[2].set_title("Pred")
            axes[2].axis("off")

            axes[3].imshow(image_np, cmap="gray")
            axes[3].imshow(pred_np, cmap="Reds", alpha=0.4)
            axes[3].set_title("Overlay")
            axes[3].axis("off")

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"sample_{idx_num}.png"))
            plt.close(fig)


# ===== screenshot start: 训练过程 =====
# 训练过程
# ===== screenshot start: 训练设置 =====
# 训练设置
# ===== screenshot end: 训练设置 =====
# ===== screenshot start: 过程监控 =====
# 过程监控
# ===== screenshot end: 过程监控 =====
# ===== screenshot start: 早停机制 =====
# 早停机制
# ===== screenshot end: 早停机制 =====

# ===== screenshot start: 训练设置 =====
# 训练设置
# ===== screenshot end: 训练设置 =====

def train_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=None):
    model.train()
    total_loss = 0.0
    metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }
    data_iter = tqdm(
        loader,
        desc=f"Train Epoch {epoch}" if epoch is not None else "Train",
        dynamic_ncols=True,
        ascii=True,
        leave=False,
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for images, masks in data_iter:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_metrics = get_binary_metrics(outputs, masks)
        for key in metrics:
            metrics[key] += batch_metrics[key]
        data_iter.set_postfix_str(f"loss={loss.item():.4f}")

    count = len(loader)
    return total_loss / count, {k: metrics[k] / count for k in metrics}


def evaluate_epoch(model, loader, criterion, device, phase="Val"):
    model.eval()
    total_loss = 0.0
    metrics = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }
    confusion = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        data_iter = tqdm(
            loader,
            desc=phase,
            dynamic_ncols=True,
            ascii=True,
            leave=False,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        for images, masks in data_iter:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            batch_metrics = get_binary_metrics(outputs, masks)
            for key in metrics:
                metrics[key] += batch_metrics[key]
            preds = (outputs > 0.5).float()
            tp = ((preds * masks).sum()).item()
            fp = ((preds * (1 - masks)).sum()).item()
            fn = (((1 - preds) * masks).sum()).item()
            tn = (((1 - preds) * (1 - masks)).sum()).item()
            confusion += np.array([[tn, fp], [fn, tp]], dtype=int)
            data_iter.set_postfix_str(f"loss={loss.item():.4f}")
    count = len(loader)
    return total_loss / count, {k: metrics[k] / count for k in metrics}, confusion


def save_checkpoint(model, path: str):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: torch.device):
    model.load_state_dict(torch.load(path, map_location=device))

if __name__ == "__main__":
    args = parse_args()
    fix_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"使用设备: {device}")

    image_transform, mask_transform = get_transforms(args.image_size, augment=args.augment)
    dataset = ACLDataset(args.img_dir, args.mask_dir, image_transform, mask_transform)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, args.val_split, args.test_split, args.seed)
    train_dataset, val_dataset, test_dataset = limit_datasets(train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if test_dataset is not None
        else None
    )

    print(
        f"数据准备完成：训练={len(train_dataset)}，验证={len(val_dataset)}，测试={len(test_dataset) if test_dataset else 0}"
    )

    # ===== screenshot start: 模型搭建 =====
    # 模型搭建
    # ===== screenshot end: 模型搭建 =====
    # ===== screenshot start: 损失函数设计 =====
    # 损失函数设计
    # ===== screenshot end: 损失函数设计 =====
    model = StandardUNet().to(device)
    criterion = ComboLoss(0.5, 0.5)
    # ===== screenshot start: 优化器设计 =====
    # 优化器设计
    # ===== screenshot end: 优化器设计 =====
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ===== screenshot start: 混合精度训练 =====
    # 混合精度训练
    # ===== screenshot end: 混合精度训练 =====
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_accuracy": [],
    }
    best_val_loss = float("inf")
    best_model_path = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch=epoch)
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_precision"].append(train_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["train_accuracy"].append(train_metrics["accuracy"])

        val_loss, val_metrics, _ = evaluate_epoch(model, val_loader, criterion, device, phase="Val")
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | Val F1: {val_metrics['f1']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = args.save_path
            save_checkpoint(model, str(output_dir / best_model_path))
            patience_counter = 0
            print(f"  已保存最优模型：{output_dir / best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("早停触发，停止训练")
                break

    history_path = output_dir / "training_history.png"
    plot_history(history, str(history_path))
    plot_metric_curves(history, str(output_dir / "iou_f1_curves.png"))
    print(f"训练结束，训练曲线已保存到 {history_path}")
    print(f"IoU/F1 曲线已保存到 {output_dir / 'iou_f1_curves.png'}")

    test_metrics = None
    test_confusion = None
    test_loss = None
    if test_loader is not None:
        best_model = StandardUNet().to(device)
        load_checkpoint(best_model, str(output_dir / best_model_path), device)
        test_loss, test_metrics, test_confusion = evaluate_epoch(best_model, test_loader, criterion, device, phase="Test")
        print(
            f"测试集评估：Test Loss={test_loss:.4f} | Test Dice={test_metrics['dice']:.4f} | "
            f"Test IoU={test_metrics['iou']:.4f} | Test F1={test_metrics['f1']:.4f}"
        )
        plot_confusion_matrix(test_confusion, str(output_dir / "confusion_matrix.png"))
        print(f"混淆矩阵已保存到 {output_dir / 'confusion_matrix.png'}")

    if test_loader is not None:
        history["test_loss"] = [None] * len(history["train_loss"])
        history["test_dice"] = [None] * len(history["train_loss"])
        history["test_iou"] = [None] * len(history["train_loss"])
        history["test_precision"] = [None] * len(history["train_loss"])
        history["test_recall"] = [None] * len(history["train_loss"])
        history["test_f1"] = [None] * len(history["train_loss"])
        history["test_accuracy"] = [None] * len(history["train_loss"])
        history["test_loss"][-1] = test_loss
        history["test_dice"][-1] = test_metrics["dice"]
        history["test_iou"][-1] = test_metrics["iou"]
        history["test_precision"][-1] = test_metrics["precision"]
        history["test_recall"][-1] = test_metrics["recall"]
        history["test_f1"][-1] = test_metrics["f1"]
        history["test_accuracy"][-1] = test_metrics["accuracy"]

    save_metrics_csv(history, str(output_dir / "training_metrics.csv"))
    print(f"训练指标已保存到 {output_dir / 'training_metrics.csv'}")

    # ===== screenshot start: 预测结果可视化对比 =====
    # 预测结果可视化对比
    # ===== screenshot end: 预测结果可视化对比 =====

    save_sample_predictions(
        model=best_model if test_loader is not None else model,
        dataset=test_dataset if test_loader is not None else val_dataset,
        device=device,
        output_dir=str(output_dir / "sample_predictions"),
        num_samples=min(args.batch_size, 4),
    )
    print(f"预测样本图已保存到 {output_dir / 'sample_predictions'}")
# ===== screenshot end: 训练过程 =====

