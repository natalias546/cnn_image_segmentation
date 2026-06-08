import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import VOCSegmentationDataset
from model import UNet
from utils import mean_iou


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_miou = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader, desc="train" if train else "val", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            total_miou += mean_iou(preds, masks, config.NUM_CLASSES, config.IGNORE_INDEX)

    n = len(loader)
    return total_loss / n, total_miou / n


def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    train_ds = VOCSegmentationDataset(
        config.DATA_ROOT, split="train",
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        augment=True,
    )
    val_ds = VOCSegmentationDataset(
        config.DATA_ROOT, split="val",
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=pin)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model = UNet(in_channels=3, num_classes=config.NUM_CLASSES,
                 features=(32, 64, 128, 256)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    best_miou = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_miou = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_miou = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  mIoU: {train_miou:.4f} | "
            f"Val Loss: {val_loss:.4f}  mIoU: {val_miou:.4f}"
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_miou": val_miou,
                },
                os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
            )
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")

    torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "final_model.pth"))
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    print(f"\nTraining complete. Best val mIoU: {best_miou:.4f}")
    return best_path


if __name__ == "__main__":
    train()
