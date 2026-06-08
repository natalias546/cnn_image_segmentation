import os
import torch
from torch.utils.data import DataLoader

import config
import ingest
from train import train, get_device
from dataset import VOCSegmentationDataset
from model import UNet
from utils import compute_iou_per_class, VOC_CLASSES


def evaluate(checkpoint_path):
    device = get_device()

    val_ds = VOCSegmentationDataset(
        config.DATA_ROOT, split="val",
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    model = UNet(in_channels=3, num_classes=config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    intersection = torch.zeros(config.NUM_CLASSES)
    union = torch.zeros(config.NUM_CLASSES)

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images).argmax(dim=1)

            valid = masks != config.IGNORE_INDEX
            p = preds[valid].cpu()
            t = masks[valid].cpu()

            for cls in range(config.NUM_CLASSES):
                pred_cls = p == cls
                true_cls = t == cls
                intersection[cls] += (pred_cls & true_cls).sum()
                union[cls] += (pred_cls | true_cls).sum()

    ious = []
    for cls in range(config.NUM_CLASSES):
        u = union[cls].item()
        ious.append(intersection[cls].item() / u if u > 0 else float("nan"))

    return ious


def print_report(ious, best_epoch):
    import numpy as np

    valid = [v for v in ious if not np.isnan(v)]
    mean = float(np.mean(valid)) if valid else 0.0

    col_w = 16
    bar_max = 30

    print("Per-class IoU  (best checkpoint)")
    print(f"{'Class':<{col_w}}  {'IoU':>6}  Bar")

    for name, iou in zip(VOC_CLASSES, ious):
        if not isinstance(iou, float) or iou != iou:
            bar = "n/a"
            iou_str = "  n/a"
        else:
            bar_len = int(iou * bar_max)
            bar = "#" * bar_len + "-" * (bar_max - bar_len)
            iou_str = f"{iou}"
        print(f"{name:<{col_w}}  {iou_str:>6}  {bar}")

    print(f"{'mean IoU':<{col_w}}  {mean}")
    print(f"{'best epoch':<{col_w}}  {best_epoch}")


def main():
    voc_root = ingest.download()
    config.DATA_ROOT = voc_root

    best_checkpoint = train()

    ious = evaluate(best_checkpoint)
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    best_epoch = checkpoint.get("epoch", -1)
    print_report(ious, best_epoch)


if __name__ == "__main__":
    main()
