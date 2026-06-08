import numpy as np
import matplotlib.pyplot as plt
import torch

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128],
], dtype=np.uint8)


def compute_iou_per_class(preds, labels, num_classes, ignore_index=255):
    preds = preds.view(-1)
    labels = labels.view(-1)
    valid = labels != ignore_index
    preds, labels = preds[valid], labels[valid]

    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls
        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        ious.append(intersection / union if union > 0 else float("nan"))
    return ious


def mean_iou(preds, labels, num_classes, ignore_index=255):
    ious = compute_iou_per_class(preds, labels, num_classes, ignore_index)
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(VOC_COLORMAP)):
        rgb[mask == cls_idx] = VOC_COLORMAP[cls_idx]
    return rgb


def visualize_prediction(image, true_mask, pred_mask, save_path=None):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * std + mean, 0, 1)

    true_rgb = mask_to_rgb(true_mask.cpu().numpy())
    pred_rgb = mask_to_rgb(pred_mask.cpu().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(true_rgb)
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
