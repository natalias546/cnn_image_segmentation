import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

import config
from model import UNet
from utils import mask_to_rgb, visualize_prediction, VOC_CLASSES


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, num_classes=config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(image_path, checkpoint_path, output_dir="outputs", show=False):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model(checkpoint_path, device)

    image = Image.open(image_path).convert("RGB")
    orig_size = image.size

    resized = image.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT), Image.BILINEAR)
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu()

    pred_rgb = Image.fromarray(mask_to_rgb(pred.numpy()))
    pred_rgb = pred_rgb.resize(orig_size, Image.NEAREST)

    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{name}_pred.png")
    pred_rgb.save(out_path)
    print(f"Saved: {out_path}")

    unique = pred.unique().tolist()
    detected = [VOC_CLASSES[c] for c in unique if c < len(VOC_CLASSES)]
    print(f"Detected classes: {', '.join(detected)}")

    if show:
        visualize_prediction(tensor.squeeze(0).cpu(), pred, pred)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <image_path> <checkpoint_path> [output_dir]")
        sys.exit(1)
    out = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    predict(sys.argv[1], sys.argv[2], output_dir=out)
