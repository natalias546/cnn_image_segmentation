# UNet Image Segmentation — Pascal VOC 2012

Semantic segmentation on 21 classes using a lightweight UNet trained on Pascal VOC 2012.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Data is not included. Download via Kaggle API:

1. Create a free account at [kaggle.com](https://www.kaggle.com) and generate an API token from your profile settings
2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run:
```bash
python ingest.py
```

## Train

```bash
python train.py
```

## Run entire pipeline

```bash
python main.py
```

Best checkpoint saved to `checkpoints/best_model.pth`.

## Predict

```bash
python predict.py <image_path> checkpoints/best_model.pth
```

## Config

All hyperparameters (image size, batch size, epochs, lr) are in `config.py`.

## Used some Generative AI for questions and debugging 
