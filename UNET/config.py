import os

# Path to the extracted Kaggle dataset root (the folder containing JPEGImages, SegmentationClass, etc.)
DATA_ROOT = "data/VOCdevkit/VOC2012"

IMAGE_DIR = os.path.join(DATA_ROOT, "JPEGImages")
MASK_DIR = os.path.join(DATA_ROOT, "SegmentationClass")
SPLIT_DIR = os.path.join(DATA_ROOT, "ImageSets", "Segmentation")

NUM_CLASSES = 21
IGNORE_INDEX = 255

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 8
NUM_WORKERS = 0       
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5

CHECKPOINT_DIR = "checkpoints"
