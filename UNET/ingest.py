import os
import glob

DATASET = "gopalbhattrai/pascal-voc-2012-dataset"
DOWNLOAD_DIR = "data"


def _find_voc_root(base):
    for dirpath, dirnames, _ in os.walk(base):
        if "JPEGImages" in dirnames:
            return dirpath
    return None


def download():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    existing = _find_voc_root(DOWNLOAD_DIR)
    if existing:
        print(f"Dataset already present at: {existing}\n")
        return existing

    try:
        import kaggle
    except ImportError:
        raise SystemExit("kaggle package missing, run: pip install kaggle")

    kaggle.api.authenticate()

    print(f"Downloading {DATASET} ...")
    kaggle.api.dataset_download_files(
        DATASET, path=DOWNLOAD_DIR, unzip=True, quiet=False
    )

    voc_root = _find_voc_root(DOWNLOAD_DIR)
    if voc_root is None:
        raise FileNotFoundError(
            f"JPEGImages/ not found under '{DOWNLOAD_DIR}/'. "
            "Check that the zip extracted correctly."
        )

    n_img = len(glob.glob(os.path.join(voc_root, "JPEGImages", "*.jpg")))
    n_mask = len(glob.glob(os.path.join(voc_root, "SegmentationClass", "*.png")))
    print(f"Dataset ready at: {voc_root}  ({n_img} images, {n_mask} masks)\n")
    return voc_root


if __name__ == "__main__":
    download()
