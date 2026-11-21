"""Fixed constants for the UATD project (see CLAUDE.md, Sections 3-4)."""

from __future__ import annotations

# Class order defines the class IDs 0-9. Never reorder.
CLASS_NAMES: tuple[str, ...] = (
    "ball",
    "circle cage",
    "cube",
    "cylinder",
    "human body",
    "metal bucket",
    "plane",
    "rov",
    "square cage",
    "tyre",
)
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Raw folders as distributed on figshare / Kaggle.
TRAINING_FOLDER = "UATD_Training"
TEST1_FOLDER = "UATD_Test_1"
TEST2_FOLDER = "UATD_Test_2"
RAW_FOLDERS: tuple[str, ...] = (TRAINING_FOLDER, TEST1_FOLDER, TEST2_FOLDER)

# Our four splits and the raw folder each one is drawn from.
SPLITS: tuple[str, ...] = ("train", "val", "test1", "test2")
SPLIT_SOURCE: dict[str, str] = {
    "train": TRAINING_FOLDER,
    "val": TRAINING_FOLDER,
    "test1": TEST1_FOLDER,
    "test2": TEST2_FOLDER,
}

# Integer prefix for COCO image IDs (pycocotools needs integer IDs).
# image_id = FOLDER_CODE[folder] * 100_000 + int(stem); unique across all splits.
FOLDER_CODE: dict[str, int] = {TRAINING_FOLDER: 1, TEST1_FOLDER: 2, TEST2_FOLDER: 3}

# Split algorithm parameters (from uatd-yolo.ipynb). Never change.
SEED = 42
VAL_FRACTION = 0.10

# File discovery (from uatd-yolo.ipynb).
IMAGE_EXTS = frozenset({".bmp", ".png", ".jpg", ".jpeg"})
IMAGE_DIR_NAMES = frozenset({"image", "images", "img", "jpegimages"})
ANNOTATION_DIR_NAMES = frozenset({"annotation", "annotations", "xml", "labels"})

# figshare article 21331143 (Xie et al., Scientific Data 2022).
FIGSHARE_URLS: dict[str, str] = {
    TRAINING_FOLDER: "https://ndownloader.figshare.com/files/37883163",
    TEST1_FOLDER: "https://ndownloader.figshare.com/files/37857276",
    TEST2_FOLDER: "https://ndownloader.figshare.com/files/37857279",
}
