from pathlib import Path
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from ai.evaluation.segmentation_dataset import (
    MVTecSegmentationDataset,
)
from ai.models.defect_segmenter import UNet


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "ai/models/defect_segmenter_unet.pt"
)

THRESHOLD_PATH = Path(
    "ai/models/segmentation_threshold.txt"
)

SPLIT_DIRECTORY = Path(
    "ai/evaluation/segmentation_splits"
)

OUTPUT_PATH = Path(
    "ai/models/size_score_boundaries.csv"
)

BATCH_SIZE = 8


# ============================================================
# Device
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# Load CSV samples
# ============================================================

def load_samples(csv_path):

    samples = []

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            samples.append(
                (
                    row["image_path"],
                    row["mask_path"],
                )
            )

    return samples


# ============================================================
# Get category
# ============================================================

def get_category(image_path):

    path = Path(image_path)

    return path.parent.parent.parent.name


# ============================================================
# Load threshold
# ============================================================

with open(
    THRESHOLD_PATH,
    "r",
) as file:

    threshold = float(
        file.read().strip()
    )

print(
    f"Frozen segmentation threshold: {threshold:.4f}"
)


# ============================================================
# Load validation samples
# ============================================================

validation_samples = load_samples(
    SPLIT_DIRECTORY / "validation.csv"
)

print(
    f"Validation samples: {len(validation_samples)}"
)


# ============================================================
# Dataset / DataLoader
# ============================================================

validation_dataset = MVTecSegmentationDataset(
    validation_samples,
    image_size=224,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# Load model
# ============================================================

model = UNet(
    in_channels=3,
    out_channels=1,
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

print(
    f"Loaded model: {MODEL_PATH}"
)


# ============================================================
# Storage
# ============================================================

category_predicted_areas = {}


for image_path, mask_path in validation_samples:

    category = get_category(
        image_path
    )

    if category not in category_predicted_areas:

        category_predicted_areas[category] = []


# ============================================================
# Run validation inference
# ============================================================

print()
print("Evaluating validation images...")
print()


sample_index = 0


with torch.no_grad():

    for images, masks in validation_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= threshold
        ).float()

        predictions_np = (
            predictions
            .cpu()
            .numpy()
        )

        for prediction in predictions_np:

            prediction = prediction[0]

            # -----------------------------------------------
            # Predicted defect area percentage
            # -----------------------------------------------

            predicted_area = (
                np.sum(prediction > 0)
                / prediction.size
                * 100.0
            )

            # -----------------------------------------------
            # Category
            # -----------------------------------------------

            image_path = validation_samples[
                sample_index
            ][0]

            category = get_category(
                image_path
            )

            category_predicted_areas[
                category
            ].append(
                predicted_area
            )

            sample_index += 1


# ============================================================
# Calculate predicted-area boundaries
# ============================================================

boundaries = {}


for category in sorted(
    category_predicted_areas.keys()
):

    areas = np.asarray(
        category_predicted_areas[category],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Percentiles are based on the MODEL'S predictions.
    # --------------------------------------------------------

    p10 = np.percentile(
        areas,
        10,
    )

    p25 = np.percentile(
        areas,
        25,
    )

    p50 = np.percentile(
        areas,
        50,
    )

    p75 = np.percentile(
        areas,
        75,
    )

    p90 = np.percentile(
        areas,
        90,
    )

    p95 = np.percentile(
        areas,
        95,
    )

    boundaries[category] = {
        "p10": p10,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
    }


# ============================================================
# Save boundaries
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "category",
            "p10",
            "p25",
            "p50",
            "p75",
            "p90",
            "p95",
        ]
    )

    for category in sorted(
        boundaries.keys()
    ):

        values = boundaries[
            category
        ]

        writer.writerow(
            [
                category,
                f"{values['p10']:.6f}",
                f"{values['p25']:.6f}",
                f"{values['p50']:.6f}",
                f"{values['p75']:.6f}",
                f"{values['p90']:.6f}",
                f"{values['p95']:.6f}",
            ]
        )


# ============================================================
# Print results
# ============================================================

print()
print(
    "SIZE SCORE CALIBRATION"
)
print(
    "======================"
)
print()

print(
    "Boundaries are based on predicted validation areas."
)

print()

print(
    f"{'Category':<15}"
    f"{'':>12}"
    f"{'P25':>12}"
    f"{'P50':>12}"
    f"{'P75':>12}"
    f"{'P90':>12}"
    f"{'P95':>12}"
)

for category in sorted(
    boundaries.keys()
):

    values = boundaries[
        category
    ]

    print(
        f"{category:<15}"
        f"{values['p10']:>12.4f}"
        f"{values['p25']:>12.4f}"
        f"{values['p50']:>12.4f}"
        f"{values['p75']:>12.4f}"
        f"{values['p90']:>12.4f}"
        f"{values['p95']:>12.4f}"
    )

print()
print(
    f"Saved boundaries to: {OUTPUT_PATH}"
)

print()
print(
    "Size Score calibration complete."
)