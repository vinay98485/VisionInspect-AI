"""
Prepare and thoroughly validate YOLO-format object detection dataset from MVTec AD.

Extracts bounding boxes from binary ground-truth defect masks using
connected component analysis (cv2.connectedComponentsWithStats) and creates
a standardized YOLO dataset directory structure:

datasets/mvtec_yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml

Reuses existing project splits:
- Defective samples: ai/evaluation/segmentation_splits/{train.csv, validation.csv, test.csv}
- Normal samples: ai/evaluation/classification_splits/{train.csv, val.csv, test.csv} (is_defective == 0)
"""

from pathlib import Path
import csv
import os
import random
import shutil
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "mvtec_anomaly_detection"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "mvtec_yolo"
SEG_SPLITS_DIR = PROJECT_ROOT / "ai" / "evaluation" / "segmentation_splits"
CLS_SPLITS_DIR = PROJECT_ROOT / "ai" / "evaluation" / "classification_splits"

MIN_DEFECT_AREA = 16  # Minimum pixel area to filter single-pixel noise


def make_safe_stem(image_path: Path) -> str:
    """
    Generate a deterministic, collision-safe filename stem:
    e.g. bottle__test__broken_large__000
    """
    rel = image_path.relative_to(DATASET_ROOT)
    parts = rel.parts  # (category, split, defect_type, filename)
    return f"{parts[0]}__{parts[1]}__{parts[2]}__{image_path.stem}"


def extract_yolo_boxes_from_mask(mask_path: Path, min_area: int = MIN_DEFECT_AREA):
    """
    Load ground-truth mask and extract normalized YOLO bounding boxes:
    0 <x_center> <y_center> <width> <height>
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    img_h, img_w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    # label 0 is background
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]

        x_center = (x + w / 2.0) / float(img_w)
        y_center = (y + h / 2.0) / float(img_h)
        norm_w = w / float(img_w)
        norm_h = h / float(img_h)

        # Clamp normalized values between 0.0 and 1.0
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        boxes.append({
            "class_id": 0,
            "x_center": x_center,
            "y_center": y_center,
            "norm_w": norm_w,
            "norm_h": norm_h,
            "raw_bbox": (x, y, w, h),
            "area": area,
        })

    return boxes, (img_w, img_h)


def load_dataset_samples():
    """
    Load defective samples from segmentation splits and normal samples from classification splits.
    Returns dict mapping target split ('train', 'val', 'test') to list of sample dicts.
    """
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"MVTec dataset not found at: {DATASET_ROOT}")

    if not SEG_SPLITS_DIR.exists():
        raise FileNotFoundError(f"Segmentation splits directory not found at: {SEG_SPLITS_DIR}")

    if not CLS_SPLITS_DIR.exists():
        raise FileNotFoundError(f"Classification splits directory not found at: {CLS_SPLITS_DIR}")

    splits = {"train": [], "val": [], "test": []}

    # 1. Defective samples from segmentation splits
    seg_mapping = {
        "train": SEG_SPLITS_DIR / "train.csv",
        "val": SEG_SPLITS_DIR / "validation.csv",
        "test": SEG_SPLITS_DIR / "test.csv",
    }

    total_defective = 0
    for split_key, csv_file in seg_mapping.items():
        if not csv_file.exists():
            raise FileNotFoundError(f"Required split file missing: {csv_file}")
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_p = PROJECT_ROOT / row["image_path"] if not Path(row["image_path"]).is_absolute() else Path(row["image_path"])
                msk_p = PROJECT_ROOT / row["mask_path"] if not Path(row["mask_path"]).is_absolute() else Path(row["mask_path"])
                splits[split_key].append({
                    "category": row["category"],
                    "defect_type": row["defect_type"],
                    "image_path": img_p,
                    "mask_path": msk_p,
                    "is_defective": True,
                })
                total_defective += 1

    # 2. Normal samples from classification splits
    cls_mapping = {
        "train": CLS_SPLITS_DIR / "train.csv",
        "val": CLS_SPLITS_DIR / "val.csv",
        "test": CLS_SPLITS_DIR / "test.csv",
    }

    total_normal = 0
    for split_key, csv_file in cls_mapping.items():
        if not csv_file.exists():
            raise FileNotFoundError(f"Required split file missing: {csv_file}")
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["is_defective"]) == 0:
                    img_p = PROJECT_ROOT / row["image_path"] if not Path(row["image_path"]).is_absolute() else Path(row["image_path"])
                    splits[split_key].append({
                        "category": row["category"],
                        "defect_type": row["defect_type"],
                        "image_path": img_p,
                        "mask_path": None,
                        "is_defective": False,
                    })
                    total_normal += 1

    print(f"Loaded {total_defective} defective samples and {total_normal} normal samples.")
    print(f"Split distributions: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, total={total_defective + total_normal}")
    return splits


def create_data_yaml(output_dir: Path):
    """
    Create standard YOLO data.yaml configuration file.
    """
    yaml_content = (
        f"path: datasets/mvtec_yolo\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: defect\n"
    )
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Created data.yaml at: {yaml_path}")


def generate_dataset():
    """
    Process samples, copy images, extract boxes, write label files and data.yaml.
    """
    print("=" * 60)
    print("MVTec -> YOLO Object Detection Dataset Generation")
    print("=" * 60)

    splits = load_dataset_samples()

    # Create destination directories
    for split_name in ["train", "val", "test"]:
        (OUTPUT_ROOT / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats_summary = {
        "total_images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "defective_images": 0,
        "normal_images": 0,
        "total_boxes": 0,
        "multi_box_images": 0,
        "empty_normal_labels": 0,
        "boxes_per_split": {"train": 0, "val": 0, "test": 0},
        "box_counts": [],
        "sample_manifest": [],
    }

    seen_stems = set()

    for split_name in ["train", "val", "test"]:
        samples = splits[split_name]
        print(f"\nProcessing {split_name} split ({len(samples)} samples)...")

        for sample in samples:
            src_img = sample["image_path"]
            safe_stem = make_safe_stem(src_img)

            if safe_stem in seen_stems:
                raise ValueError(f"Collision detected for stem: {safe_stem}")
            seen_stems.add(safe_stem)

            dest_img = OUTPUT_ROOT / "images" / split_name / f"{safe_stem}.png"
            dest_lbl = OUTPUT_ROOT / "labels" / split_name / f"{safe_stem}.txt"

            # 1. Copy image (avoid overwriting existing files with read-only permissions)
            if not dest_img.exists():
                shutil.copyfile(src_img, dest_img)
                try:
                    os.chmod(dest_img, 0o644)
                except OSError:
                    pass

            # 2. Generate label
            if sample["is_defective"]:
                boxes, dims = extract_yolo_boxes_from_mask(sample["mask_path"])
                num_boxes = len(boxes)

                lines = [
                    f"{b['class_id']} {b['x_center']:.6f} {b['y_center']:.6f} {b['norm_w']:.6f} {b['norm_h']:.6f}"
                    for b in boxes
                ]
                dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

                stats_summary["defective_images"] += 1
                stats_summary["total_boxes"] += num_boxes
                stats_summary["boxes_per_split"][split_name] += num_boxes
                stats_summary["box_counts"].append(num_boxes)
                if num_boxes > 1:
                    stats_summary["multi_box_images"] += 1

                stats_summary["sample_manifest"].append({
                    "stem": safe_stem,
                    "split": split_name,
                    "is_defective": True,
                    "num_boxes": num_boxes,
                    "src_img": str(src_img),
                    "src_mask": str(sample["mask_path"]),
                    "category": sample["category"],
                    "defect_type": sample["defect_type"],
                })
            else:
                # Normal image -> Empty label file (0 bytes)
                dest_lbl.write_text("", encoding="utf-8")
                stats_summary["normal_images"] += 1
                stats_summary["empty_normal_labels"] += 1

                stats_summary["sample_manifest"].append({
                    "stem": safe_stem,
                    "split": split_name,
                    "is_defective": False,
                    "num_boxes": 0,
                    "src_img": str(src_img),
                    "src_mask": None,
                    "category": sample["category"],
                    "defect_type": sample["defect_type"],
                })

            stats_summary["total_images"] += 1
            if split_name == "train":
                stats_summary["train_images"] += 1
            elif split_name == "val":
                stats_summary["val_images"] += 1
            elif split_name == "test":
                stats_summary["test_images"] += 1

    create_data_yaml(OUTPUT_ROOT)

    print("\nDataset generation complete!")
    return stats_summary


def validate_dataset(stats_summary):
    """
    Perform thorough validation checks across generated files.
    """
    print("\n" + "=" * 60)
    print("RUNNING POST-GENERATION VALIDATION CHECKS")
    print("=" * 60)

    errors = []

    # A. Check file counts
    for split_name in ["train", "val", "test"]:
        img_dir = OUTPUT_ROOT / "images" / split_name
        lbl_dir = OUTPUT_ROOT / "labels" / split_name

        imgs = list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        print(f"[{split_name.upper()}] Images: {len(imgs)}, Labels: {len(lbls)}")
        if len(imgs) != len(lbls):
            errors.append(f"Count mismatch in {split_name}: {len(imgs)} images vs {len(lbls)} labels")

        # C. Check 1:1 image to label pairing
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        if img_stems != lbl_stems:
            missing = (img_stems ^ lbl_stems)
            errors.append(f"Mismatched stems in {split_name}: {len(missing)} mismatches, sample: {list(missing)[:5]}")

    # B. Check annotation format and validity
    total_valid_boxes = 0
    total_non_empty_labels = 0
    empty_labels = 0

    for lbl_path in (OUTPUT_ROOT / "labels").glob("*/*.txt"):
        content = lbl_path.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels += 1
            continue

        total_non_empty_labels += 1
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"Line {line_num} in {lbl_path.name} has {len(parts)} values (expected 5)")
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError as e:
                errors.append(f"Non-numeric values in {lbl_path.name}: {e}")
                continue

            if cls_id != 0:
                errors.append(f"Invalid class ID {cls_id} in {lbl_path.name} (expected 0)")

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                errors.append(f"Center out of bounds ({xc}, {yc}) in {lbl_path.name}")

            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                errors.append(f"Width/height out of bounds ({w}, {h}) in {lbl_path.name}")

            total_valid_boxes += 1

    print(f"Total verified valid bounding boxes: {total_valid_boxes}")
    print(f"Non-empty label files (defective): {total_non_empty_labels}")
    print(f"Empty label files (normal): {empty_labels}")

    # E. Bounding-box correctness checks on selected representative samples
    print("\nVerifying bounding-box reconstruction against source masks...")
    manifest = stats_summary["sample_manifest"]
    defective_manifest = [m for m in manifest if m["is_defective"]]

    # Pick samples: 1 component, multi component, small component, large defect
    single_comp = [m for m in defective_manifest if m["num_boxes"] == 1]
    multi_comp = [m for m in defective_manifest if m["num_boxes"] >= 4]
    max_comp = [m for m in defective_manifest if m["num_boxes"] == max(stats_summary["box_counts"])]

    sampled_cases = [
        ("Single Component", single_comp[0]),
        ("Single Component (alt)", single_comp[10]),
        ("Multi-Component (>=4)", multi_comp[0]),
        ("Maximum Components", max_comp[0]),
    ]

    for label, item in sampled_cases:
        lbl_file = OUTPUT_ROOT / "labels" / item["split"] / f"{item['stem']}.txt"
        lines = lbl_file.read_text(encoding="utf-8").strip().split("\n")

        mask = cv2.imread(item["src_mask"], cv2.IMREAD_GRAYSCALE)
        h_orig, w_orig = mask.shape[:2]

        print(f" - [{label}] {item['category']} ({item['defect_type']}) -> {len(lines)} boxes in {lbl_file.name}")
        for idx, l in enumerate(lines):
            cid, xc, yc, nw, nh = map(float, l.split())
            x1 = int(round((xc - nw / 2.0) * w_orig))
            y1 = int(round((yc - nh / 2.0) * h_orig))
            x2 = int(round((xc + nw / 2.0) * w_orig))
            y2 = int(round((yc + nh / 2.0) * h_orig))

            box_crop = mask[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
            defect_in_box = np.sum(box_crop > 0)
            if defect_in_box == 0:
                errors.append(f"Box {idx} in {item['stem']} contains 0 defect pixels!")

    # F. Verify normal images have 0 boxes
    print("\nVerifying normal images contain 0 bounding boxes...")
    normal_manifest = [m for m in manifest if not m["is_defective"]]
    for m in normal_manifest[:50]:
        lbl_file = OUTPUT_ROOT / "labels" / m["split"] / f"{m['stem']}.txt"
        if lbl_file.stat().st_size != 0:
            errors.append(f"Normal image label is not empty: {lbl_file}")

    # G. Source integrity check
    print("\nVerifying source MVTec directory integrity...")
    all_source_images = [p for p in DATASET_ROOT.glob("*/*/*/*.png") if "ground_truth" not in p.parts]
    all_source_masks = list(DATASET_ROOT.glob("*/ground_truth/*/*.png"))
    print(f"Original MVTec dataset untouched: {len(all_source_images)} images, {len(all_source_masks)} masks.")
    if len(all_source_images) != 5354 or len(all_source_masks) != 1258:
        errors.append(f"Source MVTec image count mismatch: expected 5354 images, found {len(all_source_images)}")

    if errors:
        print("\nERRORS ENCOUNTERED:")
        for err in errors[:10]:
            print(f" - {err}")
        raise RuntimeError(f"Validation failed with {len(errors)} errors!")
    else:
        print("\nALL VALIDATION CHECKS PASSED SUCCESSFULLY (0 errors, 0 warnings).")


def generate_visual_samples(stats_summary, num_samples: int = 4):
    """
    Generate temporary visual validation images demonstrating reconstructed bounding boxes.
    """
    vis_dir = OUTPUT_ROOT / "sample_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    manifest = stats_summary["sample_manifest"]
    defective_manifest = [m for m in manifest if m["is_defective"]]

    # Select representative examples from different categories
    selected = []
    seen_cats = set()
    for m in defective_manifest:
        if m["category"] not in seen_cats and m["num_boxes"] >= 1:
            selected.append(m)
            seen_cats.add(m["category"])
        if len(selected) >= num_samples:
            break

    # Add at least one multi-component sample
    multi = [m for m in defective_manifest if m["num_boxes"] >= 4 and m not in selected]
    if multi:
        selected.append(multi[0])

    print(f"\nGenerating {len(selected)} validation visualizations in: {vis_dir}")
    for item in selected:
        img_file = OUTPUT_ROOT / "images" / item["split"] / f"{item['stem']}.png"
        lbl_file = OUTPUT_ROOT / "labels" / item["split"] / f"{item['stem']}.txt"

        img = cv2.imread(str(img_file))
        h, w = img.shape[:2]

        lines = lbl_file.read_text(encoding="utf-8").strip().split("\n")
        vis = img.copy()

        for line in lines:
            if not line: continue
            cid, xc, yc, nw, nh = map(float, line.split())
            x1 = int(round((xc - nw / 2.0) * w))
            y1 = int(round((yc - nh / 2.0) * h))
            x2 = int(round((xc + nw / 2.0) * w))
            y2 = int(round((yc + nh / 2.0) * h))

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "defect", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out_path = vis_dir / f"vis_{item['stem']}.png"
        cv2.imwrite(str(out_path), vis)
        print(f" - Saved: {out_path.name} ({len(lines)} boxes)")


if __name__ == "__main__":
    summary = generate_dataset()
    validate_dataset(summary)
    generate_visual_samples(summary)
