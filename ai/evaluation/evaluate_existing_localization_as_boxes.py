"""
VisionInspect AI - Existing Localization as Bounding Boxes Evaluation
Evaluates whether the existing U-Net segmentation model (and Anomaly Detector)
can produce defect bounding boxes comparable to YOLO ground-truth boxes.

Validation Scope:
- 876 test images (260 defective, 616 normal)
- 403 ground-truth defect boxes from datasets/mvtec_yolo/labels/test
- One-to-one greedy bipartite matching (IoU >= 0.50 primary)
- No training, no model changes, no production pipeline modifications.
"""

import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.defect_segmenter import UNet
UNET_MODEL_PATH = PROJECT_ROOT / "ai/models/defect_segmenter_unet.pt"
UNET_THRESHOLD_PATH = PROJECT_ROOT / "ai/models/segmentation_threshold.txt"
UNET_POSTPROCESS_PATH = PROJECT_ROOT / "ai/models/segmentation_postprocessing.json"
ANOMALY_THRESHOLDS_PATH = PROJECT_ROOT / "ai/models/anomaly_thresholds.csv"
TEST_IMG_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/images/test"
TEST_LBL_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/labels/test"
OUTPUT_DIR = PROJECT_ROOT / "ai/weights/yolo/evaluation_existing_localization"

EVAL_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.80]
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

SIZE_BINS = [
    ("<0.1%", 0.0, 0.1),
    ("0.1–0.5%", 0.1, 0.5),
    ("0.5–1%", 0.5, 1.0),
    ("1–5%", 1.0, 5.0),
    (">5%", 5.0, 1000.0),
]


def box_iou(box1, box2):
    """
    Compute IoU between two normalized boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_test_manifest():
    """
    Parse test images and ground-truth bounding box labels.
    """
    images = sorted(list(TEST_IMG_DIR.glob("*.png")))
    manifest = []

    for img_path in images:
        stem = img_path.stem
        lbl_path = TEST_LBL_DIR / f"{stem}.txt"
        parts = stem.split("__")
        category = parts[0]
        source_split = parts[1]
        defect_type = parts[2]
        is_defective = (defect_type != "good")

        gt_boxes = []
        if lbl_path.exists():
            lines = [l.strip() for l in lbl_path.read_text().strip().split("\n") if l.strip()]
            for line in lines:
                cid, xc, yc, nw, nh = map(float, line.split())
                x1 = xc - nw / 2.0
                y1 = yc - nh / 2.0
                x2 = xc + nw / 2.0
                y2 = yc + nh / 2.0
                area_pct = nw * nh * 100.0
                gt_boxes.append({
                    "class_id": int(cid),
                    "box": [x1, y1, x2, y2],
                    "xc": xc,
                    "yc": yc,
                    "nw": nw,
                    "nh": nh,
                    "area_pct": area_pct,
                })

        manifest.append({
            "stem": stem,
            "img_path": img_path,
            "category": category,
            "defect_type": defect_type,
            "is_defective": is_defective,
            "gt_boxes": gt_boxes,
            "num_gt": len(gt_boxes),
        })

    return manifest


def match_predictions_greedy(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Perform one-to-one greedy bipartite matching on an image:
    - Sort predictions by confidence descending.
    - Match to highest-IoU unmatched GT with IoU >= iou_thresh.
    """
    matched_preds = set()
    matched_gts = set()
    pred_matches = {}

    sorted_p_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i]["conf"], reverse=True)

    for p_idx in sorted_p_indices:
        p = pred_boxes[p_idx]
        best_iou = 0.0
        best_g_idx = -1

        for g_idx, g in enumerate(gt_boxes):
            if g_idx in matched_gts:
                continue
            iou = box_iou(p["box_norm"], g["box"])
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx

        if best_iou >= iou_thresh and best_g_idx != -1:
            matched_preds.add(p_idx)
            matched_gts.add(best_g_idx)
            pred_matches[p_idx] = (best_g_idx, best_iou)

    return matched_preds, matched_gts, pred_matches


def extract_boxes_from_prob_map(prob_map, threshold, postprocess=True, min_area=25, morph_kernel=3):
    """
    Convert a 224x224 predicted probability map into normalized bounding boxes.
    Uses morphological opening and connected-component analysis matching the project's pipeline.
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    if postprocess and morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = binary

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)

    boxes = []
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        x = stats[lbl, cv2.CC_STAT_LEFT]
        y = stats[lbl, cv2.CC_STAT_TOP]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        h = stats[lbl, cv2.CC_STAT_HEIGHT]

        # Normalized coordinates in [0, 1] relative to 224x224
        x1 = max(0.0, min(1.0, x / 224.0))
        y1 = max(0.0, min(1.0, y / 224.0))
        x2 = max(0.0, min(1.0, (x + w) / 224.0))
        y2 = max(0.0, min(1.0, (y + h) / 224.0))

        # Confidence: average predicted probability inside the connected component
        mask_pixels = (labels == lbl)
        conf = float(prob_map[mask_pixels].mean()) if np.any(mask_pixels) else float(threshold)

        boxes.append({
            "box_norm": [x1, y1, x2, y2],
            "conf": conf,
            "area_px": area,
            "area_pct": (area / (224.0 * 224.0)) * 100.0,
        })

    return boxes


def evaluate_threshold(manifest, cached_probs, threshold, iou_thresh=0.5, postprocess=True, min_area=25, morph_kernel=3):
    """
    Evaluate all 876 test images at a specific U-Net probability threshold and IoU cutoff.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred = 0
    total_gt = sum(m["num_gt"] for m in manifest)

    image_level_eval = []

    for m, prob in zip(manifest, cached_probs):
        pred_boxes = extract_boxes_from_prob_map(
            prob,
            threshold=threshold,
            postprocess=postprocess,
            min_area=min_area,
            morph_kernel=morph_kernel
        )
        total_pred += len(pred_boxes)

        matched_p, matched_g, _ = match_predictions_greedy(m["gt_boxes"], pred_boxes, iou_thresh=iou_thresh)

        tp = len(matched_p)
        fp = len(pred_boxes) - tp
        fn = len(m["gt_boxes"]) - len(matched_g)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        image_level_eval.append({
            "stem": m["stem"],
            "category": m["category"],
            "defect_type": m["defect_type"],
            "is_defective": m["is_defective"],
            "num_gt": len(m["gt_boxes"]),
            "num_pred": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matched_gts": matched_g,
        })

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "threshold": threshold,
        "iou_thresh": iou_thresh,
        "postprocess": postprocess,
        "min_area": min_area,
        "total_predictions": total_pred,
        "total_gt": total_gt,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "image_level_eval": image_level_eval,
    }


def compute_size_breakdown(manifest, eval_res):
    """
    Calculate recall across the 5 standard GT defect size bins.
    """
    img_eval = eval_res["image_level_eval"]
    records = []

    for label, low, high in SIZE_BINS:
        gt_in_bin = 0
        tp_in_bin = 0
        for item in img_eval:
            stem = item["stem"]
            m = next(entry for entry in manifest if entry["stem"] == stem)
            for g_idx, g in enumerate(m["gt_boxes"]):
                if low <= g["area_pct"] < high:
                    gt_in_bin += 1
                    if g_idx in item["matched_gts"]:
                        tp_in_bin += 1
        fn_in_bin = gt_in_bin - tp_in_bin
        rec = (tp_in_bin / gt_in_bin * 100.0) if gt_in_bin > 0 else 0.0
        records.append({
            "size_bin": label,
            "gt_boxes": gt_in_bin,
            "TP": tp_in_bin,
            "FN": fn_in_bin,
            "recall_pct": rec,
        })
    return records


def compute_category_breakdown(manifest, eval_res):
    """
    Calculate TP, FP, FN, Precision, Recall across all 15 categories.
    """
    img_eval = eval_res["image_level_eval"]
    records = []

    for cat in CATEGORIES:
        cat_items = [it for it in img_eval if it["category"] == cat]
        gt_cnt = sum(it["num_gt"] for it in cat_items)
        tp_cnt = sum(it["tp"] for it in cat_items)
        fp_cnt = sum(it["fp"] for it in cat_items)
        fn_cnt = sum(it["fn"] for it in cat_items)

        prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
        rec = tp_cnt / gt_cnt if gt_cnt > 0 else 0.0

        records.append({
            "category": cat,
            "gt_boxes": gt_cnt,
            "TP": tp_cnt,
            "FP": fp_cnt,
            "FN": fn_cnt,
            "Precision": prec,
            "Recall": rec,
        })
    return records


def main():
    print("=" * 85)
    print("VisionInspect AI - Existing Localization as Bounding Boxes Evaluation")
    print("=" * 85)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load manifest and verify test set
    manifest = load_test_manifest()
    total_images = len(manifest)
    total_defective = sum(1 for m in manifest if m["is_defective"])
    total_normal = sum(1 for m in manifest if not m["is_defective"])
    total_gt_boxes = sum(m["num_gt"] for m in manifest)

    assert total_images == 876, f"Expected 876 test images, found {total_images}"
    assert total_defective == 260, f"Expected 260 defective test images, found {total_defective}"
    assert total_normal == 616, f"Expected 616 normal test images, found {total_normal}"
    assert total_gt_boxes == 403, f"Expected 403 GT defect boxes, found {total_gt_boxes}"

    # Load existing configured threshold and postprocessing
    existing_threshold = 0.65
    if UNET_THRESHOLD_PATH.exists():
        existing_threshold = float(UNET_THRESHOLD_PATH.read_text().strip())

    morph_kernel = 3
    min_component_area = 25
    if UNET_POSTPROCESS_PATH.exists():
        with open(UNET_POSTPROCESS_PATH, "r") as f:
            pp_cfg = json.load(f)
            morph_kernel = int(pp_cfg.get("morph_kernel_size", 3))
            min_component_area = int(pp_cfg.get("min_component_area", 25))

    print("Verified Test Set Scope:")
    print(f" - Test Set Images:     {total_images} (Defective: {total_defective}, Normal: {total_normal})")
    print(f" - Ground-Truth Boxes:  {total_gt_boxes}")
    print(f" - Existing Threshold:  {existing_threshold:.4f}")
    print(f" - Post-Processing:     morph_kernel={morph_kernel}, min_component_area={min_component_area}")
    print(f" - Weights Path:        {UNET_MODEL_PATH}")

    # 2. Run batched U-Net inference and cache probability maps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print(f"\nRunning U-Net inference on {total_images} images (batch_size=32 on {device})...")
    t0 = time.perf_counter()
    cached_probs = []
    batch_size = 32

    for i in range(0, total_images, batch_size):
        chunk = manifest[i:i + batch_size]
        batch_tensors = []
        for m in chunk:
            img = cv2.imread(str(m["img_path"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            batch_tensors.append(torch.from_numpy(img).permute(2, 0, 1))

        inp = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(inp)
            probs = torch.sigmoid(logits)[:, 0].cpu().numpy()

        for p in probs:
            cached_probs.append(p)

    inference_wall = time.perf_counter() - t0
    print(f"Inference complete in {inference_wall:.2f}s ({total_images / inference_wall:.1f} FPS, {inference_wall / total_images * 1000.0:.2f} ms/img).")

    # =========================================================================
    # TASK 3: U-NET THRESHOLD ANALYSIS (IoU >= 0.50)
    # =========================================================================
    print("\n" + "=" * 85)
    print("TASK 3: U-NET THRESHOLD SWEEP (IoU >= 0.50, morph=3, min_area=25)")
    print("=" * 85)

    threshold_results = []
    threshold_eval_map = {}

    for th in EVAL_THRESHOLDS:
        res = evaluate_threshold(
            manifest,
            cached_probs,
            threshold=th,
            iou_thresh=0.50,
            postprocess=True,
            min_area=min_component_area,
            morph_kernel=morph_kernel
        )
        threshold_eval_map[th] = res
        threshold_results.append({
            "Threshold": f"{th:.2f}",
            "Predictions": res["total_predictions"],
            "GT Boxes": res["total_gt"],
            "TP": res["TP"],
            "FP": res["FP"],
            "FN": res["FN"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
        })

    df_thresh = pd.DataFrame(threshold_results)
    print(df_thresh.to_string(index=False, formatters={
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    # Identify best threshold by F1
    best_th = max(EVAL_THRESHOLDS, key=lambda t: threshold_eval_map[t]["F1"])
    print(f"\nBest threshold by F1: {best_th:.2f} (F1: {threshold_eval_map[best_th]['F1']:.4f})")
    print(f"Existing configured threshold: {existing_threshold:.2f} (F1: {threshold_eval_map[existing_threshold]['F1']:.4f})")

    # =========================================================================
    # TASK 2: LOCALIZATION ANALYSIS AT IoU 0.25, 0.50, 0.75
    # =========================================================================
    print("\n" + "=" * 85)
    print("TASK 2: LOCALIZATION THRESHOLD COMPARISON (IoU 0.25 vs 0.50 vs 0.75)")
    print("=" * 85)

    loc_results = []
    for eval_th, label in [(existing_threshold, f"Existing (th={existing_threshold:.2f})"), (best_th, f"Best F1 (th={best_th:.2f})")]:
        for iou_th in [0.25, 0.50, 0.75]:
            res = evaluate_threshold(
                manifest,
                cached_probs,
                threshold=eval_th,
                iou_thresh=iou_th,
                postprocess=True,
                min_area=min_component_area,
                morph_kernel=morph_kernel
            )
            loc_results.append({
                "Configuration": label,
                "IoU Threshold": f">={iou_th:.2f}",
                "TP": res["TP"],
                "FP": res["FP"],
                "FN": res["FN"],
                "Precision": res["Precision"],
                "Recall": res["Recall"],
                "F1": res["F1"],
            })

    df_loc = pd.DataFrame(loc_results)
    print(df_loc.to_string(index=False, formatters={
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    # =========================================================================
    # TASK 4: DEFECT SIZE ANALYSIS
    # =========================================================================
    print("\n" + "=" * 85)
    print("TASK 4: DEFECT SIZE ANALYSIS BY GROUND-TRUTH AREA BIN (IoU >= 0.50)")
    print("=" * 85)

    size_breakdown_existing = compute_size_breakdown(manifest, threshold_eval_map[existing_threshold])
    size_breakdown_best = compute_size_breakdown(manifest, threshold_eval_map[best_th])

    size_comp_rows = []
    sb_exist_dict = {it["size_bin"]: it for it in size_breakdown_existing}
    sb_best_dict = {it["size_bin"]: it for it in size_breakdown_best}

    for bin_lbl, _, _ in SIZE_BINS:
        ex = sb_exist_dict[bin_lbl]
        bs = sb_best_dict[bin_lbl]
        size_comp_rows.append({
            "Size Bin": bin_lbl,
            "GT Boxes": ex["gt_boxes"],
            f"TP (th={existing_threshold:.2f})": ex["TP"],
            f"Recall (th={existing_threshold:.2f})": f"{ex['recall_pct']:.2f}%",
            f"TP (th={best_th:.2f})": bs["TP"],
            f"Recall (th={best_th:.2f})": f"{bs['recall_pct']:.2f}%",
        })

    df_size = pd.DataFrame(size_comp_rows)
    print(df_size.to_string(index=False))

    # =========================================================================
    # TASK 5: DIRECT YOLO vs U-NET COMPARISON
    # =========================================================================
    print("\n" + "=" * 85)
    print("TASK 5: DIRECT YOLO vs U-NET COMPARISON (IoU >= 0.50)")
    print("=" * 85)

    # YOLO baseline figures from prior validated experiments
    # YOLO baseline: conf=0.25, IoU=0.50, imgsz=640
    # YOLO max candidates: conf=0.01, IoU=0.50, imgsz=640
    comp_methods = [
        {
            "Method": "YOLO11n Baseline (conf=0.25, 640)",
            "Predictions": 282,
            "TP": 162,
            "FP": 120,
            "FN": 241,
            "Precision": 0.5745,
            "Recall": 0.4020,
            "F1": 0.4730,
        },
        {
            "Method": "YOLO11n Max-Recall (conf=0.01, 640)",
            "Predictions": 3307,
            "TP": 280,
            "FP": 3027,
            "FN": 123,
            "Precision": 0.0847,
            "Recall": 0.6948,
            "F1": 0.1509,
        },
        {
            "Method": f"U-Net Boxes Existing (th={existing_threshold:.2f})",
            "Predictions": threshold_eval_map[existing_threshold]["total_predictions"],
            "TP": threshold_eval_map[existing_threshold]["TP"],
            "FP": threshold_eval_map[existing_threshold]["FP"],
            "FN": threshold_eval_map[existing_threshold]["FN"],
            "Precision": threshold_eval_map[existing_threshold]["Precision"],
            "Recall": threshold_eval_map[existing_threshold]["Recall"],
            "F1": threshold_eval_map[existing_threshold]["F1"],
        },
        {
            "Method": f"U-Net Boxes Optimal F1 (th={best_th:.2f})",
            "Predictions": threshold_eval_map[best_th]["total_predictions"],
            "TP": threshold_eval_map[best_th]["TP"],
            "FP": threshold_eval_map[best_th]["FP"],
            "FN": threshold_eval_map[best_th]["FN"],
            "Precision": threshold_eval_map[best_th]["Precision"],
            "Recall": threshold_eval_map[best_th]["Recall"],
            "F1": threshold_eval_map[best_th]["F1"],
        },
    ]

    df_method_comp = pd.DataFrame(comp_methods)
    print(df_method_comp.to_string(index=False, formatters={
        "Precision": lambda x: f"{x * 100.0:.2f}%",
        "Recall": lambda x: f"{x * 100.0:.2f}%",
        "F1": lambda x: f"{x:.4f}",
    }))

    # Size-bin comparison between YOLO baseline, YOLO max, and U-Net
    size_comp_all = [
        {
            "Method": "YOLO11n Baseline (conf=0.25)",
            "<0.1%": "0.00% (0/43)",
            "0.1–0.5%": "36.56% (34/93)",
            "0.5–1%": "37.74% (20/53)",
            "1–5%": "57.38% (70/122)",
            ">5%": "41.30% (38/92)",
        },
        {
            "Method": "YOLO11n Max-Recall (conf=0.01)",
            "<0.1%": "16.28% (7/43)",
            "0.1–0.5%": "67.74% (63/93)",
            "0.5–1%": "67.92% (36/53)",
            "1–5%": "86.89% (106/122)",
            ">5%": "73.91% (68/92)",
        },
        {
            "Method": f"U-Net Existing (th={existing_threshold:.2f})",
            "<0.1%": f"{sb_exist_dict['<0.1%']['recall_pct']:.2f}% ({sb_exist_dict['<0.1%']['TP']}/{sb_exist_dict['<0.1%']['gt_boxes']})",
            "0.1–0.5%": f"{sb_exist_dict['0.1–0.5%']['recall_pct']:.2f}% ({sb_exist_dict['0.1–0.5%']['TP']}/{sb_exist_dict['0.1–0.5%']['gt_boxes']})",
            "0.5–1%": f"{sb_exist_dict['0.5–1%']['recall_pct']:.2f}% ({sb_exist_dict['0.5–1%']['TP']}/{sb_exist_dict['0.5–1%']['gt_boxes']})",
            "1–5%": f"{sb_exist_dict['1–5%']['recall_pct']:.2f}% ({sb_exist_dict['1–5%']['TP']}/{sb_exist_dict['1–5%']['gt_boxes']})",
            ">5%": f"{sb_exist_dict['>5%']['recall_pct']:.2f}% ({sb_exist_dict['>5%']['TP']}/{sb_exist_dict['>5%']['gt_boxes']})",
        },
        {
            "Method": f"U-Net Optimal F1 (th={best_th:.2f})",
            "<0.1%": f"{sb_best_dict['<0.1%']['recall_pct']:.2f}% ({sb_best_dict['<0.1%']['TP']}/{sb_best_dict['<0.1%']['gt_boxes']})",
            "0.1–0.5%": f"{sb_best_dict['0.1–0.5%']['recall_pct']:.2f}% ({sb_best_dict['0.1–0.5%']['TP']}/{sb_best_dict['0.1–0.5%']['gt_boxes']})",
            "0.5–1%": f"{sb_best_dict['0.5–1%']['recall_pct']:.2f}% ({sb_best_dict['0.5–1%']['TP']}/{sb_best_dict['0.5–1%']['gt_boxes']})",
            "1–5%": f"{sb_best_dict['1–5%']['recall_pct']:.2f}% ({sb_best_dict['1–5%']['TP']}/{sb_best_dict['1–5%']['gt_boxes']})",
            ">5%": f"{sb_best_dict['>5%']['recall_pct']:.2f}% ({sb_best_dict['>5%']['TP']}/{sb_best_dict['>5%']['gt_boxes']})",
        },
    ]
    df_size_comp_all = pd.DataFrame(size_comp_all)
    print("\nSize Bin Recall Comparison:")
    print(df_size_comp_all.to_string(index=False))

    # =========================================================================
    # PER-CATEGORY BREAKDOWN (IoU >= 0.50)
    # =========================================================================
    print("\n" + "=" * 85)
    print("PER-CATEGORY BOX-LEVEL RECALL (IoU >= 0.50)")
    print("=" * 85)

    cat_breakdown_exist = compute_category_breakdown(manifest, threshold_eval_map[existing_threshold])
    cat_breakdown_best = compute_category_breakdown(manifest, threshold_eval_map[best_th])

    cat_comp_rows = []
    cb_exist_dict = {it["category"]: it for it in cat_breakdown_exist}
    cb_best_dict = {it["category"]: it for it in cat_breakdown_best}

    for cat in CATEGORIES:
        ex = cb_exist_dict[cat]
        bs = cb_best_dict[cat]
        cat_comp_rows.append({
            "Category": cat,
            "GT": ex["gt_boxes"],
            f"TP ({existing_threshold:.2f})": ex["TP"],
            f"FP ({existing_threshold:.2f})": ex["FP"],
            f"Recall ({existing_threshold:.2f})": f"{ex['Recall'] * 100.0:.1f}%",
            f"TP ({best_th:.2f})": bs["TP"],
            f"FP ({best_th:.2f})": bs["FP"],
            f"Recall ({best_th:.2f})": f"{bs['Recall'] * 100.0:.1f}%",
        })

    df_cat = pd.DataFrame(cat_comp_rows)
    print(df_cat.to_string(index=False))

    # =========================================================================
    # SAVE JSON AND TXT REPORTS
    # =========================================================================
    json_path = OUTPUT_DIR / "localization_box_analysis.json"
    txt_path = OUTPUT_DIR / "localization_box_analysis.txt"

    json_payload = {
        "metadata": {
            "unet_model_path": str(UNET_MODEL_PATH),
            "test_images": total_images,
            "defective_images": total_defective,
            "normal_images": total_normal,
            "gt_defect_boxes": total_gt_boxes,
            "existing_threshold": existing_threshold,
            "best_threshold_by_f1": best_th,
            "morph_kernel_size": morph_kernel,
            "min_component_area": min_component_area,
            "device": str(device),
        },
        "threshold_sweep_iou50": threshold_results,
        "localization_iou_sweep": loc_results,
        "size_breakdown_comparison": size_comp_rows,
        "direct_method_comparison": comp_methods,
        "size_comparison_all_methods": size_comp_all,
        "category_breakdown": cat_comp_rows,
        "task6_anomaly_heatmap_audit": {
            "status": "NOT_FEASIBLE_WITHOUT_PRODUCTION_MODIFICATION",
            "findings": [
                "The production AnomalyDetector model operates at a coarse 14x14 patch resolution (stride 16) on ResNet18 Layer3 features.",
                "The values in the anomaly map represent L2 Euclidean feature distances to normal memory, ranging continuously from ~0.03 to ~3.0.",
                "The existing anomaly_thresholds.csv defines whole-image score thresholds based on the mean of the top-10% patches (19 of 196 patches). It contains no spatial, patch-level, or pixel-level segmentation thresholds.",
                "The AnomalyDetector class exposes no localization_mask or bounding-box generation methods. (ai/evaluation/localization_evaluation.py crashes with KeyError: 'localization_mask').",
                "Converting the continuous 14x14 distance map into discrete bounding boxes would require inventing arbitrary spatial calibration thresholds not grounded in existing production logic.",
                "Evaluating the 711k normal feature tensor across 876 images requires >20 minutes on MPS and yields only coarse, blurry 14x14 blobs unsuitable for bounding-box detection."
            ]
        }
    }

    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write("VisionInspect AI - Existing Localization as Bounding Boxes Evaluation Report\n")
        f.write("=" * 85 + "\n")
        f.write(f"Model:           {UNET_MODEL_PATH}\n")
        f.write(f"Test Images:     {total_images} (Defective: {total_defective}, Normal: {total_normal})\n")
        f.write(f"GT Defect Boxes: {total_gt_boxes}\n")
        f.write(f"Configured Th:   {existing_threshold:.4f}, Best F1 Th: {best_th:.4f}\n")
        f.write(f"Post-Process:    morph={morph_kernel}, min_area={min_component_area}\n\n")

        f.write("1. U-NET THRESHOLD SWEEP (IoU >= 0.50):\n")
        f.write(df_thresh.to_string(index=False) + "\n\n")

        f.write("2. LOCALIZATION THRESHOLD COMPARISON (IoU 0.25 vs 0.50 vs 0.75):\n")
        f.write(df_loc.to_string(index=False) + "\n\n")

        f.write("3. DEFECT SIZE ANALYSIS (GT Area Bins at IoU >= 0.50):\n")
        f.write(df_size.to_string(index=False) + "\n\n")

        f.write("4. DIRECT YOLO vs U-NET BOX-LEVEL COMPARISON (IoU >= 0.50):\n")
        f.write(df_method_comp.to_string(index=False) + "\n\n")

        f.write("5. SIZE BIN RECALL COMPARISON ACROSS METHODS:\n")
        f.write(df_size_comp_all.to_string(index=False) + "\n\n")

        f.write("6. PER-CATEGORY BOX-LEVEL RECALL COMPARISON (IoU >= 0.50):\n")
        f.write(df_cat.to_string(index=False) + "\n\n")

        f.write("7. TASK 6: ANOMALY HEATMAP AUDIT:\n")
        for finding in json_payload["task6_anomaly_heatmap_audit"]["findings"]:
            f.write(f" - {finding}\n")
        f.write("\n" + "=" * 85 + "\n")

    print(f"\nEvaluation reports saved successfully:")
    print(f" - JSON: {json_path}")
    print(f" - TXT:  {txt_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
