"""
VisionInspect AI - Comprehensive Box-Level YOLO Confidence & Error Analysis
Diagnostic script to evaluate the baseline YOLO11n model (best.pt) on the test split.
- Runs ONE single inference pass at conf=0.01 in memory-safe batches (batch_size=32).
- Evaluates box-level metrics at conf: 0.01, 0.05, 0.10, 0.15, 0.20, 0.25 at IoU >= 0.50.
- Evaluates confidence bands ([0.01, 0.05), [0.05, 0.10), [0.10, 0.15), [0.15, 0.20), [0.20, 0.25), [0.25, 1.0]).
- Evaluates localization at IoU >= 0.25, 0.50, 0.75.
- Evaluates small defects by size bins (<0.1%, 0.1-0.5%, 0.5-1%, 1-5%, >5%).
- Evaluates per-category box-level metrics across all thresholds.
- Saves JSON and TXT reports to ai/weights/yolo/evaluation_confidence/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "ai/weights/yolo/baseline/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/images/test"
TEST_LBL_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/labels/test"
OUTPUT_DIR = PROJECT_ROOT / "ai/weights/yolo/evaluation_confidence"


def box_iou(box1, box2):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
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
    Parse test images and labels.
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
    Returns:
        matched_preds: set of pred indices matched
        matched_gts: set of gt indices matched
        pred_matches: dict mapping pred_idx -> (gt_idx, iou)
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


def evaluate_dataset_at_threshold(manifest, all_preds, conf_thresh, iou_thresh=0.5):
    """
    Evaluate all test images at a specific confidence and IoU threshold.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_remaining = 0
    total_gt = sum(m["num_gt"] for m in manifest)

    image_level_eval = []

    for m, preds in zip(manifest, all_preds):
        # Filter predictions by confidence
        filtered_preds = [p for p in preds if p["conf"] >= conf_thresh]
        total_pred_remaining += len(filtered_preds)

        matched_p, matched_g, _ = match_predictions_greedy(m["gt_boxes"], filtered_preds, iou_thresh=iou_thresh)

        tp = len(matched_p)
        fp = len(filtered_preds) - tp
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
            "num_pred": len(filtered_preds),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "matched_gts": matched_g,
        })

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "conf_thresh": conf_thresh,
        "iou_thresh": iou_thresh,
        "total_gt": total_gt,
        "total_predictions": total_pred_remaining,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "image_level_eval": image_level_eval,
    }


def main():
    print("=" * 80)
    print("VisionInspect AI - In-Depth YOLO Box-Level Confidence & Defect Analysis")
    print("=" * 80)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Baseline weights not found at: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    manifest = load_test_manifest()
    img_paths = [str(m["img_path"]) for m in manifest]

    total_images = len(manifest)
    total_defective = sum(1 for m in manifest if m["is_defective"])
    total_normal = sum(1 for m in manifest if not m["is_defective"])
    total_gt_boxes = sum(m["num_gt"] for m in manifest)

    assert total_images == 876, f"Expected 876 test images, found {total_images}"
    assert total_gt_boxes == 403, f"Expected 403 GT defect boxes, found {total_gt_boxes}"

    print(f"Verified Test Set Scope:")
    print(f" - Total Images:       {total_images} (Defective: {total_defective}, Normal: {total_normal})")
    print(f" - Total GT Defect Boxes: {total_gt_boxes}")
    print(f" - Image Size:         640x640")
    print(f" - Hardware Device:    mps")

    # 1. Single Inference Pass at conf=0.01, iou=0.70
    print("\nRunning single inference pass on all 876 test images at conf=0.01 (batch_size=32)...")
    batch_size = 32
    all_cached_preds = []

    for i in range(0, len(img_paths), batch_size):
        chunk = img_paths[i:i + batch_size]
        chunk_preds = model.predict(chunk, conf=0.01, iou=0.7, imgsz=640, device="mps", verbose=False)
        for p in chunk_preds:
            boxes = []
            if len(p.boxes) > 0:
                h_img, w_img = p.orig_shape
                xyxy = p.boxes.xyxy.cpu().numpy()
                confs = p.boxes.conf.cpu().numpy()
                cls_ids = p.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                    boxes.append({
                        "class_id": int(cid),
                        "conf": float(c),
                        "box_norm": [x1 / w_img, y1 / h_img, x2 / w_img, y2 / h_img],
                        "box_px": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    })
            all_cached_preds.append(boxes)

    total_raw_preds = sum(len(b) for b in all_cached_preds)
    print(f"Inference complete! Total candidate predictions cached (conf >= 0.01): {total_raw_preds}")

    # =========================================================================
    # 2. BOX-LEVEL METRICS AT SPECIFIED CONFIDENCE THRESHOLDS (IoU >= 0.50)
    # =========================================================================
    print("\n" + "=" * 80)
    print("BOX-LEVEL METRICS ACROSS CONFIDENCE THRESHOLDS (at IoU >= 0.50)")
    print("=" * 80)

    conf_thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
    overall_conf_results = []
    eval_by_conf = {}

    for conf in conf_thresholds:
        res = evaluate_dataset_at_threshold(manifest, all_cached_preds, conf, iou_thresh=0.50)
        eval_by_conf[conf] = res
        overall_conf_results.append({
            "conf_threshold": conf,
            "predictions_remaining": res["total_predictions"],
            "total_gt": res["total_gt"],
            "TP": res["TP"],
            "FP": res["FP"],
            "FN": res["FN"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
        })

    df_overall = pd.DataFrame(overall_conf_results)
    print(df_overall.to_string(index=False, formatters={
        "conf_threshold": lambda x: f"{x:.2f}",
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    # =========================================================================
    # 3. CONFIDENCE-BAND MATCH RATES (IoU >= 0.50)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONFIDENCE-BAND MATCH RATES (Evaluation of Predictions by Confidence Range)")
    print("=" * 80)

    # Perform greedy matching at conf >= 0.01 globally on each image
    bands = [
        ("<0.05", 0.01, 0.05),
        ("0.05–0.10", 0.05, 0.10),
        ("0.10–0.15", 0.10, 0.15),
        ("0.15–0.20", 0.15, 0.20),
        ("0.20–0.25", 0.20, 0.25),
        (">=0.25", 0.25, 1.01),
    ]

    band_stats = {b[0]: {"total": 0, "matched": 0, "unmatched": 0} for b in bands}

    for m, preds in zip(manifest, all_cached_preds):
        matched_preds, _, _ = match_predictions_greedy(m["gt_boxes"], preds, iou_thresh=0.50)

        for p_idx, p in enumerate(preds):
            c = p["conf"]
            # Find band
            for label, low, high in bands:
                if low <= c < high:
                    band_stats[label]["total"] += 1
                    if p_idx in matched_preds:
                        band_stats[label]["matched"] += 1
                    else:
                        band_stats[label]["unmatched"] += 1
                    break

    band_records = []
    for label, low, high in bands:
        tot = band_stats[label]["total"]
        m_cnt = band_stats[label]["matched"]
        u_cnt = band_stats[label]["unmatched"]
        m_rate = (m_cnt / tot * 100) if tot > 0 else 0.0
        band_records.append({
            "confidence_band": label,
            "total_predictions": tot,
            "matched_to_gt_iou0.50": m_cnt,
            "unmatched": u_cnt,
            "match_rate_pct": m_rate,
        })

    df_bands = pd.DataFrame(band_records)
    print(df_bands.to_string(index=False, formatters={
        "match_rate_pct": lambda x: f"{x:.2f}%",
    }))

    # =========================================================================
    # 4. LOCALIZATION ANALYSIS AT conf >= 0.01 (IoU >= 0.25, 0.50, 0.75)
    # =========================================================================
    print("\n" + "=" * 80)
    print("LOCALIZATION ANALYSIS AT conf >= 0.01 (IoU Thresholds: 0.25, 0.50, 0.75)")
    print("=" * 80)

    iou_thresholds = [0.25, 0.50, 0.75]
    loc_results = []

    for th in iou_thresholds:
        res = evaluate_dataset_at_threshold(manifest, all_cached_preds, conf_thresh=0.01, iou_thresh=th)
        loc_results.append({
            "iou_threshold": th,
            "total_predictions": res["total_predictions"],
            "TP": res["TP"],
            "FP": res["FP"],
            "FN": res["FN"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "F1": res["F1"],
        })

    df_loc = pd.DataFrame(loc_results)
    print(df_loc.to_string(index=False, formatters={
        "iou_threshold": lambda x: f">={x:.2f}",
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    # =========================================================================
    # 5. SMALL-DEFECT ANALYSIS BY SIZE BINS (at conf=0.25, 0.10, 0.05, 0.01)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SMALL-DEFECT ANALYSIS BY GROUND-TRUTH BOUNDING BOX AREA")
    print("=" * 80)

    size_bins = [
        ("<0.1%", 0.0, 0.1),
        ("0.1–0.5%", 0.1, 0.5),
        ("0.5–1%", 0.5, 1.0),
        ("1–5%", 1.0, 5.0),
        (">5%", 5.0, 1000.0),
    ]

    size_analysis_data = {}

    for conf in [0.25, 0.10, 0.05, 0.01]:
        img_eval = eval_by_conf[conf]["image_level_eval"]
        records = []
        for label, low, high in size_bins:
            gt_in_bin = 0
            tp_in_bin = 0
            for item in img_eval:
                stem = item["stem"]
                # Find corresponding manifest entry
                m = next(entry for entry in manifest if entry["stem"] == stem)
                for g_idx, g in enumerate(m["gt_boxes"]):
                    if low <= g["area_pct"] < high:
                        gt_in_bin += 1
                        if g_idx in item["matched_gts"]:
                            tp_in_bin += 1
            fn_in_bin = gt_in_bin - tp_in_bin
            rec = (tp_in_bin / gt_in_bin * 100) if gt_in_bin > 0 else 0.0
            records.append({
                "size_bin": label,
                "gt_boxes": gt_in_bin,
                "TP": tp_in_bin,
                "FN": fn_in_bin,
                "recall_pct": rec,
            })
        size_analysis_data[conf] = records

    print("Defect Size Analysis at Baseline conf=0.25 (IoU >= 0.50):")
    df_size_25 = pd.DataFrame(size_analysis_data[0.25])
    print(df_size_25.to_string(index=False, formatters={"recall_pct": lambda x: f"{x:.2f}%"}))

    print("\nDefect Size Analysis at conf=0.05 (IoU >= 0.50):")
    df_size_05 = pd.DataFrame(size_analysis_data[0.05])
    print(df_size_05.to_string(index=False, formatters={"recall_pct": lambda x: f"{x:.2f}%"}))

    # =========================================================================
    # 6. PER-CATEGORY BOX-LEVEL RESULTS (conf=0.05, 0.10, 0.15, 0.20, 0.25)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PER-CATEGORY BOX-LEVEL RESULTS (IoU >= 0.50)")
    print("=" * 80)

    categories = sorted(list(set(m["category"] for m in manifest)))
    cat_results_by_conf = {}

    for conf in [0.25, 0.20, 0.15, 0.10, 0.05]:
        img_eval = eval_by_conf[conf]["image_level_eval"]
        rows = []
        for cat in categories:
            cat_items = [it for it in img_eval if it["category"] == cat]
            gt_cnt = sum(it["num_gt"] for it in cat_items)
            tp_cnt = sum(it["tp"] for it in cat_items)
            fp_cnt = sum(it["fp"] for it in cat_items)
            fn_cnt = sum(it["fn"] for it in cat_items)
            prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
            rec = tp_cnt / gt_cnt if gt_cnt > 0 else 0.0

            rows.append({
                "category": cat,
                "gt_boxes": gt_cnt,
                "TP": tp_cnt,
                "FP": fp_cnt,
                "FN": fn_cnt,
                "Precision": prec,
                "Recall": rec,
            })
        cat_results_by_conf[conf] = rows

    for conf in [0.25, 0.10, 0.05]:
        print(f"\nCategory Box-Level Breakdown at conf={conf:.2f}:")
        df_c = pd.DataFrame(cat_results_by_conf[conf])
        print(df_c.to_string(index=False, formatters={
            "Precision": lambda x: f"{x:.4f}",
            "Recall": lambda x: f"{x:.4f}",
        }))

    # =========================================================================
    # 7. SAVE MACHINE-READABLE JSON & HUMAN-READABLE TXT REPORTS
    # =========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "confidence_analysis.json"
    txt_path = OUTPUT_DIR / "confidence_analysis.txt"

    # Prepare serializable dictionary
    report_dict = {
        "metadata": {
            "model_path": str(WEIGHTS_PATH),
            "test_images": total_images,
            "defective_images": total_defective,
            "normal_images": total_normal,
            "gt_defect_boxes": total_gt_boxes,
            "image_size": 640,
            "device": "mps",
        },
        "box_level_confidence_sweep": overall_conf_results,
        "confidence_bands": band_records,
        "localization_at_conf0.01": loc_results,
        "defect_size_breakdown": {
            f"conf_{conf}": size_analysis_data[conf] for conf in size_analysis_data
        },
        "category_breakdown": {
            f"conf_{conf}": cat_results_by_conf[conf] for conf in cat_results_by_conf
        },
    }

    # Clean out unneeded image_level_eval for JSON compactness
    for item in report_dict["box_level_confidence_sweep"]:
        item.pop("image_level_eval", None)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nSaved machine-readable JSON: {json_path}")

    # Build TXT report
    txt_content = []
    txt_content.append("=" * 80)
    txt_content.append("VisionInspect AI - YOLO11n Box-Level Confidence & Error Analysis Report")
    txt_content.append("=" * 80)
    txt_content.append(f"Model:           {WEIGHTS_PATH}")
    txt_content.append(f"Test Images:     {total_images} (Defective: {total_defective}, Normal: {total_normal})")
    txt_content.append(f"GT Defect Boxes: {total_gt_boxes}")
    txt_content.append("")
    txt_content.append("1. BOX-LEVEL METRICS ACROSS CONFIDENCE THRESHOLDS (IoU >= 0.50):")
    txt_content.append(df_overall.to_string(index=False))
    txt_content.append("")
    txt_content.append("2. CONFIDENCE-BAND MATCH RATES (IoU >= 0.50):")
    txt_content.append(df_bands.to_string(index=False))
    txt_content.append("")
    txt_content.append("3. LOCALIZATION ANALYSIS AT conf >= 0.01:")
    txt_content.append(df_loc.to_string(index=False))
    txt_content.append("")
    txt_content.append("4. SMALL-DEFECT SIZE BREAKDOWN at conf=0.25 (Baseline):")
    txt_content.append(df_size_25.to_string(index=False))
    txt_content.append("")
    txt_content.append("5. SMALL-DEFECT SIZE BREAKDOWN at conf=0.05:")
    txt_content.append(df_size_05.to_string(index=False))
    txt_content.append("")
    txt_content.append("6. PER-CATEGORY BOX-LEVEL METRICS at conf=0.25 (Baseline):")
    txt_content.append(pd.DataFrame(cat_results_by_conf[0.25]).to_string(index=False))
    txt_content.append("")
    txt_content.append("7. PER-CATEGORY BOX-LEVEL METRICS at conf=0.05:")
    txt_content.append(pd.DataFrame(cat_results_by_conf[0.05]).to_string(index=False))

    txt_path.write_text("\n".join(txt_content), encoding="utf-8")
    print(f"Saved human-readable TXT:    {txt_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
