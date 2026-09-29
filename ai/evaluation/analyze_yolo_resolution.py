"""
VisionInspect AI - YOLO Inference Resolution Diagnostic
Evaluates the baseline YOLO11n model (best.pt) across three inference resolutions:
- 640x640
- 768x768
- 1024x1024

No retraining or model modification is performed.
Diagnostic pass at conf=0.01, iou=0.70 to evaluate maximum candidate recall and localization.
Reports:
1. Overall comparison: TP, FP, FN, Precision, Recall, F1 (IoU >= 0.50)
2. Localization comparison: IoU >= 0.25, 0.50, 0.75
3. Defect size analysis (<0.1%, 0.1-0.5%, 0.5-1%, 1-5%, >5%)
4. Per-category box-level recall for all 15 MVTec categories
5. Inference latency / speed benchmark
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "ai/weights/yolo/baseline/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/images/test"
TEST_LBL_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/labels/test"
OUTPUT_DIR = PROJECT_ROOT / "ai/weights/yolo/evaluation_resolution"

RESOLUTIONS = [640, 768, 1024]
BATCH_SIZE = 16
CONF_DIAGNOSTIC = 0.01
NMS_IOU = 0.70

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


def evaluate_dataset(manifest, all_preds, conf_thresh=0.01, iou_thresh=0.5):
    """
    Evaluate dataset predictions at specified confidence and IoU threshold.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_pred_remaining = 0
    total_gt = sum(m["num_gt"] for m in manifest)

    image_level_eval = []

    for m, preds in zip(manifest, all_preds):
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


def compute_defect_size_breakdown(manifest, eval_result):
    """
    Calculate recall per defect size bin.
    """
    img_eval = eval_result["image_level_eval"]
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


def compute_category_breakdown(manifest, eval_result):
    """
    Calculate TP, FP, FN, Precision, Recall per category.
    """
    img_eval = eval_result["image_level_eval"]
    cat_records = []

    for cat in CATEGORIES:
        cat_items = [it for it in img_eval if it["category"] == cat]
        gt_cnt = sum(it["num_gt"] for it in cat_items)
        tp_cnt = sum(it["tp"] for it in cat_items)
        fp_cnt = sum(it["fp"] for it in cat_items)
        fn_cnt = sum(it["fn"] for it in cat_items)

        prec = tp_cnt / (tp_cnt + fp_cnt) if (tp_cnt + fp_cnt) > 0 else 0.0
        rec = tp_cnt / gt_cnt if gt_cnt > 0 else 0.0

        cat_records.append({
            "category": cat,
            "gt_boxes": gt_cnt,
            "TP": tp_cnt,
            "FP": fp_cnt,
            "FN": fn_cnt,
            "Precision": prec,
            "Recall": rec,
        })
    return cat_records


def run_inference_for_resolution(model, manifest, imgsz, batch_size=16):
    """
    Run inference over all manifest images at a specific resolution.
    Measures latency and returns cached normalized predictions.
    """
    img_paths = [str(m["img_path"]) for m in manifest]
    all_cached_preds = []
    speed_stats = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}

    print(f"\n---> Running inference pass for imgsz={imgsz} on {len(img_paths)} images (batch_size={batch_size})...")
    start_wall = time.perf_counter()

    for i in range(0, len(img_paths), batch_size):
        chunk = img_paths[i:i + batch_size]
        chunk_preds = model.predict(
            chunk,
            conf=CONF_DIAGNOSTIC,
            iou=NMS_IOU,
            imgsz=imgsz,
            device="mps",
            verbose=False
        )

        for p in chunk_preds:
            # Accumulate Ultralytics speed metrics (in ms)
            if hasattr(p, "speed") and isinstance(p.speed, dict):
                speed_stats["preprocess"] += p.speed.get("preprocess", 0.0)
                speed_stats["inference"] += p.speed.get("inference", 0.0)
                speed_stats["postprocess"] += p.speed.get("postprocess", 0.0)

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

        # Clear MPS cache periodically to prevent peak memory growth
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    elapsed_wall = time.perf_counter() - start_wall
    n_images = len(img_paths)
    avg_speed = {
        "preprocess_ms": speed_stats["preprocess"] / n_images,
        "inference_ms": speed_stats["inference"] / n_images,
        "postprocess_ms": speed_stats["postprocess"] / n_images,
        "total_pipeline_ms": (speed_stats["preprocess"] + speed_stats["inference"] + speed_stats["postprocess"]) / n_images,
        "wall_time_s": elapsed_wall,
        "wall_ms_per_image": (elapsed_wall / n_images) * 1000.0,
        "wall_fps": n_images / elapsed_wall if elapsed_wall > 0 else 0.0,
    }

    raw_preds_cnt = sum(len(b) for b in all_cached_preds)
    print(f"      Completed imgsz={imgsz}: {raw_preds_cnt} candidate predictions found.")
    print(f"      Latency: {avg_speed['inference_ms']:.2f} ms inference, {avg_speed['wall_ms_per_image']:.2f} ms total wall/img ({avg_speed['wall_fps']:.1f} FPS)")

    return all_cached_preds, avg_speed


def main():
    print("=" * 85)
    print("VisionInspect AI - YOLO Inference Resolution Diagnostic (640 vs 768 vs 1024)")
    print("=" * 85)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Baseline weights not found at: {WEIGHTS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_test_manifest()
    total_images = len(manifest)
    total_defective = sum(1 for m in manifest if m["is_defective"])
    total_normal = sum(1 for m in manifest if not m["is_defective"])
    total_gt_boxes = sum(m["num_gt"] for m in manifest)

    # Validation assertions
    assert total_images == 876, f"Expected 876 test images, found {total_images}"
    assert total_defective == 260, f"Expected 260 defective test images, found {total_defective}"
    assert total_normal == 616, f"Expected 616 normal test images, found {total_normal}"
    assert total_gt_boxes == 403, f"Expected 403 GT defect boxes, found {total_gt_boxes}"

    print(f"Verified Scope:")
    print(f" - Model Weights:       {WEIGHTS_PATH}")
    print(f" - Test Set Images:     {total_images} (Defective: {total_defective}, Normal: {total_normal})")
    print(f" - Ground-Truth Boxes:  {total_gt_boxes}")
    print(f" - Diagnostic Conf:     {CONF_DIAGNOSTIC}")
    print(f" - NMS IoU:             {NMS_IOU}")
    print(f" - Batch Size:          {BATCH_SIZE}")
    print(f" - Device:              mps")

    model = YOLO(str(WEIGHTS_PATH))

    res_evaluations = {}
    res_speeds = {}

    for res in RESOLUTIONS:
        preds, speed_info = run_inference_for_resolution(model, manifest, imgsz=res, batch_size=BATCH_SIZE)
        res_speeds[res] = speed_info

        # 1. Primary evaluation at IoU >= 0.50 (conf=0.01)
        eval_iou50 = evaluate_dataset(manifest, preds, conf_thresh=CONF_DIAGNOSTIC, iou_thresh=0.50)
        # 2. Localization evaluations at IoU >= 0.25 and 0.75
        eval_iou25 = evaluate_dataset(manifest, preds, conf_thresh=CONF_DIAGNOSTIC, iou_thresh=0.25)
        eval_iou75 = evaluate_dataset(manifest, preds, conf_thresh=CONF_DIAGNOSTIC, iou_thresh=0.75)

        # 3. Defect size breakdown at IoU >= 0.50
        size_breakdown = compute_defect_size_breakdown(manifest, eval_iou50)

        # 4. Per-category breakdown at IoU >= 0.50
        cat_breakdown = compute_category_breakdown(manifest, eval_iou50)

        # Also compute conf=0.25 operational baseline for reference
        eval_conf25_iou50 = evaluate_dataset(manifest, preds, conf_thresh=0.25, iou_thresh=0.50)
        size_breakdown_conf25 = compute_defect_size_breakdown(manifest, eval_conf25_iou50)

        res_evaluations[res] = {
            "primary_iou50": eval_iou50,
            "loc_iou25": eval_iou25,
            "loc_iou75": eval_iou75,
            "size_breakdown": size_breakdown,
            "category_breakdown": cat_breakdown,
            "operational_conf25": {
                "overall": {
                    "total_predictions": eval_conf25_iou50["total_predictions"],
                    "TP": eval_conf25_iou50["TP"],
                    "FP": eval_conf25_iou50["FP"],
                    "FN": eval_conf25_iou50["FN"],
                    "Precision": eval_conf25_iou50["Precision"],
                    "Recall": eval_conf25_iou50["Recall"],
                    "F1": eval_conf25_iou50["F1"],
                },
                "size_breakdown": size_breakdown_conf25,
            }
        }

    # =========================================================================
    # BUILD COMPARISON TABLES
    # =========================================================================

    # Table 1: Overall Comparison (IoU >= 0.50, conf=0.01)
    overall_comp = []
    for res in RESOLUTIONS:
        ev = res_evaluations[res]["primary_iou50"]
        overall_comp.append({
            "Resolution": f"{res}x{res}",
            "Predictions": ev["total_predictions"],
            "GT Boxes": ev["total_gt"],
            "TP": ev["TP"],
            "FP": ev["FP"],
            "FN": ev["FN"],
            "Precision": ev["Precision"],
            "Recall": ev["Recall"],
            "F1": ev["F1"],
        })
    df_overall = pd.DataFrame(overall_comp)

    # Table 2: Defect Size Breakdown Comparison
    size_comp = []
    for res in RESOLUTIONS:
        sb = {item["size_bin"]: item["recall_pct"] for item in res_evaluations[res]["size_breakdown"]}
        size_comp.append({
            "Resolution": f"{res}x{res}",
            "<0.1% Recall": f"{sb['<0.1%']:.2f}%",
            "0.1–0.5% Recall": f"{sb['0.1–0.5%']:.2f}%",
            "0.5–1% Recall": f"{sb['0.5–1%']:.2f}%",
            "1–5% Recall": f"{sb['1–5%']:.2f}%",
            ">5% Recall": f"{sb['>5%']:.2f}%",
        })
    df_size = pd.DataFrame(size_comp)

    # Table 3: Localization Comparison (IoU 0.25 vs 0.50 vs 0.75)
    loc_comp = []
    for res in RESOLUTIONS:
        for iou_lbl, key in [("IoU >= 0.25", "loc_iou25"), ("IoU >= 0.50", "primary_iou50"), ("IoU >= 0.75", "loc_iou75")]:
            ev = res_evaluations[res][key]
            loc_comp.append({
                "Resolution": f"{res}x{res}",
                "IoU Threshold": iou_lbl,
                "TP": ev["TP"],
                "FP": ev["FP"],
                "FN": ev["FN"],
                "Precision": ev["Precision"],
                "Recall": ev["Recall"],
                "F1": ev["F1"],
            })
    df_loc = pd.DataFrame(loc_comp)

    # Table 4: Per-Category Box Recall Comparison
    cat_comp = []
    for cat in CATEGORIES:
        row = {"Category": cat}
        for res in RESOLUTIONS:
            c_data = next(it for it in res_evaluations[res]["category_breakdown"] if it["category"] == cat)
            row[f"GT ({res})"] = c_data["gt_boxes"]
            row[f"TP ({res})"] = c_data["TP"]
            row[f"Recall ({res})"] = f"{c_data['Recall'] * 100.0:.1f}%"
        cat_comp.append(row)
    df_cat = pd.DataFrame(cat_comp)

    # Table 5: Inference Speed Comparison
    speed_comp = []
    for res in RESOLUTIONS:
        sp = res_speeds[res]
        speed_comp.append({
            "Resolution": f"{res}x{res}",
            "Preprocess (ms)": f"{sp['preprocess_ms']:.2f}",
            "Inference (ms)": f"{sp['inference_ms']:.2f}",
            "Postprocess (ms)": f"{sp['postprocess_ms']:.2f}",
            "Total Pipeline (ms)": f"{sp['total_pipeline_ms']:.2f}",
            "Wall ms/img": f"{sp['wall_ms_per_image']:.2f}",
            "Throughput (FPS)": f"{sp['wall_fps']:.1f}",
        })
    df_speed = pd.DataFrame(speed_comp)

    # Print to console
    print("\n" + "=" * 85)
    print("1. OVERALL BOX-LEVEL COMPARISON (IoU >= 0.50, conf=0.01)")
    print("=" * 85)
    print(df_overall.to_string(index=False, formatters={
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    print("\n" + "=" * 85)
    print("2. DEFECT SIZE ANALYSIS COMPARISON (Recall % by Ground-Truth Area Bin)")
    print("=" * 85)
    print(df_size.to_string(index=False))

    print("\n" + "=" * 85)
    print("3. LOCALIZATION THRESHOLD COMPARISON (IoU 0.25 vs 0.50 vs 0.75)")
    print("=" * 85)
    print(df_loc.to_string(index=False, formatters={
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    print("\n" + "=" * 85)
    print("4. PER-CATEGORY BOX-LEVEL RECALL COMPARISON (IoU >= 0.50)")
    print("=" * 85)
    print(df_cat.to_string(index=False))

    print("\n" + "=" * 85)
    print("5. INFERENCE SPEED & THROUGHPUT BENCHMARK (MPS)")
    print("=" * 85)
    print(df_speed.to_string(index=False))

    # =========================================================================
    # SAVE JSON AND TXT REPORTS
    # =========================================================================
    json_path = OUTPUT_DIR / "resolution_analysis.json"
    txt_path = OUTPUT_DIR / "resolution_analysis.txt"

    # Prepare JSON serializable structure
    json_payload = {
        "metadata": {
            "model_path": str(WEIGHTS_PATH),
            "test_images": total_images,
            "defective_images": total_defective,
            "normal_images": total_normal,
            "gt_defect_boxes": total_gt_boxes,
            "conf_threshold": CONF_DIAGNOSTIC,
            "nms_iou": NMS_IOU,
            "device": "mps",
            "batch_size": BATCH_SIZE,
        },
        "overall_comparison": overall_comp,
        "size_comparison": size_comp,
        "localization_comparison": loc_comp,
        "category_comparison": cat_comp,
        "speed_comparison": speed_comp,
        "detailed_results_by_resolution": {}
    }

    for res in RESOLUTIONS:
        ev50 = res_evaluations[res]["primary_iou50"]
        ev25 = res_evaluations[res]["loc_iou25"]
        ev75 = res_evaluations[res]["loc_iou75"]
        json_payload["detailed_results_by_resolution"][str(res)] = {
            "speed": res_speeds[res],
            "primary_iou50": {
                "total_predictions": ev50["total_predictions"],
                "total_gt": ev50["total_gt"],
                "TP": ev50["TP"],
                "FP": ev50["FP"],
                "FN": ev50["FN"],
                "Precision": ev50["Precision"],
                "Recall": ev50["Recall"],
                "F1": ev50["F1"],
            },
            "loc_iou25": {
                "TP": ev25["TP"], "FP": ev25["FP"], "FN": ev25["FN"],
                "Precision": ev25["Precision"], "Recall": ev25["Recall"], "F1": ev25["F1"],
            },
            "loc_iou75": {
                "TP": ev75["TP"], "FP": ev75["FP"], "FN": ev75["FN"],
                "Precision": ev75["Precision"], "Recall": ev75["Recall"], "F1": ev75["F1"],
            },
            "size_breakdown": res_evaluations[res]["size_breakdown"],
            "category_breakdown": res_evaluations[res]["category_breakdown"],
            "operational_conf25": res_evaluations[res]["operational_conf25"],
        }

    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    # Generate Human-Readable Text Report
    with open(txt_path, "w") as f:
        f.write("=" * 85 + "\n")
        f.write("VisionInspect AI - YOLO Inference Resolution Diagnostic Report (640 vs 768 vs 1024)\n")
        f.write("=" * 85 + "\n")
        f.write(f"Model:           {WEIGHTS_PATH}\n")
        f.write(f"Test Images:     {total_images} (Defective: {total_defective}, Normal: {total_normal})\n")
        f.write(f"GT Defect Boxes: {total_gt_boxes}\n")
        f.write(f"Diagnostic Conf: {CONF_DIAGNOSTIC}, NMS IoU: {NMS_IOU}\n\n")

        f.write("1. OVERALL BOX-LEVEL COMPARISON (IoU >= 0.50, conf=0.01):\n")
        f.write(df_overall.to_string(index=False) + "\n\n")

        f.write("2. DEFECT SIZE ANALYSIS COMPARISON (Recall % by Ground-Truth Area Bin):\n")
        f.write(df_size.to_string(index=False) + "\n\n")

        f.write("3. LOCALIZATION THRESHOLD COMPARISON (IoU 0.25 vs 0.50 vs 0.75):\n")
        f.write(df_loc.to_string(index=False) + "\n\n")

        f.write("4. PER-CATEGORY BOX-LEVEL RECALL COMPARISON (IoU >= 0.50):\n")
        f.write(df_cat.to_string(index=False) + "\n\n")

        f.write("5. INFERENCE SPEED & THROUGHPUT BENCHMARK (MPS):\n")
        f.write(df_speed.to_string(index=False) + "\n\n")

    print(f"\nDiagnostic reports saved successfully:")
    print(f" - JSON: {json_path}")
    print(f" - TXT:  {txt_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
