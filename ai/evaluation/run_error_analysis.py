"""
VisionInspect AI - Comprehensive YOLO Prediction Error Analysis
Evaluates the existing baseline YOLO11n model (best.pt) on the test split.
No training, no weights modification, no dataset modification.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_PATH = PROJECT_ROOT / "ai/weights/yolo/baseline/weights/best.pt"
TEST_IMG_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/images/test"
TEST_LBL_DIR = PROJECT_ROOT / "datasets/mvtec_yolo/labels/test"
OUTPUT_VIS_DIR = PROJECT_ROOT / "ai/weights/yolo/error_analysis"


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


def main():
    print("=" * 75)
    print("VisionInspect AI - YOLO11n Baseline Diagnostic Error Analysis")
    print("=" * 75)

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Baseline weights not found at: {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))
    manifest = load_test_manifest()
    img_paths = [str(m["img_path"]) for m in manifest]

    total_images = len(manifest)
    total_defective_images = sum(1 for m in manifest if m["is_defective"])
    total_normal_images = sum(1 for m in manifest if not m["is_defective"])
    total_gt_boxes = sum(m["num_gt"] for m in manifest)

    print(f"Total Test Images:     {total_images}")
    print(f" - Defective Images:   {total_defective_images}")
    print(f" - Normal Images:      {total_normal_images}")
    print(f"Total GT Defect Boxes: {total_gt_boxes}")

    # =========================================================================
    # 1. RUN PREDICTIONS AT MULTIPLE CONFIDENCE THRESHOLDS
    # =========================================================================
    print("\n" + "=" * 75)
    print("1. PREDICTIONS AT MULTIPLE CONFIDENCE THRESHOLDS")
    print("=" * 75)

    conf_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    conf_results = []
    preds_by_conf = {}

    print("Running chunked test set inference (batch_size=32) at conf=0.01, iou=0.70...")
    batch_size = 32
    all_parsed_preds = []

    for i in range(0, len(img_paths), batch_size):
        chunk_paths = img_paths[i:i + batch_size]
        chunk_preds = model.predict(chunk_paths, conf=0.01, iou=0.7, imgsz=640, device="mps", verbose=False)
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
            all_parsed_preds.append(boxes)

    for conf in conf_thresholds:
        parsed_preds = [[b for b in img_boxes if b["conf"] >= conf] for img_boxes in all_parsed_preds]
        preds_by_conf[conf] = parsed_preds

        # Metrics for this confidence threshold
        def_with_pred = 0
        norm_with_pred = 0
        total_pred_boxes = 0
        normal_fp_boxes = 0

        for m, preds in zip(manifest, parsed_preds):
            n_b = len(preds)
            total_pred_boxes += n_b
            if m["is_defective"]:
                if n_b > 0:
                    def_with_pred += 1
            else:
                if n_b > 0:
                    norm_with_pred += 1
                    normal_fp_boxes += n_b

        conf_results.append({
            "conf": conf,
            "def_with_pred": def_with_pred,
            "def_img_recall_pct": (def_with_pred / total_defective_images * 100),
            "norm_with_pred": norm_with_pred,
            "norm_img_fp_pct": (norm_with_pred / total_normal_images * 100),
            "total_pred_boxes": total_pred_boxes,
            "normal_fp_boxes": normal_fp_boxes,
        })

    df_conf = pd.DataFrame(conf_results)
    print("\nConfidence Threshold Analysis Results:")
    print(df_conf.to_string(index=False, formatters={
        "conf": lambda x: f"{x:.2f}",
        "def_img_recall_pct": lambda x: f"{x:.2f}%",
        "norm_img_fp_pct": lambda x: f"{x:.2f}%",
    }))

    # =========================================================================
    # 2. GROUND-TRUTH MATCHING & RECALL AT MULTIPLE IoU THRESHOLDS (at conf=0.25)
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. GROUND-TRUTH MATCHING (at baseline conf=0.25)")
    print("=" * 75)

    base_preds = preds_by_conf[0.25]

    def match_gt_predictions(manifest_list, preds_list, iou_thresh=0.5):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        gt_matches = []

        for m, preds in zip(manifest_list, preds_list):
            gt_boxes = m["gt_boxes"]
            matched_gt = set()
            matched_pred = set()

            # Sort predictions by confidence descending
            sorted_pred_indices = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)

            for p_idx in sorted_pred_indices:
                p = preds[p_idx]
                best_iou = 0.0
                best_g_idx = -1
                for g_idx, g in enumerate(gt_boxes):
                    if g_idx in matched_gt:
                        continue
                    iou = box_iou(p["box_norm"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_g_idx = g_idx

                if best_iou >= iou_thresh and best_g_idx != -1:
                    matched_gt.add(best_g_idx)
                    matched_pred.add(p_idx)

            tp = len(matched_pred)
            fp = len(preds) - tp
            fn = len(gt_boxes) - len(matched_gt)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "iou_threshold": iou_thresh,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
        }

    iou_thresholds = [0.25, 0.50, 0.75]
    iou_eval_results = [match_gt_predictions(manifest, base_preds, th) for th in iou_thresholds]
    df_iou = pd.DataFrame(iou_eval_results)
    print("\nMatching Metrics at Different IoU Thresholds (conf=0.25):")
    print(df_iou.to_string(index=False, formatters={
        "iou_threshold": lambda x: f">={x:.2f}",
        "Precision": lambda x: f"{x:.4f}",
        "Recall": lambda x: f"{x:.4f}",
        "F1": lambda x: f"{x:.4f}",
    }))

    # =========================================================================
    # 3. MISSED DEFECT ANALYSIS (conf=0.25, IoU=0.50)
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. MISSED DEFECT ANALYSIS (conf=0.25, IoU=0.50)")
    print("=" * 75)

    missed_records = []
    detected_records = []

    for m, preds in zip(manifest, base_preds):
        gt_boxes = m["gt_boxes"]
        matched_gt = set()

        sorted_pred_indices = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)
        for p_idx in sorted_pred_indices:
            p = preds[p_idx]
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = box_iou(p["box_norm"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            if best_iou >= 0.5 and best_g_idx != -1:
                matched_gt.add(best_g_idx)

        for g_idx, g in enumerate(gt_boxes):
            # Find best overlapping prediction (even if < 0.5 iou or below threshold)
            best_iou_any = 0.0
            highest_conf_overlap = 0.0
            for p in preds:
                iou = box_iou(p["box_norm"], g["box"])
                if iou > best_iou_any:
                    best_iou_any = iou
                if iou > 0.0 and p["conf"] > highest_conf_overlap:
                    highest_conf_overlap = p["conf"]

            record = {
                "category": m["category"],
                "defect_type": m["defect_type"],
                "stem": m["stem"],
                "nw": g["nw"],
                "nh": g["nh"],
                "area_pct": g["area_pct"],
                "best_iou": best_iou_any,
                "highest_conf_overlap": highest_conf_overlap,
            }

            if g_idx in matched_gt:
                detected_records.append(record)
            else:
                missed_records.append(record)

    df_missed = pd.DataFrame(missed_records)
    df_all_gt = pd.DataFrame(missed_records + detected_records)

    # Size bin definition
    size_bins = [
        ("<0.1%", 0.0, 0.1),
        ("0.1–0.5%", 0.1, 0.5),
        ("0.5–1.0%", 0.5, 1.0),
        ("1.0–5.0%", 1.0, 5.0),
        (">5.0%", 5.0, 100.0),
    ]

    size_summary = []
    for label, low, high in size_bins:
        all_in_bin = df_all_gt[(df_all_gt["area_pct"] >= low) & (df_all_gt["area_pct"] < high)]
        missed_in_bin = df_missed[(df_missed["area_pct"] >= low) & (df_missed["area_pct"] < high)]

        total_b = len(all_in_bin)
        missed_b = len(missed_in_bin)
        detected_b = total_b - missed_b
        recall_pct = (detected_b / total_b * 100) if total_b > 0 else 0.0
        miss_rate_pct = (missed_b / total_b * 100) if total_b > 0 else 0.0

        size_summary.append({
            "size_bin": label,
            "total_gt_boxes": total_b,
            "detected_boxes": detected_b,
            "missed_boxes": missed_b,
            "recall_pct": recall_pct,
            "miss_rate_pct": miss_rate_pct,
        })

    df_size = pd.DataFrame(size_summary)
    print("\nMissed Defects by Bounding Box Area (% of image):")
    print(df_size.to_string(index=False, formatters={
        "recall_pct": lambda x: f"{x:.2f}%",
        "miss_rate_pct": lambda x: f"{x:.2f}%",
    }))

    # =========================================================================
    # 4. CATEGORY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 75)
    print("4. CATEGORY ANALYSIS (conf=0.25, IoU=0.50)")
    print("=" * 75)

    categories = sorted(list(set(m["category"] for m in manifest)))
    cat_summary = []

    for cat in categories:
        cat_m = [m for m in manifest if m["category"] == cat]
        cat_preds = [base_preds[i] for i, m in enumerate(manifest) if m["category"] == cat]

        def_imgs = sum(1 for m in cat_m if m["is_defective"])
        gt_boxes_cat = sum(m["num_gt"] for m in cat_m)

        # Match at IoU=0.50
        matched_result = match_gt_predictions(cat_m, cat_preds, iou_thresh=0.50)
        det_boxes = matched_result["TP"]
        rec = (det_boxes / gt_boxes_cat * 100) if gt_boxes_cat > 0 else 0.0

        # Normal FP boxes
        norm_fp_boxes = 0
        for m, preds in zip(cat_m, cat_preds):
            if not m["is_defective"]:
                norm_fp_boxes += len(preds)

        cat_summary.append({
            "category": cat,
            "defective_images": def_imgs,
            "gt_boxes": gt_boxes_cat,
            "detected_gt_boxes": det_boxes,
            "box_recall_pct": rec,
            "normal_fp_boxes": norm_fp_boxes,
        })

    df_cat = pd.DataFrame(cat_summary)
    print("\nPer-Category Defect Box Detection and Normal False Positives:")
    print(df_cat.to_string(index=False, formatters={
        "box_recall_pct": lambda x: f"{x:.2f}%",
    }))

    # =========================================================================
    # 5. MULTI-BOX ANALYSIS
    # =========================================================================
    print("\n" + "=" * 75)
    print("5. MULTI-BOX ANALYSIS (conf=0.25, IoU=0.50)")
    print("=" * 75)

    group_definitions = [
        ("Exactly 1 GT Box", lambda n: n == 1),
        ("Exactly 2 GT Boxes", lambda n: n == 2),
        ("3+ GT Boxes", lambda n: n >= 3),
    ]

    multibox_summary = []

    for label, cond in group_definitions:
        group_m = [m for m in manifest if m["is_defective"] and cond(m["num_gt"])]
        group_indices = [i for i, m in enumerate(manifest) if m["is_defective"] and cond(m["num_gt"])]
        group_preds = [base_preds[i] for i in group_indices]

        n_imgs = len(group_m)
        gt_b = sum(m["num_gt"] for m in group_m)
        matched_result = match_gt_predictions(group_m, group_preds, iou_thresh=0.50)
        det_b = matched_result["TP"]
        rec = (det_b / gt_b * 100) if gt_b > 0 else 0.0

        multibox_summary.append({
            "group": label,
            "image_count": n_imgs,
            "gt_box_count": gt_b,
            "detected_box_count": det_b,
            "recall_pct": rec,
        })

    df_multibox = pd.DataFrame(multibox_summary)
    print("\nPerformance Grouped by Defect Box Count Per Image:")
    print(df_multibox.to_string(index=False, formatters={
        "recall_pct": lambda x: f"{x:.2f}%",
    }))

    # =========================================================================
    # 6. VISUAL ERROR SAMPLES
    # =========================================================================
    print("\n" + "=" * 75)
    print("6. GENERATING DIAGNOSTIC VISUALIZATION SAMPLES")
    print("=" * 75)

    OUTPUT_VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Categories of samples to select:
    # A. Correct detection (TP with high IoU >= 0.7)
    # B. Missed defect (FN: 0 predictions or no overlap)
    # C. False positive (FP on normal image)
    # D. Poor localization / low IoU (0.1 <= IoU < 0.5)
    # E. Multiple-defect image (2+ defects)

    samples_to_draw = []

    # Find candidates
    for i, m in enumerate(manifest):
        preds = base_preds[i]
        gt = m["gt_boxes"]

        # Case C: False positive on normal image
        if not m["is_defective"] and len(preds) > 0:
            samples_to_draw.append(("C_false_positive", m, preds, "False Positive on Normal Image"))

        # Defective image cases
        if m["is_defective"]:
            # Evaluate matches
            ious = []
            for g in gt:
                best_i = max([box_iou(p["box_norm"], g["box"]) for p in preds], default=0.0)
                ious.append(best_i)

            # Case A: Correct detection (1 box, IoU >= 0.75, conf >= 0.6)
            if len(gt) == 1 and len(preds) >= 1 and ious[0] >= 0.75 and preds[0]["conf"] >= 0.6:
                if sum(1 for s in samples_to_draw if s[0] == "A_correct_detection") < 4:
                    samples_to_draw.append(("A_correct_detection", m, preds, f"Correct Detection (IoU={ious[0]:.2f})"))

            # Case B: Missed defect (0 predictions on defective image)
            if len(preds) == 0 and len(gt) == 1:
                if sum(1 for s in samples_to_draw if s[0] == "B_missed_defect") < 4:
                    samples_to_draw.append(("B_missed_defect", m, preds, "Missed Defect (Zero Predictions)"))

            # Case D: Poor localization (prediction exists, but 0.15 <= best IoU < 0.45)
            if len(gt) >= 1 and len(preds) >= 1 and 0.15 <= max(ious) < 0.45:
                if sum(1 for s in samples_to_draw if s[0] == "D_poor_localization") < 4:
                    samples_to_draw.append(("D_poor_localization", m, preds, f"Poor Localization (IoU={max(ious):.2f})"))

            # Case E: Multi-defect image
            if len(gt) >= 3 and len(preds) >= 1:
                if sum(1 for s in samples_to_draw if s[0] == "E_multi_defect") < 4:
                    samples_to_draw.append(("E_multi_defect", m, preds, f"Multi-Defect ({len(gt)} GT boxes, {len(preds)} Preds)"))

    print(f"Selected {len(samples_to_draw)} diagnostic visualization samples.")

    for case_type, m, preds, desc in samples_to_draw:
        img = cv2.imread(str(m["img_path"]))
        h, w = img.shape[:2]

        # Draw Ground Truth in GREEN
        for g in m["gt_boxes"]:
            x1 = int(round(g["box"][0] * w))
            y1 = int(round(g["box"][1] * h))
            x2 = int(round(g["box"][2] * w))
            y2 = int(round(g["box"][3] * h))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "GT", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Predictions in RED/ORANGE
        for p in preds:
            b = p["box_px"]
            conf_val = p["conf"]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            label_text = f"Pred {conf_val:.2f}"
            cv2.putText(img, label_text, (b[0], min(h - 8, b[3] + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Annotate description header
        header_text = f"[{m['category']}/{m['defect_type']}] {desc}"
        cv2.putText(img, header_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3)
        cv2.putText(img, header_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)

        out_name = f"{case_type}_{m['stem']}.png"
        out_path = OUTPUT_VIS_DIR / out_name
        cv2.imwrite(str(out_path), img)
        print(f" - Saved: {out_name}")

    print("\nVisual samples saved to: " + str(OUTPUT_VIS_DIR))
    print("=" * 75)


if __name__ == "__main__":
    main()
