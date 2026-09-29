"""
VisionInspect AI - Industrial YOLO11n Training & Evaluation Experiment

Isolates the effects of:
- Higher resolution (imgsz=1024)
- Disabled destructive augmentations (mosaic=0.0, scale=0.0, erasing=0.0)
- Longer training schedule (epochs=100)
- Cosine learning rate schedule (cos_lr=True)
- Increased patience (patience=20)
- batch=8 for reliable MPS memory headroom at 1024x1024
- Default loss weights (no box/cls scaling)
"""

from pathlib import Path
import time
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "datasets" / "mvtec_yolo" / "data.yaml"
OUTPUT_DIR = PROJECT_ROOT / "ai" / "weights" / "yolo"
EXP_NAME = "industrial_1024"
EVAL_EXP_NAME = "evaluation_industrial_1024"
BASELINE_BEST = OUTPUT_DIR / "baseline" / "weights" / "best.pt"


def main():
    print("=" * 70)
    print("VisionInspect AI - YOLO11n Industrial-Tuned Experiment (1024x1024)")
    print("=" * 70)

    # 1. Pre-training verifications
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")
    if not BASELINE_BEST.exists():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_BEST}")

    exp_dir = OUTPUT_DIR / EXP_NAME
    print(f"Dataset configuration: {DATA_YAML}")
    print(f"Baseline weights:      {BASELINE_BEST}")
    print(f"Experiment directory:  {exp_dir}")
    print(f"Separate from baseline: {exp_dir != OUTPUT_DIR / 'baseline'}")

    print("\nTraining Configuration:")
    print(" - Model:                 yolo11n.pt")
    print(" - Image size (imgsz):    1024")
    print(" - Epochs:                100")
    print(" - Batch size:            8")
    print(" - Device:                mps")
    print(" - Seed:                  42")
    print(" - Mosaic:                0.0 (disabled)")
    print(" - Scale:                 0.0 (disabled)")
    print(" - Erasing:               0.0 (disabled)")
    print(" - Cosine LR (cos_lr):    True")
    print(" - Patience:              20")
    print(" - Save checkpoints:      True")
    print(" - Loss weights:          Default Ultralytics box & cls")

    # 2. Train model
    print("\nStarting Training...")
    start_train_time = time.time()

    model = YOLO("yolo11n.pt")

    train_results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1024,
        batch=8,
        seed=42,
        device="mps",
        project=str(OUTPUT_DIR),
        name=EXP_NAME,
        exist_ok=True,
        patience=20,
        mosaic=0.0,
        scale=0.0,
        erasing=0.0,
        cos_lr=True,
        save=True,
        workers=2,
        verbose=True,
    )

    train_duration = time.time() - start_train_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE in {train_duration:.1f}s ({train_duration / 3600:.2f} hours)")
    print("=" * 70)

    # 3. Verify best weights
    best_weights = exp_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Trained weights not found: {best_weights}")

    print(f"\nBest Model Saved At: {best_weights}")

    # 4. Evaluate on test split
    print("\n" + "=" * 70)
    print("EVALUATING NEW BEST MODEL ON OFFICIAL TEST SET (split='test', imgsz=1024)")
    print("=" * 70)

    test_model = YOLO(str(best_weights))
    eval_dir = OUTPUT_DIR / EVAL_EXP_NAME

    start_eval_time = time.time()
    metrics = test_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=1024,
        batch=8,
        device="mps",
        project=str(eval_dir),
        name="test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    eval_duration = time.time() - start_eval_time

    # 5. Extract metrics
    new_p = float(metrics.box.mp)
    new_r = float(metrics.box.mr)
    new_map50 = float(metrics.box.map50)
    new_map = float(metrics.box.map)

    # Speed metrics
    speed_info = metrics.speed  # dict with preprocess, inference, loss, postprocess in ms
    inf_speed = speed_info.get("inference", 0.0)

    # 6. Baseline metrics for comparison
    base_p = 0.5555
    base_r = 0.3995
    base_map50 = 0.4398
    base_map = 0.2019

    delta_p = new_p - base_p
    delta_r = new_r - base_r
    delta_map50 = new_map50 - base_map50
    delta_map = new_map - base_map

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENTAL COMPARISON: Baseline vs Industrial (1024)")
    print("=" * 70)
    print(f"{'Metric':<18} {'Baseline (640, 50ep)':<24} {'Industrial (1024, 100ep)':<26} {'Absolute Change':<15}")
    print("-" * 75)
    print(f"{'Precision':<18} {base_p:<24.4f} {new_p:<26.4f} {delta_p:+15.4f}")
    print(f"{'Recall':<18} {base_r:<24.4f} {new_r:<26.4f} {delta_r:+15.4f}")
    print(f"{'mAP@50':<18} {base_map50:<24.4f} {new_map50:<26.4f} {delta_map50:+15.4f}")
    print(f"{'mAP@50-95':<18} {base_map:<24.4f} {new_map:<26.4f} {delta_map:+15.4f}")
    print("-" * 75)
    print(f"Training Time:   {train_duration:.1f}s ({train_duration / 3600:.2f} hours)")
    print(f"Inference Speed: {inf_speed:.2f} ms / image (preprocess: {speed_info.get('preprocess', 0):.2f}ms, postprocess: {speed_info.get('postprocess', 0):.2f}ms)")
    print(f"Evaluation Time: {eval_duration:.1f}s")
    print(f"Test Plots Directory: {eval_dir / 'test'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
