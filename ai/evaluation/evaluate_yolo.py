from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "datasets" / "mvtec_yolo" / "data.yaml"

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "weights"
    / "yolo"
    / "small"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "weights"
    / "yolo"
    / "evaluation_small"
)


def main():
    print("=" * 60)
    print("VisionInspect AI - YOLO11s Test Evaluation")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {DATA_YAML}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO11s weights not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATA_YAML}")
    print("Split:   test")
    print("Device:  mps")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=16,
        device="mps",
        project=str(OUTPUT_DIR),
        name="test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("YOLO11s TEST SET RESULTS")
    print("=" * 60)

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")

    print("\nResults saved to:")
    print(OUTPUT_DIR / "test")


if __name__ == "__main__":
    main()