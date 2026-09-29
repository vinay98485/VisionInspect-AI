from pathlib import Path

from ultralytics import YOLO


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# YOLO dataset configuration
DATA_YAML = PROJECT_ROOT / "datasets" / "mvtec_yolo" / "data.yaml"

# Existing trained model
BASE_MODEL = PROJECT_ROOT / "ai" / "weights" / "yolo" / "baseline" / "weights" / "best.pt"

# Training output
OUTPUT_DIR = PROJECT_ROOT / "ai" / "weights" / "yolo"


def main():
    print("=" * 60)
    print("VisionInspect AI - YOLO11n Fine-Tuning")
    print("=" * 60)

    print(f"Dataset:    {DATA_YAML}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output:     {OUTPUT_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    if not BASE_MODEL.exists():
        raise FileNotFoundError(
            f"Base model not found: {BASE_MODEL}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Start from our existing best YOLO11n model
    model = YOLO(str(BASE_MODEL))

    results = model.train(
        data=str(DATA_YAML),

        # Short fine-tuning experiment
        epochs=30,
        imgsz=640,
        batch=16,

        # Reproducibility
        seed=42,

        # Apple Silicon GPU
        device="mps",

        # Separate output from baseline
        project=str(OUTPUT_DIR),
        name="fine_tuned",
        exist_ok=True,

        # Stop early if validation stops improving
        patience=8,

        # Save checkpoints
        save=True,

        # Workers
        workers=2,

        verbose=True,
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)

    print(f"Results: {results}")

    best_weights = (
        OUTPUT_DIR
        / "fine_tuned"
        / "weights"
        / "best.pt"
    )

    if best_weights.exists():
        print(f"\nFine-tuned best model: {best_weights}")
    else:
        print("\nWARNING: best.pt was not found.")


if __name__ == "__main__":
    main()