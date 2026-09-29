from pathlib import Path

from ultralytics import YOLO


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# YOLO dataset configuration        
DATA_YAML = PROJECT_ROOT / "datasets" / "mvtec_yolo" / "data.yaml"

# Training output
OUTPUT_DIR = PROJECT_ROOT / "ai" / "weights" / "yolo"


def main():
    print("=" * 60)
    print("VisionInspect AI - YOLO Baseline Training")
    print("=" * 60)

    print(f"Dataset: {DATA_YAML}")
    print(f"Output:  {OUTPUT_DIR}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Lightweight YOLO baseline
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_YAML),

        # Training configuration
        epochs=50,
        imgsz=640,
        batch=16,

        # Reproducibility
        seed=42,

        # Use Apple Silicon GPU through MPS
        device="mps",

        # Training output
        project=str(OUTPUT_DIR),
        name="baseline",
        exist_ok=True,

        # Early stopping
        patience=15,

        # Save checkpoints
        save=True,

        # Workers
        workers=2,

        # Verbose training output
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Results: {results}")

    best_weights = OUTPUT_DIR / "baseline" / "weights" / "best.pt"

    if best_weights.exists():
        print(f"\nBest model: {best_weights}")
    else:
        print("\nWARNING: best.pt was not found.")


if __name__ == "__main__":
    main() 