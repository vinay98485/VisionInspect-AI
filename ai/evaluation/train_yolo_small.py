from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = PROJECT_ROOT / "datasets" / "mvtec_yolo" / "data.yaml"

OUTPUT_DIR = PROJECT_ROOT / "ai" / "weights" / "yolo"


def main():
    print("=" * 60)
    print("VisionInspect AI - YOLO11s Baseline Training")
    print("=" * 60)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_YAML}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATA_YAML),

        epochs=50,
        imgsz=640,
        batch=16,

        seed=42,
        device="mps",

        project=str(OUTPUT_DIR),
        name="small",
        exist_ok=True,

        patience=15,
        save=True,
        workers=2,
        verbose=True,
    )

    best_weights = (
        OUTPUT_DIR
        / "small"
        / "weights"
        / "best.pt"
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if best_weights.exists():
        print(f"Best model: {best_weights}")
    else:
        print("WARNING: best.pt was not found.")


if __name__ == "__main__":
    main()