from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "weights"
    / "yolo"
    / "baseline"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    PROJECT_ROOT
    / "datasets"
    / "mvtec_yolo"
    / "data.yaml"
)


def main():
    model = YOLO(str(MODEL_PATH))

    confidence_levels = [0.25, 0.20, 0.15, 0.10, 0.05]

    for conf in confidence_levels:
        print("\n" + "=" * 60)
        print(f"TESTING CONFIDENCE = {conf}")
        print("=" * 60)

        results = model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=640,
            batch=16,
            conf=conf,
            device="mps",
            workers=2,
            verbose=False,
        )

        print(f"Confidence: {conf}")
        print(f"Precision:  {results.box.mp:.4f}")
        print(f"Recall:     {results.box.mr:.4f}")
        print(f"mAP50:      {results.box.map50:.4f}")
        print(f"mAP50-95:   {results.box.map:.4f}")


if __name__ == "__main__":
    main()