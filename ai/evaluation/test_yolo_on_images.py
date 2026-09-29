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

# Change this to the folder containing your 10 test images
INPUT_DIR = PROJECT_ROOT / "test_images"

OUTPUT_DIR = PROJECT_ROOT / "yolo_threshold_test"

CONFIDENCES = [0.25, 0.10, 0.05]


def main():
    model = YOLO(str(MODEL_PATH))

    images = [
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]

    if not images:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    print(f"Found {len(images)} images")

    for conf in CONFIDENCES:

        output_dir = OUTPUT_DIR / f"conf_{conf:.2f}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"CONFIDENCE = {conf}")
        print("=" * 60)

        total_boxes = 0

        for image_path in images:

            results = model.predict(
                source=str(image_path),
                imgsz=640,
                conf=conf,
                device="mps",
                verbose=False,
                save=False,
            )

            result = results[0]
            boxes = result.boxes

            count = len(boxes)
            total_boxes += count

            print(f"{image_path.name}: {count} boxes")

            # Save annotated image
            annotated = result.plot()

            save_path = output_dir / image_path.name

            # Use OpenCV to save the image
            import cv2
            cv2.imwrite(str(save_path), annotated)

        print(f"\nTotal boxes at conf {conf}: {total_boxes}")
        print(f"Saved images to: {output_dir}")


if __name__ == "__main__":
    main()