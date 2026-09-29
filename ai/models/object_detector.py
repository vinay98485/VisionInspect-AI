from pathlib import Path
from typing import Any, Dict, List, Union
import numpy as np
import torch
from ultralytics import YOLO


class ObjectDetector:
    """
    Ultralytics YOLO11n object detector module for VisionInspect AI.
    Runs standalone object-level defect bounding box detection.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = None,
        default_conf: float = 0.25,
        default_iou: float = 0.70,
        default_imgsz: int = 640,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at: {self.model_path.resolve()}")

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.default_conf = default_conf
        self.default_iou = default_iou
        self.default_imgsz = default_imgsz

        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf: float = None,
        iou: float = None,
        imgsz: int = None,
    ) -> Dict[str, Any]:
        """
        Run object detection on an image input.

        Args:
            image_input: Filepath string/Path or numpy array.
            conf: Confidence cutoff (defaults to 0.25).
            iou: NMS IoU threshold (defaults to 0.70).
            imgsz: Input resolution (defaults to 640).

        Returns:
            Dict containing:
                "detected_objects_count": int,
                "bounding_boxes": List of dicts with absolute pixel coordinates [x1, y1, x2, y2],
                                  confidence, class_id, and class_name.
        """
        conf_val = self.default_conf if conf is None else conf
        iou_val = self.default_iou if iou is None else iou
        imgsz_val = self.default_imgsz if imgsz is None else imgsz

        source = str(image_input) if isinstance(image_input, Path) else image_input

        results = self.model.predict(
            source=source,
            conf=conf_val,
            iou=iou_val,
            imgsz=imgsz_val,
            device=self.device,
            verbose=False,
        )

        bounding_boxes: List[Dict[str, Any]] = []

        if len(results) > 0:
            res = results[0]
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy()

                for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                    bounding_boxes.append({
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2),
                        "confidence": round(float(c), 4),
                        "class_id": int(cid),
                        "class_name": "defect",
                    })

        return {
            "detected_objects_count": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
        }
