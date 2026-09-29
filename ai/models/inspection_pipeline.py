import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from ai.models.anomaly_detector import AnomalyDetector
from ai.models.category_classifier import CategoryInference
from ai.models.confidence import DetectionConfidence
from ai.models.decision_fusion import DecisionFusionLayer
from ai.models.defect_classifier import DefectInference
from ai.models.defect_segmenter import UNet
from ai.models.defect_type_scorer import DefectTypeScorer
from ai.models.location_scorer import LocationScorer
from ai.models.object_detector import ObjectDetector
from ai.models.severity_scorer import SeverityScorer
from ai.models.size_scorer import SizeScorer

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InspectionPipeline:

    def __init__(
        self,
        normal_features_path: str,
        confidence_scores_path: str,
        size_boundaries_path: str,
        segmentation_model_path: str,
        segmentation_threshold_path: str,
        anomaly_thresholds_path: str = None,
        postprocessing_config_path: str = None,
        category_classifier_path: str = None,
        category_classes_path: str = None,
        defect_classifier_path: str = None,
        defect_classes_path: str = None,
        object_detector_path: str = None,
    ):
        self.segmentation_model = UNet(
            in_channels=3,
            out_channels=1,
        )

        self.segmentation_device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        checkpoint = torch.load(
            segmentation_model_path,
            map_location=self.segmentation_device,
            weights_only=True,
        )

        self.segmentation_model.load_state_dict(checkpoint)

        self.segmentation_model = (
            self.segmentation_model
            .to(self.segmentation_device)
        )

        self.segmentation_model.eval()

        with open(
            segmentation_threshold_path,
            "r",
        ) as file:
            self.segmentation_threshold = float(
                file.read().strip()
            )

        self.anomaly_detector = AnomalyDetector(
            normal_features_path
        )

        self.location_scorer = LocationScorer()

        self.defect_type_scorer = DefectTypeScorer()

        self.severity_scorer = SeverityScorer()

        self.size_scorer = SizeScorer(
            self._load_size_boundaries(
                size_boundaries_path
            )
        )

        # Default paths if not explicitly provided
        base_dir = Path(segmentation_threshold_path).parent

        if anomaly_thresholds_path is None:
            anomaly_thresholds_path = str(base_dir / "anomaly_thresholds.csv")

        if postprocessing_config_path is None:
            postprocessing_config_path = str(base_dir / "segmentation_postprocessing.json")

        if category_classifier_path is None:
            default_cat = base_dir / "category_classifier_resnet18.pt"
            category_classifier_path = str(default_cat) if default_cat.exists() else None

        if category_classes_path is None:
            default_cat_cls = base_dir / "category_classes.json"
            category_classes_path = str(default_cat_cls) if default_cat_cls.exists() else None

        if defect_classifier_path is None:
            default_def = base_dir / "defect_classifier_hierarchical.pt"
            defect_classifier_path = str(default_def) if default_def.exists() else None

        if defect_classes_path is None:
            default_def_cls = base_dir / "defect_classes.json"
            defect_classes_path = str(default_def_cls) if default_def_cls.exists() else None

        self.category_classifier = None
        if (
            category_classifier_path
            and category_classes_path
            and Path(category_classifier_path).exists()
            and Path(category_classes_path).exists()
        ):
            self.category_classifier = CategoryInference(
                model_path=category_classifier_path,
                classes_path=category_classes_path,
                device=self.segmentation_device,
            )

        self.defect_classifier = None
        if (
            defect_classifier_path
            and defect_classes_path
            and Path(defect_classifier_path).exists()
            and Path(defect_classes_path).exists()
        ):
            self.defect_classifier = DefectInference(
                model_path=defect_classifier_path,
                defect_classes_path=defect_classes_path,
                device=self.segmentation_device,
            )

        if object_detector_path is None:
            default_yolo = PROJECT_ROOT / "ai/weights/yolo/baseline/weights/best.pt"
            if default_yolo.exists():
                object_detector_path = str(default_yolo)

        self.object_detector = None
        if object_detector_path and Path(object_detector_path).exists():
            self.object_detector = ObjectDetector(
                model_path=object_detector_path,
                device=str(self.segmentation_device),
                default_conf=0.25,
                default_iou=0.70,
                default_imgsz=640,
            )

        self.anomaly_thresholds = self._load_anomaly_thresholds(
            anomaly_thresholds_path
        )

        # Initialize validation-derived decision fusion layer
        simple_thresholds = {
            cat: data["anomaly_threshold"]
            for cat, data in self.anomaly_thresholds.items()
            if "anomaly_threshold" in data
        }
        self.decision_fusion = DecisionFusionLayer(simple_thresholds)

        # Initialize calibrated confidence scorer using validated thresholds
        self.confidence_scorer = DetectionConfidence(
            scores_path=confidence_scores_path,
            anomaly_thresholds_path=anomaly_thresholds_path,
        )

        self._load_postprocessing_config(
            postprocessing_config_path
        )

    def _load_size_boundaries(
        self,
        path: str,
    ):
        dataframe = pd.read_csv(path)

        boundaries = {}

        for _, row in dataframe.iterrows():
            boundaries[row["category"]] = {
                "p10": float(row["p10"]),
                "p25": float(row["p25"]),
                "p50": float(row["p50"]),
                "p75": float(row["p75"]),
                "p90": float(row["p90"]),
                "p95": float(row["p95"]),
            }

        return boundaries

    def _load_anomaly_thresholds(
        self,
        path: str,
    ):
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Anomaly thresholds CSV file not found at: {path_obj.resolve()}")

        df = pd.read_csv(path_obj)
        required_cols = {"category", "anomaly_threshold"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Missing required columns in {path_obj}: {required_cols - set(df.columns)}")

        thresholds = {}
        for _, row in df.iterrows():
            cat = str(row["category"]).strip().lower()
            thresholds[cat] = {
                "normal_boundary": float(row["normal_boundary"]) if "normal_boundary" in row else float(row["anomaly_threshold"]),
                "defect_boundary": float(row["defect_boundary"]) if "defect_boundary" in row else float(row["anomaly_threshold"]) * 1.25,
                "anomaly_threshold": float(row["anomaly_threshold"]),
            }
        return thresholds

    def _load_postprocessing_config(
        self,
        path: str,
    ):
        path_obj = Path(path)
        if path_obj.exists():
            with open(path_obj, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.morph_kernel_size = int(config.get("morph_kernel_size", 3))
            self.min_component_area = int(config.get("min_component_area", 10))
        else:
            self.morph_kernel_size = 3
            self.min_component_area = 10

    def _get_category(
        self,
        image_path: str,
    ) -> str:
        path = Path(image_path)
        return path.parent.parent.parent.name

    def _calculate_predicted_area(
        self,
        heatmap: np.ndarray,
        segmentation_mask: np.ndarray,
    ) -> float:
        if segmentation_mask is None:
            return 0.0

        defect_pixels = np.sum(
            segmentation_mask > 0
        )

        total_pixels = segmentation_mask.size

        return float(
            defect_pixels
            / total_pixels
            * 100.0
        )

    def _postprocess_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Post-process binary defect mask:
        1. Morphological opening to eliminate isolated pixel noise.
        2. Minimum connected-component area filtering to prune non-defect specks.
        """
        if mask is None or np.sum(mask > 0) == 0:
            return np.zeros((224, 224), dtype=np.uint8)

        # Morphological opening
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        # Connected-component filtering
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
        cleaned = np.zeros_like(opened)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_component_area:
                cleaned[labels == i] = 1

        return cleaned

    def _is_anomalous(
        self,
        category: str,
        anomaly_score: float,
        confidence_score: float,
    ) -> bool:
        """
        Data-derived anomaly decision gate:
        An image is considered anomalous if:
        1. anomaly_score >= category_threshold (derived from validation set)
        2. confidence_score > 0.0%
        3. anomaly_score > normal_boundary (95th percentile of normal validation samples)

        If any condition fails, the image belongs to the normal distribution.
        """
        if category in self.anomaly_thresholds:
            normal_boundary = self.anomaly_thresholds[category]["normal_boundary"]
            threshold = self.anomaly_thresholds[category]["anomaly_threshold"]
        else:
            normal_boundary = self.confidence_scorer.boundaries.get(
                category, {}
            ).get("normal_boundary", 0.0035)
            threshold = normal_boundary

        if (
            confidence_score <= 0.0
            or anomaly_score <= normal_boundary
            or anomaly_score < threshold
        ):
            return False

        return True

    def segment_image(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(
                f"Could not read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )

        image = image.astype(
            np.float32
        ) / 255.0

        tensor = torch.from_numpy(
            image
        ).permute(2, 0, 1).unsqueeze(0)

        tensor = tensor.to(
            self.segmentation_device
        )

        with torch.no_grad():
            logits = self.segmentation_model(
                tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            mask = (
                probabilities
                >= self.segmentation_threshold
            )

        mask = (
            mask[0, 0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        return mask

    def predict(
        self,
        image_path: str,
        category: str = None,
        defect_type: str = None,
    ):
        # ----------------------------------------
        # 0. Machine Learning Category Classification
        # ----------------------------------------
        pred_category = None
        cat_confidence = 1.0

        if self.category_classifier is not None:
            pred_category, cat_confidence, _ = self.category_classifier.predict(
                image_input=image_path
            )

        # Resolve active category
        if category and category.strip():
            active_category = category.strip().lower()
        elif pred_category:
            active_category = pred_category
        else:
            active_category = self._get_category(image_path)

        # ----------------------------------------
        # 0b. Machine Learning Defect-Type Classification
        # ----------------------------------------
        pred_defect = "good"
        defect_confidence = 1.0

        if (
            self.defect_classifier is not None
            and active_category in self.defect_classifier.defect_classes
        ):
            pred_defect, defect_confidence, _ = self.defect_classifier.predict(
                image_input=image_path,
                category=active_category,
            )

        classification_confidence = round(float(cat_confidence * defect_confidence * 100.0), 2)

        # ----------------------------------------
        # 1. Anomaly Detection (ResNet18 Layer3)
        # ----------------------------------------
        detection_result = (
            self.anomaly_detector.predict(
                image_path
            )
        )

        anomaly_score = (
            detection_result["anomaly_score"]
        )

        heatmap = detection_result["heatmap"]

        # ----------------------------------------
        # 2. Decision Fusion Layer (Validated Quadrants A, B, C, D)
        # ----------------------------------------
        category_threshold = float(
            self.anomaly_thresholds[active_category]["anomaly_threshold"]
            if active_category in self.anomaly_thresholds
            else 1.20
        )

        above_threshold = bool(anomaly_score >= category_threshold)
        clf_is_defective = bool(pred_defect not in ["good", "normal", "none"])

        fusion_result = self.decision_fusion.evaluate(
            category=active_category,
            pred_defect=pred_defect,
            defect_confidence=defect_confidence,
            anomaly_score=anomaly_score,
            category_threshold=category_threshold,
            manual_defect_type=defect_type,
        )

        is_defective = (fusion_result.decision == "DEFECTIVE")
        resolved_defect_status = fusion_result.resolved_defect_status

        # ----------------------------------------
        # 3. Detection Confidence (Category Operating Boundaries)
        # ----------------------------------------
        if not is_defective:
            confidence_score = 0.0
        else:
            confidence_score = self.confidence_scorer.calculate(
                category=active_category,
                anomaly_score=anomaly_score,
            )
            if confidence_score <= 0.0:
                confidence_score = 50.0

        # ----------------------------------------
        # 4. Critical Normal vs Defective Invariant Branches
        # ----------------------------------------
        if not is_defective:
            # ------------------------------------------------
            # NORMAL INVARIANT: Confirmed non-defective specimen
            # ------------------------------------------------
            resolved_defect_type = "normal"
            predicted_defect_type = "normal"
            defect_type_score = 0.0
            segmentation_mask = np.zeros((224, 224), dtype=np.uint8)
            location_score = 0.0
            predicted_area = 0.0
            size_score = 0.0
            severity_score = 0.0
            severity_level = "Low"
            quality_decision = "Accept"
            detected_objects_count = 0
            bounding_boxes = []
        else:
            # ------------------------------------------------
            # DEFECTIVE INVARIANT: Run U-Net and defect analysis
            # ------------------------------------------------
            resolved_defect_type = resolved_defect_status
            predicted_defect_type = fusion_result.predicted_defect_type

            raw_mask = self.segment_image(
                image_path
            )
            segmentation_mask = self._postprocess_mask(
                raw_mask
            )

            location_score = (
                self.location_scorer.calculate(
                    segmentation_mask
                )
            )

            # Calculate physical defect area strictly from cleaned segmentation mask
            predicted_area = float(
                (np.sum(segmentation_mask > 0) / (224.0 * 224.0)) * 100.0
            )

            size_score = (
                self.size_scorer.calculate(
                    category=active_category,
                    predicted_area_percent=predicted_area,
                )
            )

            defect_type_score = (
                self.defect_type_scorer.calculate(
                    resolved_defect_type
                )
            )

            severity_score = (
                self.severity_scorer.calculate(
                    size_score=size_score,
                    location_score=location_score,
                    defect_type_score=defect_type_score,
                    confidence_score=confidence_score,
                )
            )

            severity_level = (
                self.severity_scorer.get_severity_level(
                    severity_score
                )
            )

            quality_decision = (
                self.severity_scorer.get_quality_decision(
                    severity_score=severity_score,
                    is_anomalous=True,
                )
            )

            # ------------------------------------------------
            # Object Detection (YOLO11n Bounding Boxes)
            # ------------------------------------------------
            detected_objects_count = 0
            bounding_boxes = []
            if self.object_detector is not None:
                detection_res = self.object_detector.predict(
                    image_input=image_path
                )
                detected_objects_count = detection_res["detected_objects_count"]
                bounding_boxes = detection_res["bounding_boxes"]

        return {
            "image_path": str(image_path),
            "category": active_category,
            "defect_type": resolved_defect_type,
            "predicted_category": pred_category or active_category,
            "predicted_defect_type": predicted_defect_type,
            "resolved_defect_status": resolved_defect_status,
            "inspection_decision": fusion_result.decision,
            "classification_confidence": classification_confidence,
            "anomaly_score": anomaly_score,
            "confidence_score": confidence_score,
            "predicted_area_percent": predicted_area,
            "size_score": size_score,
            "location_score": location_score,
            "defect_type_score": defect_type_score,
            "severity_score": severity_score,
            "severity_level": severity_level,
            "quality_decision": quality_decision,
            "heatmap": heatmap,
            "segmentation_mask": segmentation_mask,
            "detected_objects_count": detected_objects_count,
            "bounding_boxes": bounding_boxes,
        }