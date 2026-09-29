class SeverityScorer:
    """Calculates overall severity score (0-100) and defect quality decisions."""

    SIZE_WEIGHT = 0.30
    LOCATION_WEIGHT = 0.25
    DEFECT_TYPE_WEIGHT = 0.25
    CONFIDENCE_WEIGHT = 0.20

    def calculate(
        self,
        size_score: float,
        location_score: float,
        defect_type_score: float,
        confidence_score: float,
    ) -> float:
        scores = [size_score, location_score, defect_type_score, confidence_score]
        for score in scores:
            if not 0 <= score <= 100:
                raise ValueError("All severity components must be between 0 and 100.")

        severity_score = (
            size_score * self.SIZE_WEIGHT
            + location_score * self.LOCATION_WEIGHT
            + defect_type_score * self.DEFECT_TYPE_WEIGHT
            + confidence_score * self.CONFIDENCE_WEIGHT
        )
        return float(max(0.0, min(100.0, severity_score)))

    def get_severity_level(self, severity_score: float) -> str:
        if not 0 <= severity_score <= 100:
            raise ValueError("Severity score must be between 0 and 100.")

        if severity_score >= 80:
            return "Critical"
        if severity_score >= 60:
            return "High"
        if severity_score >= 40:
            return "Medium"
        return "Low"

    def get_quality_decision(self, severity_score: float, is_anomalous: bool = True) -> str:
        if not is_anomalous:
            return "Accept"

        severity_level = self.get_severity_level(severity_score)
        if severity_level in {"Medium", "High", "Critical"}:
            return "Reject"
        return "Accept"