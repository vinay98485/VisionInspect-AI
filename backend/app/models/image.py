from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.base import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, nullable=False)
    stored_filename = Column(String, nullable=False, unique=True)
    storage_path = Column(String, nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    inspection_status = Column(String, nullable=False, default="pending")
    supervisor_decision = Column(String, nullable=True)
    supervisor_notes = Column(String, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    user = relationship("User", foreign_keys=[uploaded_by])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    category = Column(String, nullable=True)
    defect_type = Column(String, nullable=True)
    predicted_category = Column(String, nullable=True)
    predicted_defect_type = Column(String, nullable=True)
    resolved_defect_status = Column(String, nullable=True)
    classification_confidence = Column(String, nullable=True)
    anomaly_score = Column(String, nullable=True)
    confidence_score = Column(String, nullable=True)
    predicted_area_percent = Column(String, nullable=True)
    size_score = Column(String, nullable=True)
    location_score = Column(String, nullable=True)
    defect_type_score = Column(String, nullable=True)
    severity_score = Column(String, nullable=True)
    severity_level = Column(String, nullable=True)
    quality_decision = Column(String, nullable=True)
    detected_objects_count = Column(Integer, nullable=True, default=0)
    bounding_boxes = Column(JSON, nullable=True)
