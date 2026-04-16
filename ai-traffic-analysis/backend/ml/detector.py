"""
AI Vehicle Detection & Violation Analysis Module
Uses YOLOv8 for object detection with OpenCV for frame processing.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from typing import Optional

from ml.lane_detector import (
    count_vehicles_per_lane,
    compute_signal_timing,
    detect_emergency_vehicles,
    compute_traffic_alert,
    annotate_lanes,
)

# COCO class IDs mapped to traffic-relevant vehicle types
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "bike",       # motorcycle in COCO
    5: "bus",
    7: "truck",
    1: "bike",       # bicycle
}

# Extended COCO IDs that may appear in larger models
EMERGENCY_CLASS_MAP = {
    # These IDs are not in standard COCO yolov8n but included for custom model support
}


# Colors per vehicle type (BGR)
CLASS_COLORS = {
    "car":    (0, 255, 0),
    "bike":   (255, 165, 0),
    "bus":    (0, 0, 255),
    "truck":  (128, 0, 128),
    "auto":   (0, 255, 255),
    "unknown": (200, 200, 200),
}

VIOLATION_COLOR = (0, 0, 255)  # Red for violations


class TrafficDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Load YOLOv8 model. Downloads automatically on first run."""
        self.model = YOLO(model_path)
        self.conf_threshold = 0.4
        self.iou_threshold = 0.45

    def detect_vehicles(self, frame: np.ndarray) -> list:
        """
        Run YOLOv8 inference on a single frame.
        Returns detections with bounding boxes, labels, and confidence.
        Also captures raw class names for emergency vehicle detection.
        """
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)[0]
        detections = []
        names = results.names  # {id: class_name}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            raw_label = names.get(cls_id, "").lower()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Map to our vehicle types or use raw label for emergency detection
            if cls_id in VEHICLE_CLASS_MAP:
                label = VEHICLE_CLASS_MAP[cls_id]
            elif any(e in raw_label for e in ["ambulance", "fire", "police"]):
                label = raw_label
            else:
                continue

            detections.append({
                "label": label,
                "raw_label": raw_label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

        return detections

    def check_violations(self, detections: list, frame: np.ndarray, frame_idx: int = 0) -> list:
        """
        Rule-based violation detection on top of YOLO detections.
        Checks: overcrowding on bikes, density-based congestion.
        """
        violations = []
        h, w = frame.shape[:2]

        bikes = [d for d in detections if d["label"] == "bike"]
        total = len(detections)

        # Overcrowding: multiple detections in very close proximity on a bike
        for bike in bikes:
            bx1, by1, bx2, by2 = bike["bbox"]
            bike_area = (bx2 - bx1) * (by2 - by1)
            # Heuristic: if bike bounding box is unusually tall, flag overcrowding
            if (by2 - by1) > (bx2 - bx1) * 1.8:
                violations.append({
                    "type": "overcrowding_two_wheeler",
                    "confidence": 0.72,
                    "frame": frame_idx,
                    "bbox": bike["bbox"],
                    "description": "Possible overcrowding detected on two-wheeler",
                })

        # Wrong lane / congestion heuristic: too many vehicles in bottom half
        bottom_half = [
            d for d in detections
            if d["bbox"][3] > h * 0.5
        ]
        if len(bottom_half) > 8:
            violations.append({
                "type": "high_congestion",
                "confidence": 0.85,
                "frame": frame_idx,
                "bbox": None,
                "description": f"High traffic congestion detected ({len(bottom_half)} vehicles in zone)",
            })

        return violations

    def annotate_frame(self, frame: np.ndarray, detections: list, violations: list) -> np.ndarray:
        """Draw bounding boxes, labels, and violation highlights on frame."""
        annotated = frame.copy()
        violation_bboxes = {
            tuple(v["bbox"]) for v in violations if v.get("bbox")
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["confidence"]
            color = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])

            # Highlight violation boxes in red
            if (x1, y1, x2, y2) in violation_bboxes:
                color = VIOLATION_COLOR

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Violation overlay banner
        if violations:
            cv2.rectangle(annotated, (0, 0), (frame.shape[1], 36), (0, 0, 180), -1)
            cv2.putText(annotated, f"VIOLATIONS DETECTED: {len(violations)}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def count_vehicles(self, detections: list) -> dict:
        counts = {}
        for d in detections:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        return counts

    def get_density_level(self, total: int) -> str:
        if total <= 5:
            return "Low"
        elif total <= 15:
            return "Medium"
        return "High"


def process_image(image_path: str, output_path: str, detector: TrafficDetector) -> dict:
    """Process a single image file and return analysis results including lane data."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = frame.shape[:2]
    detections = detector.detect_vehicles(frame)
    violations = detector.check_violations(detections, frame)

    # Lane analysis
    lane_counts = count_vehicles_per_lane(detections, w)
    signal_timing = compute_signal_timing(lane_counts)
    emergency_status = detect_emergency_vehicles(detections)
    total = len(detections)
    traffic_alert = compute_traffic_alert(total)

    # If emergency, override all signals to green
    if emergency_status["emergency"]:
        signal_timing = {lane: {"signal": "GREEN", "time": 45} for lane in signal_timing}

    annotated = detector.annotate_frame(frame, detections, violations)
    annotated = annotate_lanes(annotated, lane_counts, signal_timing, emergency_status)
    if not cv2.imwrite(output_path, annotated):
        raise ValueError(f"Cannot write processed image: {output_path}")

    counts = detector.count_vehicles(detections)

    return {
        "total_vehicles": total,
        "vehicle_counts": counts,
        "violations": violations,
        "density_level": detector.get_density_level(total),
        "processed_path": output_path,
        "lane_counts": lane_counts,
        "signal_timing": signal_timing,
        "emergency_status": emergency_status,
        "traffic_alert": traffic_alert,
    }


def process_video(video_path: str, output_path: str, detector: TrafficDetector,
                  sample_every: int = 5) -> dict:
    """
    Process a video file frame by frame.
    sample_every: analyze every Nth frame to balance speed vs accuracy.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create processed video: {output_path}")

    all_violations = []
    cumulative_counts: dict = {}
    cumulative_lane_counts = {f"lane{i}": 0 for i in range(1, 5)}
    frame_idx = 0
    last_detections = []
    last_violations = []
    last_lane_counts = {f"lane{i}": 0 for i in range(1, 5)}
    last_signal_timing = compute_signal_timing(last_lane_counts)
    last_emergency = {"emergency": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            last_detections = detector.detect_vehicles(frame)
            last_violations = detector.check_violations(last_detections, frame, frame_idx)
            all_violations.extend(last_violations)
            for label, cnt in detector.count_vehicles(last_detections).items():
                cumulative_counts[label] = max(cumulative_counts.get(label, 0), cnt)

            last_lane_counts = count_vehicles_per_lane(last_detections, width)
            for lane, cnt in last_lane_counts.items():
                cumulative_lane_counts[lane] = max(cumulative_lane_counts[lane], cnt)

            last_emergency = detect_emergency_vehicles(last_detections)
            last_signal_timing = compute_signal_timing(last_lane_counts)
            if last_emergency["emergency"]:
                last_signal_timing = {lane: {"signal": "GREEN", "time": 45} for lane in last_signal_timing}

        annotated = detector.annotate_frame(frame, last_detections, last_violations)
        annotated = annotate_lanes(annotated, last_lane_counts, last_signal_timing, last_emergency)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    total = sum(cumulative_counts.values())
    final_signal_timing = compute_signal_timing(cumulative_lane_counts)
    final_emergency = last_emergency
    if final_emergency.get("emergency"):
        final_signal_timing = {lane: {"signal": "GREEN", "time": 45} for lane in final_signal_timing}

    return {
        "total_vehicles": total,
        "vehicle_counts": cumulative_counts,
        "violations": all_violations[:50],
        "density_level": detector.get_density_level(total),
        "processed_path": output_path,
        "frames_processed": frame_idx,
        "lane_counts": cumulative_lane_counts,
        "signal_timing": final_signal_timing,
        "emergency_status": final_emergency,
        "traffic_alert": compute_traffic_alert(total),
    }
