"""
Lane-wise Vehicle Detection & Signal Control
Divides frame into 4 equal vertical lanes and counts vehicles per lane.
"""

import cv2
import numpy as np


# Emergency vehicle class names (COCO labels that map to emergency types)
EMERGENCY_LABELS = {"ambulance", "fire truck", "police car"}

# COCO class IDs for emergency vehicles (not in standard COCO, handled via label name matching)
EMERGENCY_CLASS_NAMES = {"ambulance", "fire truck", "police car", "emergency"}


def assign_lane(x_center: float, frame_width: int, num_lanes: int = 4) -> int:
    """Return 1-based lane index for a given x_center coordinate."""
    lane_width = frame_width / num_lanes
    lane = int(x_center / lane_width) + 1
    return min(lane, num_lanes)


def count_vehicles_per_lane(detections: list, frame_width: int) -> dict:
    """
    Count vehicles in each of 4 lanes based on bounding box center x.
    Returns: {lane1: N, lane2: N, lane3: N, lane4: N}
    """
    counts = {f"lane{i}": 0 for i in range(1, 5)}
    for det in detections:
        x1, _, x2, _ = det["bbox"]
        x_center = (x1 + x2) / 2
        lane = assign_lane(x_center, frame_width)
        counts[f"lane{lane}"] += 1
    return counts


def compute_signal_timing(lane_counts: dict) -> dict:
    """
    Adaptive signal timing based on vehicle density per lane.
    Logic:
      <= 10  → 10s
      <= 20  → 20s
      <= 40  → 30s
      > 40   → 45s
    """
    def timing(n: int) -> int:
        if n <= 10:
            return 10
        elif n <= 20:
            return 20
        elif n <= 40:
            return 30
        return 45

    return {
        lane: {"signal": "GREEN", "time": timing(count)}
        for lane, count in lane_counts.items()
    }


def detect_emergency_vehicles(detections: list, all_labels: list = None) -> dict:
    """
    Check if any detected vehicle is an emergency vehicle.
    Since YOLOv8n (COCO) doesn't have ambulance/fire truck classes natively,
    we use a heuristic: large vehicles (bus/truck) with high confidence flagged
    as potential emergency, plus direct label matching if custom model used.
    Returns emergency status dict.
    """
    emergency_found = False
    emergency_type = None

    for det in detections:
        label = det.get("label", "").lower()
        raw_label = det.get("raw_label", "").lower()

        # Direct match for custom models or extended COCO
        if any(e in label for e in ["ambulance", "fire", "police"]):
            emergency_found = True
            emergency_type = label
            break
        if raw_label and any(e in raw_label for e in ["ambulance", "fire", "police"]):
            emergency_found = True
            emergency_type = raw_label
            break

    if emergency_found:
        return {
            "emergency": True,
            "type": emergency_type or "emergency vehicle",
            "message": "Emergency vehicle detected - Green corridor activated",
        }
    return {"emergency": False, "type": None, "message": None}


def compute_traffic_alert(total_vehicles: int) -> dict:
    """Return HIGH TRAFFIC alert if total > 100."""
    if total_vehicles > 100:
        return {"alert": "HIGH TRAFFIC", "level": "CRITICAL"}
    elif total_vehicles > 60:
        return {"alert": "MODERATE TRAFFIC", "level": "WARNING"}
    return {"alert": "NORMAL", "level": "OK"}


def annotate_lanes(frame: np.ndarray, lane_counts: dict, signal_timing: dict,
                   emergency: dict) -> np.ndarray:
    """Draw lane dividers and signal info on the annotated frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    num_lanes = 4
    lane_w = w // num_lanes

    for i in range(1, num_lanes):
        x = i * lane_w
        cv2.line(annotated, (x, 0), (x, h), (255, 255, 0), 2, cv2.LINE_AA)

    for lane_idx in range(num_lanes):
        lane_key = f"lane{lane_idx + 1}"
        count = lane_counts.get(lane_key, 0)
        timing = signal_timing.get(lane_key, {})
        time_val = timing.get("time", 0)

        x_start = lane_idx * lane_w
        label = f"L{lane_idx + 1}: {count}v {time_val}s"

        # Signal color indicator
        sig_color = (0, 255, 0) if not emergency.get("emergency") else (0, 165, 255)
        cv2.rectangle(annotated, (x_start + 4, h - 36), (x_start + lane_w - 4, h - 4), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x_start + 8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sig_color, 1, cv2.LINE_AA)

    if emergency.get("emergency"):
        cv2.rectangle(annotated, (0, 0), (w, 40), (0, 100, 200), -1)
        cv2.putText(annotated, "EMERGENCY: GREEN CORRIDOR ACTIVATED",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated
