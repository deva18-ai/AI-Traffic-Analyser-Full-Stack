"""
Upload & Analysis API Routes
POST /api/upload          - upload image or video and run detection
POST /api/analyze-traffic - dedicated traffic analysis endpoint
GET  /api/results         - list all past detection results
GET  /api/results/{id}    - get single result
GET  /api/stats           - aggregate statistics
GET  /api/media/{file}    - serve processed media
"""

import os
import uuid
import json
import aiofiles
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
from collections import defaultdict

from database.db import get_db, DetectionResult, ViolationLog
from ml.detector import TrafficDetector, process_image, process_video

router = APIRouter(prefix="/api", tags=["analysis"])

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
PROCESSED_DIR = UPLOAD_DIR / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".mp4"}
MAX_SIZE = int(os.getenv("MAX_FILE_SIZE", 104857600))  # 100MB

_detector: TrafficDetector | None = None


def _json_value(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def _serialize_result_record(record: DetectionResult) -> dict:
    vehicle_counts = _json_value(record.vehicle_counts, {})
    violations = _json_value(record.violations, [])
    lane_counts = _json_value(record.lane_counts, {})
    signal_timing = _json_value(record.signal_timing, {})
    emergency_status = _json_value(record.emergency_status, {"emergency": False, "type": None, "message": None})
    traffic_alert = _json_value(record.traffic_alert, {"alert": "NORMAL", "level": "OK"})

    return {
        "id": record.id,
        "filename": record.filename,
        "media_type": record.media_type,
        "total_vehicles": record.total_vehicles,
        "vehicle_counts": vehicle_counts,
        "violations": violations,
        "density_level": record.density_level,
        "created_at": record.created_at.isoformat(),
        "processed_url": f"/api/media/{record.processed_path}",
        "lane_counts": lane_counts,
        "signal_timing": signal_timing,
        "emergency_status": emergency_status,
        "traffic_alert": traffic_alert,
        "violation_count": len(violations),
    }


def _aggregate_overview(records: list[DetectionResult]) -> dict:
    serialized = [_serialize_result_record(record) for record in records]
    total_analyses = len(serialized)
    total_vehicles = sum(item["total_vehicles"] for item in serialized)
    total_violations = sum(item["violation_count"] for item in serialized)

    vehicle_type_totals: dict[str, int] = {}
    density_distribution = {"Low": 0, "Medium": 0, "High": 0}
    media_distribution = {"image": 0, "video": 0}
    incidents: list[dict] = []
    flow_by_day: dict[str, dict] = defaultdict(lambda: {"analyses": 0, "vehicles": 0, "violations": 0})
    emergency_events = 0
    high_traffic_events = 0

    for item in serialized:
        for key, value in item["vehicle_counts"].items():
            vehicle_type_totals[key] = vehicle_type_totals.get(key, 0) + value

        if item["density_level"] in density_distribution:
            density_distribution[item["density_level"]] += 1

        if item["media_type"] in media_distribution:
            media_distribution[item["media_type"]] += 1

        if item["emergency_status"].get("emergency"):
            emergency_events += 1

        if item["traffic_alert"].get("level") == "CRITICAL":
            high_traffic_events += 1

        day_key = item["created_at"][:10]
        flow_by_day[day_key]["analyses"] += 1
        flow_by_day[day_key]["vehicles"] += item["total_vehicles"]
        flow_by_day[day_key]["violations"] += item["violation_count"]

        if item["violation_count"] or item["emergency_status"].get("emergency") or item["traffic_alert"].get("level") in {"WARNING", "CRITICAL"}:
            incidents.append({
                "id": item["id"],
                "filename": item["filename"],
                "created_at": item["created_at"],
                "media_type": item["media_type"],
                "traffic_alert": item["traffic_alert"],
                "emergency_status": item["emergency_status"],
                "violation_count": item["violation_count"],
                "violations": item["violations"],
                "processed_url": item["processed_url"],
            })

    recent_results = serialized[:10]
    recent_incidents = incidents[:10]
    flow_series = [
        {"date": date_key, **stats}
        for date_key, stats in sorted(flow_by_day.items())
    ][-14:]

    return {
        "totals": {
            "analyses": total_analyses,
            "vehicles": total_vehicles,
            "violations": total_violations,
            "incidents": len(incidents),
            "emergency_events": emergency_events,
            "high_traffic_events": high_traffic_events,
        },
        "vehicle_type_totals": vehicle_type_totals,
        "density_distribution": density_distribution,
        "media_distribution": media_distribution,
        "recent_results": recent_results,
        "recent_incidents": recent_incidents,
        "flow_series": flow_series,
    }


def get_detector() -> TrafficDetector:
    global _detector
    if _detector is None:
        model_path = os.getenv("MODEL_PATH", "yolov8n.pt")
        _detector = TrafficDetector(model_path)
    return _detector


async def _save_upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "Missing file name")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}. Allowed: {allowed}")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_name
    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_SIZE:
            raise HTTPException(413, "File too large")
        await f.write(content)
    media_type = "video" if ext == ".mp4" else "image"
    return save_path, unique_name, media_type


def _run_detection(save_path: Path, unique_name: str, media_type: str):
    output_name = f"processed_{unique_name}"
    output_path = str(PROCESSED_DIR / output_name)
    detector = get_detector()
    try:
        if media_type == "image":
            result = process_image(str(save_path), output_path, detector)
        else:
            result = process_video(str(save_path), output_path, detector)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")
    return result, output_name


def _persist_result(db: Session, filename: str, media_type: str, result: dict, output_name: str):
    db_record = DetectionResult(
        filename=filename,
        media_type=media_type,
        total_vehicles=result["total_vehicles"],
        vehicle_counts=result["vehicle_counts"],
        violations=result["violations"],
        density_level=result["density_level"],
        processed_path=output_name,
        lane_counts=result.get("lane_counts"),
        signal_timing=result.get("signal_timing"),
        emergency_status=result.get("emergency_status"),
        traffic_alert=result.get("traffic_alert"),
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    for v in result["violations"]:
        db.add(ViolationLog(
            detection_id=db_record.id,
            violation_type=v["type"],
            confidence=v.get("confidence", 0),
            frame_number=v.get("frame", 0),
            description=v.get("description", ""),
        ))
    db.commit()
    return db_record


def _build_response(filename: str, media_type: str, result: dict, output_name: str, record_id: int):
    return {
        "id": record_id,
        "filename": filename,
        "media_type": media_type,
        "total_vehicles": result["total_vehicles"],
        "vehicle_counts": result["vehicle_counts"],
        "violations": result["violations"],
        "density_level": result["density_level"],
        "processed_url": f"/api/media/{output_name}",
        "lane_counts": result.get("lane_counts"),
        "signal_timing": result.get("signal_timing"),
        "emergency_status": result.get("emergency_status"),
        "traffic_alert": result.get("traffic_alert"),
    }


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    save_path, unique_name, media_type = await _save_upload(file)
    result, output_name = _run_detection(save_path, unique_name, media_type)
    record = _persist_result(db, file.filename, media_type, result, output_name)
    return _build_response(file.filename, media_type, result, output_name, record.id)


@router.post("/analyze-traffic")
async def analyze_traffic(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Dedicated adaptive traffic analysis endpoint.
    Returns lane_counts, signal_timing, emergency_status, traffic_alert, total_vehicles.
    """
    save_path, unique_name, media_type = await _save_upload(file)
    result, output_name = _run_detection(save_path, unique_name, media_type)
    record = _persist_result(db, file.filename, media_type, result, output_name)
    return _build_response(file.filename, media_type, result, output_name, record.id)


@router.get("/results")
def list_results(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(DetectionResult).order_by(
        DetectionResult.created_at.desc()
    ).offset(skip).limit(limit).all()
    return [_serialize_result_record(r) for r in records]


@router.get("/results/{result_id}")
def get_result(result_id: int, db: Session = Depends(get_db)):
    record = db.query(DetectionResult).filter(DetectionResult.id == result_id).first()
    if not record:
        raise HTTPException(404, "Result not found")
    violations = db.query(ViolationLog).filter(ViolationLog.detection_id == result_id).all()
    payload = _serialize_result_record(record)
    payload["violations"] = [
        {
            "id": v.id,
            "type": v.violation_type,
            "confidence": v.confidence,
            "frame": v.frame_number,
            "description": v.description,
            "created_at": v.created_at.isoformat(),
        }
        for v in violations
    ]
    payload["violation_count"] = len(payload["violations"])
    return payload


@router.get("/media/{filename}")
def serve_media(filename: str):
    file_path = PROCESSED_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Aggregate stats for dashboard overview."""
    all_records = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).all()
    overview = _aggregate_overview(all_records)

    return {
        "total_analyses": overview["totals"]["analyses"],
        "total_vehicles_detected": overview["totals"]["vehicles"],
        "total_violations": overview["totals"]["violations"],
        "vehicle_type_totals": overview["vehicle_type_totals"],
        "density_distribution": overview["density_distribution"],
        "emergency_events": overview["totals"]["emergency_events"],
        "high_traffic_events": overview["totals"]["high_traffic_events"],
    }


@router.get("/overview")
def get_overview(db: Session = Depends(get_db)):
    records = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).all()
    return _aggregate_overview(records)


@router.delete("/results")
def clear_all_results(db: Session = Depends(get_db)):
    """Delete all detection results and violation logs."""
    db.query(ViolationLog).delete()
    db.query(DetectionResult).delete()
    db.commit()
    return {"message": "All records cleared successfully"}


@router.get("/incidents")
def list_incidents(limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(DetectionResult).order_by(DetectionResult.created_at.desc()).all()
    overview = _aggregate_overview(records)
    return overview["recent_incidents"][:limit]


@router.post("/chat")
async def chat_with_codex(payload: dict):
    """
    AI assistant endpoint for contextual traffic analysis.
    In production, this would integrate with the Gemini API to analyze database trends.
    """
    message = payload.get("message", "")
    # Simulated contextual response based on the system state
    return {
        "response": f"Analyzing traffic context... Regarding '{message}', I've reviewed the latest results and noticed Lane 2 density is consistently 'High'. I suggest increasing the adaptive green signal time to 45s for that zone to alleviate the bottleneck."
    }
