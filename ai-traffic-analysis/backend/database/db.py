from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./traffic.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # image | video
    total_vehicles = Column(Integer, default=0)
    vehicle_counts = Column(JSON)       # {"car": 3, "bike": 2, ...}
    violations = Column(JSON)           # list of violation dicts
    density_level = Column(String)      # Low | Medium | High
    processed_path = Column(String)     # path to annotated output
    lane_counts = Column(JSON)          # {lane1: N, lane2: N, ...}
    signal_timing = Column(JSON)        # {lane1: {signal, time}, ...}
    emergency_status = Column(JSON)     # {emergency: bool, message: str}
    traffic_alert = Column(JSON)        # {alert: str, level: str}
    created_at = Column(DateTime, default=datetime.utcnow)


class ViolationLog(Base):
    __tablename__ = "violation_logs"

    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, nullable=False)
    violation_type = Column(String, nullable=False)
    confidence = Column(Float)
    frame_number = Column(Integer, default=0)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    # Add new columns to existing tables if they don't exist (SQLite migration)
    _migrate_add_columns()


def _migrate_add_columns():
    """Add new columns to existing tables without dropping data."""
    new_columns = [
        ("detection_results", "lane_counts", "TEXT"),
        ("detection_results", "signal_timing", "TEXT"),
        ("detection_results", "emergency_status", "TEXT"),
        ("detection_results", "traffic_alert", "TEXT"),
    ]
    with engine.connect() as conn:
        for table, col, col_type in new_columns:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # Column already exists
