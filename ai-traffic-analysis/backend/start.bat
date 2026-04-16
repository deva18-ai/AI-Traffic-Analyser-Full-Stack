@echo off
echo Installing dependencies...
venv\Scripts\pip install fastapi uvicorn python-multipart opencv-python-headless ultralytics Pillow sqlalchemy python-dotenv numpy aiofiles pydantic

echo.
echo Starting FastAPI server...
venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000
