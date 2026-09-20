"""FastAPI service for synchronous first-version analysis.

The job dictionary is intentionally small and in-memory for the competition prototype.
The ``AnalysisService`` boundary makes it straightforward to move execution to a queue
later without changing the frontend event schema.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import DEFAULT_CONFIG_PATH, ensure_runtime_directories, load_config, resolve_path
from src.pipeline import AnalysisRunResult, analyze_video
from src.video.clipper import check_ffmpeg_available, ffmpeg_executable

REPO_ROOT = Path(__file__).resolve().parents[2]


class AnalyzeRequest(BaseModel):
    input_path: str = Field(
        min_length=1, description="Path relative to the project root or an absolute path"
    )
    config_path: str | None = None


class JobState(BaseModel):
    job_id: str
    status: str
    created_at: str
    input_path: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class AnalysisService:
    """Application service kept separate from route definitions for future async work."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)

    def run(
        self, input_path: str | Path, config_path: str | Path | None = None
    ) -> AnalysisRunResult:
        return analyze_video(input_path, config_path=config_path or self.config_path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    config = load_config(config_path)
    ensure_runtime_directories(config)
    service = AnalysisService(config_path)
    jobs: dict[str, JobState] = {}
    app = FastAPI(title="SoloDirector API", version="0.1.0")

    @app.get("/health")
    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "project": "SoloDirector",
            "ffmpeg": check_ffmpeg_available(),
            "ffmpeg_path": ffmpeg_executable(),
        }

    @app.get("/api/v1/videos")
    def list_videos() -> dict[str, Any]:
        root = Path(str(config["project"]["root"]))
        input_dir = root / "data" / "input"
        videos = [
            path.relative_to(root).as_posix()
            for path in sorted(input_dir.glob("**/*"))
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
        ]
        return {"videos": videos}

    @app.post("/api/v1/analyze", response_model=JobState)
    def analyze(request: AnalyzeRequest) -> JobState:
        root = Path(str(config["project"]["root"]))
        input_path = Path(request.input_path).expanduser()
        if not input_path.is_absolute():
            input_path = (root / input_path).resolve()
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"Video does not exist: {input_path}")
        job = JobState(
            job_id=uuid.uuid4().hex, status="running", created_at=_now(), input_path=str(input_path)
        )
        jobs[job.job_id] = job
        try:
            result = service.run(input_path, request.config_path)
            job.status = "completed"
            job.result = result.as_dict()
        except Exception as error:
            job.status = "failed"
            job.error = str(error)
        jobs[job.job_id] = job
        return job

    @app.post("/api/v1/analyze/upload", response_model=JobState)
    async def analyze_upload(
        file: UploadFile = File(...),
        config_path: str | None = Form(default=None),
    ) -> JobState:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename")
        upload_dir = resolve_path(
            config, config.get("api", {}).get("upload_dir", "data/input/uploads")
        )
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename).name
        target = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        with target.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        job = JobState(
            job_id=uuid.uuid4().hex, status="running", created_at=_now(), input_path=str(target)
        )
        jobs[job.job_id] = job
        try:
            result = service.run(target, config_path)
            job.status = "completed"
            job.result = result.as_dict()
        except Exception as error:
            job.status = "failed"
            job.error = str(error)
        jobs[job.job_id] = job
        return job

    @app.get("/api/v1/jobs/{job_id}", response_model=JobState)
    def get_job(job_id: str) -> JobState:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        return job

    @app.get("/api/v1/jobs/{job_id}/result")
    def get_result(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
        if job.status != "completed" or job.result is None:
            raise HTTPException(status_code=409, detail={"status": job.status, "error": job.error})
        return job.result

    return app


app = create_app()
