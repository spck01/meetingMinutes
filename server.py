"""スパチキ議事録メーカーのWebサーバー。

起動: uvicorn server:app --port 8000
"""
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

import core  # noqa: E402  (.env のモデル設定を読んでから import する)

app = FastAPI()
jobs: dict[str, dict] = {}


def _run_job(job: dict, src: Path, tmpdir: Path, api_key: str, language: str | None, title: str, notes: str):
    try:
        client = OpenAI(api_key=api_key)
        job["status"] = "extracting"
        chunks = core.extract_audio_chunks(src, tmpdir)

        job["status"] = "transcribing"
        job["progress"] = {"done": 0, "total": len(chunks)}

        def on_progress(done: int, total: int):
            job["progress"] = {"done": done, "total": total}

        transcript = core.transcribe(client, chunks, language=language, on_progress=on_progress)
        job["transcript"] = transcript

        job["status"] = "writing"
        job["minutes"] = core.generate_minutes(client, transcript, title=title, notes=notes)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.get("/api/config")
def config():
    return {
        "has_server_key": bool(os.getenv("OPENAI_API_KEY")),
        "transcribe_model": core.TRANSCRIBE_MODEL,
        "minutes_model": core.MINUTES_MODEL,
    }


@app.post("/api/jobs")
def create_job(
    file: UploadFile = File(...),
    title: str = Form(""),
    notes: str = Form(""),
    language: str = Form("ja"),
    x_openai_key: str | None = Header(None),
):
    api_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(400, "OpenAI APIキーが設定されていません。右上の「設定」から入力してください。")

    tmpdir = Path(tempfile.mkdtemp(prefix="minutes_"))
    src = tmpdir / f"input{Path(file.filename or '').suffix}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "name": Path(file.filename or "meeting").stem}
    jobs[job_id] = job
    threading.Thread(
        target=_run_job,
        args=(job, src, tmpdir, api_key, None if language == "auto" else language, title, notes),
        daemon=True,
    ).start()
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "ジョブが見つかりません。")
    return job


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
