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
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

import core  # noqa: E402  (.env のモデル設定を読んでから import する)

app = FastAPI()
jobs: dict[str, dict] = {}

MAX_UPLOAD_BYTES = 5 * 1000**3
# 音声の抽出・分割ファイル用に、アップロード分とは別に確保しておく空き容量
DISK_MARGIN_BYTES = 1 * 1000**3


def _run_job(job: dict, src: Path, tmpdir: Path, api_key: str, language: str | None, title: str, notes: str):
    try:
        client = OpenAI(api_key=api_key)
        job["status"] = "extracting"
        chunks = core.extract_audio_chunks(src, tmpdir)

        job["status"] = "transcribing"
        job["progress"] = {"done": 0, "total": len(chunks)}

        def on_progress(done: int, total: int):
            job["progress"] = {"done": done, "total": total}

        transcript = core.transcribe(
            client, chunks, language=language, hint=core.build_hint(title, notes), on_progress=on_progress
        )
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
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


@app.post("/api/jobs")
async def create_job(
    request: Request,
    filename: str = "meeting",
    title: str = "",
    notes: str = "",
    language: str = "ja",
    x_openai_key: str | None = Header(None),
):
    """ファイル本体はリクエストボディにそのまま載せて送る（multipartだと一時ファイル経由で二重に書き込むため）。
    会議名などの付帯情報はクエリパラメータで受け取る。"""
    api_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(400, "OpenAI APIキーが設定されていません。右上の「設定」から入力してください。")

    size = int(request.headers.get("content-length") or 0)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "ファイルが大きすぎます（上限5GB）。")
    tmpdir = Path(tempfile.mkdtemp(prefix="minutes_"))
    if shutil.disk_usage(tmpdir).free < size + DISK_MARGIN_BYTES:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(507, "サーバーのディスク空き容量が足りません。")

    src = tmpdir / f"input{Path(filename).suffix}"
    written = 0
    try:
        with open(src, "wb") as f:
            async for chunk in request.stream():
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "ファイルが大きすぎます（上限5GB）。")
                f.write(chunk)
    except BaseException:
        # 上限超過や、アップロード途中でのブラウザ切断時に書きかけのファイルを残さない
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    job_id = uuid.uuid4().hex
    job = {"id": job_id, "status": "queued", "name": Path(filename).stem}
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
