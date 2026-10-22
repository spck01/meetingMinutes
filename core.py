"""録画/録音ファイル → 文字起こし(Whisper) → 議事録(GPT) の共通処理。

Webアプリ(app.py)からも、将来のDiscord botからも同じ関数を呼べるようにしている。
"""
import os
import subprocess
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
from openai import OpenAI

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")
MINUTES_MODEL = os.getenv("MINUTES_MODEL", "gpt-5-mini")
# Whisper APIは1ファイル25MBまでなので、10分ごとに分割する（32kbpsモノラルで1チャンク約2.4MB）
CHUNK_SECONDS = 600

MINUTES_PROMPT = """あなたは優秀な議事録作成者です。以下は会議の文字起こし（音声認識の結果なので誤字・脱字を含む）です。
これをもとに、日本語のMarkdownで議事録を作成してください。

# 出力フォーマット
## 会議概要
（2〜4文で要約）
## 決定事項
- 箇条書き
## ToDo
| 内容 | 担当 | 期限 |
|---|---|---|
## 議論の要点
- トピックごとに見出し or 箇条書きで、誰がどんな意見を出したか分かる範囲で
## 保留・次回への持ち越し
- 箇条書き

# ルール
- 文字起こしに無いことは書かない。担当や期限が不明なら「未定」と書く。
- 話者は音声が混ざっていて特定できないことが多い。名前が発言内で明示されている場合以外は話者を推測しない。
- 明らかな音声認識ミスは文脈から自然に補正してよい。
- 該当なしのセクションは「なし」と書く。
"""


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def extract_audio_chunks(src: Path, workdir: Path) -> list[Path]:
    """動画/音声から音声だけを取り出し、CHUNK_SECONDSごとのmp3に分割する。"""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = workdir / "chunk_%03d.mp3"
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k",
        "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-reset_timestamps", "1",
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"音声の抽出に失敗しました: {result.stderr.strip()}")
    chunks = sorted(workdir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("音声トラックが見つかりませんでした。")
    return chunks


def transcribe(
    client: OpenAI,
    chunks: list[Path],
    language: str | None = "ja",
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    """チャンクを順に文字起こしし、[h:mm:ss] 付きのテキストにまとめる。"""
    lines: list[str] = []
    for i, path in enumerate(chunks):
        offset = i * CHUNK_SECONDS
        kwargs = {"model": TRANSCRIBE_MODEL}
        if language:
            kwargs["language"] = language
        with open(path, "rb") as f:
            if TRANSCRIBE_MODEL == "whisper-1":
                # whisper-1 だけがセグメント単位のタイムスタンプを返せる
                res = client.audio.transcriptions.create(file=f, response_format="verbose_json", **kwargs)
                for seg in res.segments or []:
                    text = seg.text.strip()
                    if text:
                        lines.append(f"[{_fmt_time(offset + seg.start)}] {text}")
            else:
                res = client.audio.transcriptions.create(file=f, **kwargs)
                if res.text.strip():
                    lines.append(f"[{_fmt_time(offset)}] {res.text.strip()}")
        if on_progress:
            on_progress(i + 1, len(chunks))
    return "\n".join(lines)


def generate_minutes(client: OpenAI, transcript: str, title: str = "", notes: str = "") -> str:
    header = ""
    if title:
        header += f"会議名: {title}\n"
    if notes:
        header += f"補足情報（参加者など）: {notes}\n"
    res = client.chat.completions.create(
        model=MINUTES_MODEL,
        messages=[
            {"role": "system", "content": MINUTES_PROMPT},
            {"role": "user", "content": f"{header}\n# 文字起こし\n{transcript}"},
        ],
    )
    return res.choices[0].message.content or ""
