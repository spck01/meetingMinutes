"""録画/録音ファイル → 文字起こし → 議事録(GPT) の共通処理。

Webアプリ(server.py)からも、将来のDiscord botからも同じ関数を呼べるようにしている。
"""
import os
import subprocess
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
from openai import OpenAI

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe")
MINUTES_MODEL = os.getenv("MINUTES_MODEL", "gpt-5-mini")
# 文字起こしAPIは1ファイル25MBまでなので分割して送る。
# whisper-1 は発言ごとのタイムスタンプを返せるので10分単位、
# それ以外のモデルはチャンク単位の時刻しか付けられないので3分単位にして目安の時刻を細かくする。
# 料金は音声の長さで決まるため、分割数を増やしても変わらない。
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600" if TRANSCRIBE_MODEL == "whisper-1" else "180"))
# 文字起こしに渡すヒント(prompt)の長さの上限。whisper-1 は末尾224トークンしか使わない
HINT_MAX_CHARS = 400
# チャンクの境目で文脈が途切れないよう、直前チャンクの末尾をヒントに含める
CONTEXT_TAIL_CHARS = 150

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
- トピックごとに見出しを立て、出た意見・結論を箇条書きで
## 保留・次回への持ち越し
- 箇条書き

# ルール
- 文字起こしに無いことは書かない。担当や期限が不明なら「未定」と書く。
- 音声は全員分が混ざっていて話者を特定できないため、発言者は書かない。「（発言：〜）」のような注記も付けない。
- ToDoの担当は、発言の中で担当者の名前がはっきり示された場合のみ書く（例：「〇〇さんお願いします」「自分がやります、〇〇です」）。
- 人名は漢字やカタカナではなく、読みをローマ字の小文字で書く。補足情報にローマ字の表記があればそれに合わせる。
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


def build_hint(title: str = "", notes: str = "") -> str:
    """会議名・補足情報を、文字起こしモデルに渡す語彙ヒントにする（固有名詞の認識精度が上がる）。"""
    hint = "。".join(s.strip() for s in (title, notes) if s.strip())
    return hint[:HINT_MAX_CHARS]


def transcribe(
    client: OpenAI,
    chunks: list[Path],
    language: str | None = "ja",
    hint: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> str:
    """チャンクを順に文字起こしし、[h:mm:ss] 付きのテキストにまとめる。"""
    lines: list[str] = []
    prev_tail = ""
    for i, path in enumerate(chunks):
        offset = i * CHUNK_SECONDS
        kwargs = {"model": TRANSCRIBE_MODEL}
        if language:
            kwargs["language"] = language
        prompt = "\n".join(s for s in (hint, prev_tail) if s)
        if prompt:
            kwargs["prompt"] = prompt
        with open(path, "rb") as f:
            if TRANSCRIBE_MODEL == "whisper-1":
                # whisper-1 だけがセグメント単位のタイムスタンプを返せる
                res = client.audio.transcriptions.create(file=f, response_format="verbose_json", **kwargs)
                texts = [(seg.start, seg.text.strip()) for seg in res.segments or []]
            else:
                res = client.audio.transcriptions.create(file=f, **kwargs)
                texts = [(0, res.text.strip())]
        for start, text in texts:
            if text:
                lines.append(f"[{_fmt_time(offset + start)}] {text}")
        chunk_text = "".join(text for _, text in texts)
        if chunk_text:
            prev_tail = chunk_text[-CONTEXT_TAIL_CHARS:]
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
