"""
Utility helpers: simple greedy text chunking & audio concatenation.
"""
from typing import List
from pydub import AudioSegment
from pathlib import Path

def chunk_text(paragraphs: List[str], max_chars: int = 2000) -> List[str]:
    buf, segs = "", []
    for p in paragraphs:
        if len(buf) + len(p) + 1 > max_chars:
            segs.append(buf.strip())
            buf = p
        else:
            buf += " " + p
    if buf:
        segs.append(buf.strip())
    return segs

def concat_wav(wavs: List[Path], outfile: Path):
    merged = AudioSegment.empty()
    for w in wavs:
        merged += AudioSegment.from_file(w)
    merged.export(outfile, format="mp3")
