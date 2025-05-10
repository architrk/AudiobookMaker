"""
CLI entry-point: python audiobook_maker.py book.epub
Produces audiobook.mp3 in the same folder.
"""
import argparse, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from text_cleaner import load_epub, llama_filter
from utils import chunk_text, concat_wav
from tts_engine import CoquiXTTSv2Engine
import sys
import logging

def build_llama(device):
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return pipeline("text-generation", model=model, tokenizer=tok, device=device)

def main(epub_path: str, use_llm: bool = False):
    epub_path = Path(epub_path)
    raw_paras = load_epub(str(epub_path))
    if use_llm:
        device = 0 if torch.cuda.is_available() else -1
        llm = build_llama(device)
        cleaned = llama_filter(raw_paras, llm)
    else:
        cleaned = raw_paras

    segments = chunk_text(cleaned)

    # Filter out empty or whitespace-only segments before synthesis
    segments = [seg for seg in segments if seg and not seg.isspace()]

    if not segments:
        print("Error: No valid text segments found after cleaning.")
        sys.exit(1)

    # Print debug info
    print(f"\nDEBUG: Number of segments after filtering: {len(segments)}")
    if segments: # Ensure list is not empty before accessing index 0
        print(f"DEBUG: First segment passed to TTS: '{segments[0]}'\n", file=sys.stderr)
    else:
        print("DEBUG: Segments list is unexpectedly empty after filtering check!", file=sys.stderr)
        sys.exit(1)

    tts = CoquiXTTSv2Engine()
    wavs = tts.synthesize(segments, epub_path.with_suffix("").with_name("tmp_audio"))

    out_mp3 = epub_path.with_suffix("").with_name("audiobook.mp3")
    concat_wav(wavs, out_mp3)
    print(f"  Audiobook saved to {out_mp3.absolute()}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("audiobook_maker.log"),
            logging.StreamHandler()
        ]
    )
    parser = argparse.ArgumentParser(description="EPUB → Audiobook")
    parser.add_argument("epub", help="Path to .epub file")
    parser.add_argument("--llm", action="store_true", help="Use Llama-3 relevance filtering")
    args = parser.parse_args()
    main(args.epub, args.llm)
