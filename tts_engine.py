"""
Wrapper around Coqui XTTS v2 engine for one-shot synthesis.
"""
# --- Coqui XTTS v2 Engine ---
try:
    import torch
except ImportError:
    print("Please install torch: pip install torch")
    torch = None
import soundfile as sf
from pathlib import Path
from typing import List
import logging
from TTS.api import TTS

# Define the Coqui XTTS model ID
COQUI_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

class CoquiXTTSv2Engine:
    def __init__(self):
        if torch is None:
            raise RuntimeError("torch library not found. Please install requirements.")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logging.info(f"Initializing Coqui XTTS engine ({COQUI_MODEL_ID}) on {self.device}...")
        # Note: XTTS v2 requires accepting terms on Hugging Face first if not cached.
        # You might need to set environment variable COQUI_TTS_AGREED=1
        # Or provide a Hugging Face token if needed.
        try:
            self.tts = TTS(COQUI_MODEL_ID).to(self.device)
            self.sample_rate = self.tts.synthesizer.output_sample_rate
            logging.info(f"Model loaded. Sample rate: {self.sample_rate}")
            logging.info("Coqui XTTS engine initialized.")
        except Exception as e:
            logging.error(f"Failed to load Coqui XTTS model: {e}")
            logging.error("Please ensure you have accepted the model terms on Hugging Face and have network connectivity.")
            # Consider adding info about COQUI_TTS_AGREED=1 env var or HF token.
            raise

    def synthesize(self, segments: List[str], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        logging.info(f"Synthesizing {len(segments)} segments using Coqui XTTS...")
        total_segments = len(segments)

        for i, text in enumerate(segments):
            # Basic cleaning (more can be added if needed)
            cleaned_text = text.strip()
            if not cleaned_text:
                logging.warning(f"  Segment {i+1}/{total_segments}: Skipped empty segment.")
                continue

            segment_num = i + 1
            output_path = out_dir / f"segment_{segment_num:04d}.wav"
            logging.info(f"  Segment {segment_num}/{total_segments}: Processing '{cleaned_text[:50]}...' -> {output_path.name}")

            try:
                # Synthesize using tts_to_file
                self.tts.tts_to_file(
                    text=cleaned_text,
                    file_path=str(output_path),
                    speaker_wav=None, # Use default voice for now
                    language='en' # Set language explicitly
                )

                # Verify file exists and has size
                if output_path.exists() and output_path.stat().st_size > 0:
                    paths.append(output_path)
                    logging.info(f"  Segment {segment_num}/{total_segments}: Saved successfully.")
                else:
                    logging.warning(f"  Segment {segment_num}/{total_segments}: Failed to save or empty file at {output_path}")

            except Exception as e:
                logging.error(f"  Segment {segment_num}/{total_segments}: Error during generation for text: '{cleaned_text[:100]}...'", exc_info=True)
                logging.error(f"  Error details: {e}")
                # Decide whether to continue or stop
                # continue

        logging.info(f"Synthesis complete. Generated {len(paths)} audio files.")
        if len(paths) < total_segments:
             logging.warning(f"Expected {total_segments} segments, but only generated {len(paths)}. Check logs for errors.")
        return paths
