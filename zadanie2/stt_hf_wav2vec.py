"""
Hugging Face Wav2Vec2 XLS-R ASR — separate models for PL and EN.
Not Whisper; standard wav2vec2 CTC.

- PL: facebook/wav2vec2-large-xlsr-53-polish
- EN: facebook/wav2vec2-large-960h-lv60-self
"""

from __future__ import annotations

import os
import threading

import numpy as np

SAMPLE_RATE = 16000

MODEL_PL = "facebook/wav2vec2-large-xlsr-53-polish"
MODEL_EN = "facebook/wav2vec2-large-960h-lv60-self"

LANG_NAMES = {"pl": "Polish", "en": "English"}

_models: dict[str, tuple] = {}
_lock = threading.Lock()


def _hf_home(model_dir: str) -> str:
    root = os.path.join(model_dir, "hf")
    os.makedirs(root, exist_ok=True)
    return root


def hf_wav2vec_transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _get_pair(model_dir: str, lang: str):
    """Return (processor, model, torch) for the given language; cached in RAM."""
    if lang not in ("pl", "en"):
        raise ValueError("hf_wav2vec: only languages 'pl' and 'en' are supported.")

    key = f"{model_dir}\x00{lang}"
    with _lock:
        if key not in _models:
            os.environ.setdefault("HF_HOME", _hf_home(model_dir))
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            import torch

            model_id = MODEL_PL if lang == "pl" else MODEL_EN
            processor = Wav2Vec2Processor.from_pretrained(model_id)
            model = Wav2Vec2ForCTC.from_pretrained(model_id)
            model.eval()
            _models[key] = (processor, model, torch)
        return _models[key]


def transcribe_hf_wav2vec(
    audio: np.ndarray,
    model_dir: str,
    language: str,
) -> dict:
    """
    Mono float32 ~16 kHz. ``language``: ``pl`` | ``en`` (same as Vosk path from Whisper detection).
    """
    if not hf_wav2vec_transformers_available():
        raise RuntimeError("Install package: transformers (pip install transformers)")

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size and audio.max() > 1.0:
        audio = audio / 32768.0

    processor, model, torch = _get_pair(model_dir, language)
    if audio.size == 0:
        return {"text": "", "language": language, "language_name": LANG_NAMES.get(language, language)}

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()

    return {
        "text": text,
        "language": language,
        "language_name": LANG_NAMES.get(language, language),
    }
