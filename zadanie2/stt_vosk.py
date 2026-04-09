"""
Offline STT via Vosk (Kaldi) — Polish and English small models only.
"""

from __future__ import annotations

import json
import os
import threading

import numpy as np

# Must match stt.SAMPLE_RATE (16 kHz for Whisper/Vosk)
SAMPLE_RATE = 16000

_vosk_models: dict[str, object] = {}
_vosk_lock = threading.Lock()


def vosk_models_root(model_dir: str) -> str:
    return os.path.join(model_dir, "vosk")


def model_path_for_lang(model_dir: str, lang: str) -> str:
    """lang: 'pl' or 'en' — unpacked vosk model directory."""
    sub = "pl" if lang == "pl" else "en"
    return os.path.join(vosk_models_root(model_dir), sub)


def vosk_model_ready(model_dir: str, lang: str) -> bool:
    """True if am/final.mdl exists (typical Vosk layout)."""
    root = model_path_for_lang(model_dir, lang)
    return os.path.isfile(os.path.join(root, "am", "final.mdl"))


def vosk_available(model_dir: str) -> bool:
    return vosk_model_ready(model_dir, "pl") and vosk_model_ready(model_dir, "en")


def _get_model(model_dir: str, lang: str):
    """Lazily load a ``vosk.Model`` for the given model directory and language."""
    global _vosk_models
    key = f"{model_dir}:{lang}"
    with _vosk_lock:
        if key not in _vosk_models:
            from vosk import Model

            path = model_path_for_lang(model_dir, lang)
            if not vosk_model_ready(model_dir, lang):
                raise FileNotFoundError(
                    f"Vosk model for language '{lang}' not found at {path}"
                )
            _vosk_models[key] = Model(path)
        return _vosk_models[key]


def transcribe_vosk(
    audio: np.ndarray,
    model_dir: str,
    language: str,
) -> dict:
    """
    Transcribe mono float32 audio at SAMPLE_RATE Hz using Vosk.

    ``language`` must be 'pl' or 'en'.

    Returns dict: text, language, language_name (aligned with Whisper output).
    """
    if language not in ("pl", "en"):
        raise ValueError("Vosk: only languages 'pl' and 'en' are supported.")

    from vosk import KaldiRecognizer

    if not vosk_model_ready(model_dir, language):
        raise FileNotFoundError(
            f"Vosk model ({language}) is not installed. Run setup.py."
        )

    model = _get_model(model_dir, language)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    audio = audio.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    chunk_samples = 4000
    offset = 0
    while offset < len(pcm):
        chunk = bytes(pcm[offset : offset + chunk_samples])
        offset += chunk_samples
        rec.AcceptWaveform(chunk)

    raw = rec.FinalResult()
    data = json.loads(raw)
    text = (data.get("text") or "").strip()

    lang_names = {"pl": "Polish", "en": "English"}
    return {
        "text": text,
        "language": language,
        "language_name": lang_names.get(language, language),
    }
