"""
Meta Massively Multilingual Speech (MMS) ASR — ``facebook/mms-1b-all`` on Hugging Face.
Wav2Vec2 with language adapters (ISO 639-3), not Whisper. Polish uses adapter ``pol``.

See: https://huggingface.co/facebook/mms-1b-all
"""

from __future__ import annotations

import os
import threading

import numpy as np

SAMPLE_RATE = 16000

MODEL_ID = "facebook/mms-1b-all"

# Whisper ISO 639-1 -> MMS ISO 639-3 adapters (PL/EN only, same as Vosk)
MMS_ADAPTER = {"pl": "pol", "en": "eng"}

LANG_NAMES = {"pl": "Polish", "en": "English"}

_mms_cache: dict[str, tuple] = {}
_lock = threading.Lock()


def _hf_home(model_dir: str) -> str:
    root = os.path.join(model_dir, "hf")
    os.makedirs(root, exist_ok=True)
    return root


def mms_transformers_available() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _get_mms(model_dir: str):
    """Load MMS processor and model (cached per ``model_dir``, thread-safe)."""
    with _lock:
        if model_dir not in _mms_cache:
            if not mms_transformers_available():
                raise RuntimeError("Install package: transformers")
            os.environ["HF_HOME"] = _hf_home(model_dir)
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            import torch

            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
            model.eval()
            _mms_cache[model_dir] = (processor, model, torch)
        return _mms_cache[model_dir]


def transcribe_mms(
    audio: np.ndarray,
    model_dir: str,
    language: str,
) -> dict:
    """
    Mono float32 ~16 kHz. ``language``: ``pl`` | ``en`` selects the MMS adapter.
    """
    adapter = MMS_ADAPTER.get(language)
    if adapter is None:
        raise ValueError("MMS: only languages 'pl' and 'en' are supported.")

    processor, model, torch = _get_mms(model_dir)

    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size and audio.max() > 1.0:
        audio = audio / 32768.0

    if audio.size == 0:
        return {"text": "", "language": language, "language_name": LANG_NAMES.get(language, language)}

    with _lock:
        processor.tokenizer.set_target_lang(adapter)
        model.load_adapter(adapter)

        inputs = processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.decode(predicted_ids[0]).strip()

    return {
        "text": text,
        "language": language,
        "language_name": LANG_NAMES.get(language, language),
    }
