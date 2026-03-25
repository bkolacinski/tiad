"""
Speech-to-Text module using OpenAI Whisper (offline).
Handles recording, transcription and language detection.
"""

import os
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


SAMPLE_RATE = 16000
CHANNELS = 1
WHISPER_MODEL_SIZE = "small"  # small (~460MB) - good balance for Polish

_model = None
_model_lock = threading.Lock()


def load_model(model_dir: str = None, progress_callback=None) -> whisper.Whisper:
    """Load Whisper model (cached after first load)."""
    global _model
    with _model_lock:
        if _model is None:
            if progress_callback:
                progress_callback("Ładowanie modelu Whisper...")
            download_root = model_dir or os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(download_root, exist_ok=True)
            _model = whisper.load_model(WHISPER_MODEL_SIZE, download_root=download_root)
            if progress_callback:
                progress_callback("Model gotowy.")
    return _model


class AudioRecorder:
    """Records audio from microphone in a background thread."""

    def __init__(self):
        self._frames = []
        self._recording = False
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        self._frames = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time, status):
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def stop(self) -> np.ndarray:
        self._recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        with self._lock:
            if self._frames:
                return np.concatenate(self._frames, axis=0).flatten()
            return np.zeros(SAMPLE_RATE, dtype="float32")


def transcribe(audio: np.ndarray, model_dir: str = None) -> dict:
    """
    Transcribe audio array using Whisper.

    Returns dict with:
        text (str): transcription
        language (str): detected language code (e.g. 'pl', 'en')
        language_name (str): human-readable language name
    """
    model = load_model(model_dir)

    # Whisper expects float32 mono at 16kHz
    audio = audio.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0

    # Detect language first
    audio_padded = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)
    _, probs = model.detect_language(mel)
    lang_code = max(probs, key=probs.get)

    # Transcribe (force detected language for accuracy)
    result = model.transcribe(audio, language=lang_code, fp16=False)

    lang_names = {
        "pl": "polski",
        "en": "angielski",
        "de": "niemiecki",
        "fr": "francuski",
        "es": "hiszpański",
        "it": "włoski",
        "ru": "rosyjski",
        "uk": "ukraiński",
        "cs": "czeski",
    }

    return {
        "text": result["text"].strip(),
        "language": lang_code,
        "language_name": lang_names.get(lang_code, lang_code),
    }


def transcribe_file(filepath: str, model_dir: str = None) -> dict:
    """Transcribe an audio file (wav, mp3, etc.)."""
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Resample via scipy if needed
        from scipy.signal import resample
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples)
    return transcribe(audio, model_dir)
