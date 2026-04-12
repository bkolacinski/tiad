"""
Speech-to-Text module using OpenAI Whisper (offline).
Handles recording, transcription and language detection.
"""

import io
import os
import sys
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper


SAMPLE_RATE = 16000           # Whisper / Vosk input rate
RECORDING_SAMPLE_RATE = 44100 # Mic capture rate — better playback quality
CHANNELS = 1
WHISPER_MODEL_SIZE = "small"  # small (~460MB) — good tradeoff for multilingual use

LANG_NAMES = {
    "pl": "Polish",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "cs": "Czech",
}

_model = None
_model_lock = threading.Lock()


def load_model(model_dir: str = None, progress_callback=None) -> whisper.Whisper:
    """Load Whisper model (cached after first load)."""
    global _model
    with _model_lock:
        if _model is None:
            if progress_callback:
                progress_callback("Loading Whisper model...")
            download_root = model_dir or os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(download_root, exist_ok=True)
            # tqdm (used by whisper during download) writes to stdout/stderr —
            # both can be None in GUI/frozen apps, which causes AttributeError.
            _null = io.StringIO()
            _prev_stdout = sys.stdout
            _prev_stderr = sys.stderr
            if sys.stdout is None:
                sys.stdout = _null
            if sys.stderr is None:
                sys.stderr = _null
            try:
                _model = whisper.load_model(WHISPER_MODEL_SIZE, download_root=download_root)
            finally:
                sys.stdout = _prev_stdout
                sys.stderr = _prev_stderr
            if progress_callback:
                progress_callback("Model ready.")
    return _model


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.max() > 1.0:
        audio = audio / 32768.0
    return audio


def detect_language_code(audio: np.ndarray, model_dir: str = None) -> str:
    """
    Detect spoken language using Whisper (single forward pass, no full transcription).
    Expects float32 mono ~16 kHz.
    """
    model = load_model(model_dir)
    audio = _normalize_audio(audio)
    audio_padded = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def transcribe_whisper(
    audio: np.ndarray, model_dir: str = None, language: Optional[str] = None
) -> dict:
    """
    Transcribe using Whisper.

    If ``language`` is None, language is auto-detected first.

    Returns dict with:
        text (str), language (str), language_name (str)
    """
    model = load_model(model_dir)
    audio = _normalize_audio(audio)
    if language is None:
        audio_padded = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_padded).to(model.device)
        _, probs = model.detect_language(mel)
        lang_code = max(probs, key=probs.get)
    else:
        lang_code = language

    result = model.transcribe(audio, language=lang_code, fp16=False)

    return {
        "text": result["text"].strip(),
        "language": lang_code,
        "language_name": LANG_NAMES.get(lang_code, lang_code),
    }


def transcribe(audio: np.ndarray, model_dir: str = None) -> dict:
    """Transcribe audio array using Whisper (alias for transcribe_whisper)."""
    return transcribe_whisper(audio, model_dir, language=None)


class AudioDeviceError(RuntimeError):
    """Raised when no usable audio input device is available."""


def audio_input_available() -> bool:
    """Return True if at least one audio input device with a working default exists."""
    try:
        devices = sd.query_devices()
    except Exception:
        return False
    for dev in devices:
        if dev.get("max_input_channels", 0) > 0:
            return True
    return False


class AudioRecorder:
    """Records audio from microphone in a background thread."""

    def __init__(self):
        self._frames = []
        self._recording = False
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        self._frames = []
        try:
            self._stream = sd.InputStream(
                samplerate=RECORDING_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._stream = None
            raise AudioDeviceError(
                f"Nie można otworzyć mikrofonu: {exc}"
            ) from exc
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(
                f"Błąd wejścia audio: {exc}"
            ) from exc
        self._recording = True

    def _callback(self, indata, frames, time, status):
        if self._recording:
            with self._lock:
                self._frames.append(indata.copy())

    def stop(self) -> np.ndarray:
        """Return raw audio at RECORDING_SAMPLE_RATE. Caller must resample for STT."""
        self._recording = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None
        with self._lock:
            if self._frames:
                return np.concatenate(self._frames, axis=0).flatten()
            return np.zeros(RECORDING_SAMPLE_RATE, dtype=np.float32)


def transcribe_file(filepath: str, model_dir: str = None) -> dict:
    """Transcribe an audio file (wav, mp3, etc.)."""
    audio, sr = sf.read(filepath, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample

        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = resample(audio, num_samples)
    return transcribe(audio, model_dir)
