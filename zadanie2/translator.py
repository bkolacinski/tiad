"""
Offline translation using argostranslate.
Used for grade-5 requirement: translate transcription to another language.
"""

import threading
from typing import Optional, List, Tuple

_installed_pairs: set = None
_lock = threading.Lock()

SUPPORTED_LANGUAGES = [
    ("pl", "polski"),
    ("en", "angielski"),
    ("de", "niemiecki"),
    ("fr", "francuski"),
    ("es", "hiszpański"),
]


def _ensure_initialized():
    global _installed_pairs
    with _lock:
        if _installed_pairs is None:
            try:
                from argostranslate import package, translate
                installed = translate.get_installed_languages()
                _installed_pairs = {
                    (f.code, t.code)
                    for f in installed
                    for t in f.translations_to
                }
            except Exception:
                _installed_pairs = set()


def get_available_targets(source_lang: str) -> List[Tuple[str, str]]:
    """Return list of (code, name) for available translation targets."""
    _ensure_initialized()
    result = []
    for code, name in SUPPORTED_LANGUAGES:
        if code != source_lang and (_installed_pairs is None or (source_lang, code) in _installed_pairs):
            result.append((code, name))
    return result


def translate_text(text: str, source: str, target: str) -> Optional[str]:
    """Translate text offline. Returns None if translation unavailable."""
    try:
        from argostranslate import translate as argtrans
        installed = argtrans.get_installed_languages()
        src_lang = next((l for l in installed if l.code == source), None)
        if not src_lang:
            return None
        translation = src_lang.get_translation(
            next((l for l in installed if l.code == target), None)
        )
        if not translation:
            return None
        return translation.translate(text)
    except Exception:
        return None


def install_language_pair(source: str, target: str, progress_callback=None) -> bool:
    """Download and install a language pair for offline use."""
    try:
        from argostranslate import package as argpkg
        if progress_callback:
            progress_callback(f"Pobieranie pakietu tłumaczeń {source}→{target}...")
        argpkg.update_package_index()
        available = argpkg.get_available_packages()
        pkg = next(
            (p for p in available if p.from_code == source and p.to_code == target),
            None
        )
        if pkg:
            argpkg.install_from_path(pkg.download())
            global _installed_pairs
            _installed_pairs = None  # Reset cache
            return True
        return False
    except Exception:
        return False
