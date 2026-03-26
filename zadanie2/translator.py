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
                    (f.code, t.code) for f in installed for t in f.translations_to
                }
            except Exception:
                _installed_pairs = set()


def get_available_targets(source_lang: str) -> List[Tuple[str, str]]:
    """Return list of (code, name) for available translation targets."""
    _ensure_initialized()
    result = []
    for code, name in SUPPORTED_LANGUAGES:
        if code != source_lang and (
            _installed_pairs is None or (source_lang, code) in _installed_pairs
        ):
            result.append((code, name))
    return result


def translate_text(text: str, source: str, target: str) -> Optional[str]:
    """
    Translate text offline.
    Returns translated string, or raises TranslationUnavailable if packages missing.
    Supports intermediate translation via English when direct pair is unavailable
    (e.g. pl→en→de).
    """
    try:
        from argostranslate import translate as argtrans

        installed = argtrans.get_installed_languages()

        src_lang = next((l for l in installed if l.code == source), None)
        if not src_lang:
            raise TranslationUnavailable(source, target)
        tgt_lang = next((l for l in installed if l.code == target), None)

        # Try direct translation first
        translation = src_lang.get_translation(tgt_lang) if tgt_lang else None
        if translation:
            return translation.translate(text)

        # Fallback: translate via English as intermediate language
        if source != "en" and target != "en":
            en_lang = next((l for l in installed if l.code == "en"), None)
            if en_lang:
                to_en = src_lang.get_translation(en_lang)
                from_en = en_lang.get_translation(tgt_lang) if tgt_lang else None
                if to_en and from_en:
                    intermediate = to_en.translate(text)
                    return from_en.translate(intermediate)

        raise TranslationUnavailable(source, target)
    except TranslationUnavailable:
        raise
    except Exception as e:
        raise TranslationUnavailable(source, target) from e


class TranslationUnavailable(Exception):
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        super().__init__(f"Brak pakietu tłumaczeń {source}→{target}")


def install_language_pair(source: str, target: str, progress_callback=None) -> bool:
    """Download and install a language pair for offline use.
    When direct pair is unavailable, installs source→en + en→target."""
    try:
        from argostranslate import package as argpkg

        if progress_callback:
            progress_callback(f"Pobieranie pakietu tłumaczeń {source}→{target}...")
        argpkg.update_package_index()
        available = argpkg.get_available_packages()

        # Try direct pair
        pkg = next(
            (p for p in available if p.from_code == source and p.to_code == target),
            None,
        )
        if pkg:
            argpkg.install_from_path(pkg.download())
            global _installed_pairs
            _installed_pairs = None
            return True

        # Fallback: install via English intermediate
        if source != "en" and target != "en":
            installed_any = False
            for pair_src, pair_tgt in [(source, "en"), ("en", target)]:
                pair_pkg = next(
                    (
                        p
                        for p in available
                        if p.from_code == pair_src and p.to_code == pair_tgt
                    ),
                    None,
                )
                if pair_pkg:
                    if progress_callback:
                        progress_callback(
                            f"Pobieranie pakietu {pair_src}→{pair_tgt}..."
                        )
                    argpkg.install_from_path(pair_pkg.download())
                    installed_any = True
            if installed_any:
                _installed_pairs = None
                return True

        return False
    except Exception:
        return False
