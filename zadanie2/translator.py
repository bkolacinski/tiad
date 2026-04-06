"""Offline translation using bundled Argos Translate packages."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


def _bootstrap_ssl_bundle_frozen() -> None:
    """W EXE biblioteki (httpx/urllib) musza widziec prawdziwy cacert.pem z _MEIPASS, nie sciezke z kompilacji."""
    if not getattr(sys, "frozen", False):
        return
    meipass = getattr(sys, "_MEIPASS", None)
    root_abs = os.path.abspath(meipass) if meipass else ""
    ca: str | None = None
    try:
        import certifi

        cand = certifi.where()
        if cand and os.path.isfile(cand) and root_abs:
            ap = os.path.abspath(cand)
            if ap.startswith(root_abs):
                ca = ap
    except Exception:
        pass
    if (not ca or not os.path.isfile(ca)) and meipass:
        for dirpath, _, filenames in os.walk(meipass):
            if "cacert.pem" in filenames:
                p = os.path.join(dirpath, "cacert.pem")
                if os.path.isfile(p):
                    ca = os.path.abspath(p)
                    break
    if ca and os.path.isfile(ca):
        os.environ["SSL_CERT_FILE"] = ca
        os.environ["REQUESTS_CA_BUNDLE"] = ca
        os.environ["CURL_CA_BUNDLE"] = ca


_bootstrap_ssl_bundle_frozen()

_lock = threading.Lock()
_configured_packages_dir: Path | None = None
_ctranslate2_bootstrap_done = False

SUPPORTED_LANGUAGES = [
    ("pl", "polski"),
    ("en", "angielski"),
    ("de", "niemiecki"),
    ("fr", "francuski"),
    ("es", "hiszpanski"),
]
SUPPORTED_CODES = {code for code, _ in SUPPORTED_LANGUAGES}


def _default_packages_dir() -> Path:
    env_dir = os.environ.get("ARGOS_PACKAGES_DIR")
    if env_dir:
        return Path(env_dir)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "translation_packages"
    return Path(__file__).resolve().parent / "translation_packages"


def _bootstrap_ctranslate2_runtime() -> None:
    """W PyInstaller na Windows biblioteki DLL ctranslate2/torch musza byc widoczne dla loadera."""
    global _ctranslate2_bootstrap_done
    if not getattr(sys, "frozen", False):
        return
    if _ctranslate2_bootstrap_done:
        return
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return
    root = os.path.abspath(meipass)
    prefix = root + os.pathsep
    path = os.environ.get("PATH", "")
    if not path.startswith(prefix):
        os.environ["PATH"] = prefix + path
    if hasattr(os, "add_dll_directory"):
        for sub in (
            ".",
            "ctranslate2",
            "ctranslate2.libs",
            "numpy.libs",
            os.path.join("torch", "lib"),
        ):
            p = root if sub == "." else os.path.join(root, sub)
            if os.path.isdir(p):
                try:
                    os.add_dll_directory(p)
                except (OSError, ValueError, AttributeError):
                    pass
    _ctranslate2_bootstrap_done = True


def configure_translation_packages(base_dir: str | Path | None = None) -> Path:
    global _configured_packages_dir
    package_dir = Path(base_dir) if base_dir else _default_packages_dir()
    if _configured_packages_dir is not None and _configured_packages_dir.resolve() == package_dir.resolve():
        return package_dir
    _bootstrap_ctranslate2_runtime()
    package_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARGOS_PACKAGES_DIR"] = str(package_dir)
    os.environ["ARGOS_TRANSLATE_PACKAGE_DIR"] = str(package_dir)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    _configured_packages_dir = package_dir
    try:
        from argostranslate import settings as argos_settings

        argos_settings.package_data_dir = package_dir
        argos_settings.package_dirs = [package_dir]
    except Exception:
        pass
    # Zaladuj DLL ctranslate2 w tym samym watku co GUI zanim worker wywoła Argos (Windows/EXE).
    try:
        import ctranslate2  # noqa: F401
    except Exception:
        pass
    return package_dir


def _ensure_runtime():
    configure_translation_packages(_configured_packages_dir or os.environ.get("ARGOS_PACKAGES_DIR"))


def _installed_pairs() -> set[tuple[str, str]]:
    _ensure_runtime()
    from argostranslate import translate as argtranslate

    pairs: set[tuple[str, str]] = set()
    for language in argtranslate.get_installed_languages():
        for translation in language.translations_to:
            pairs.add((translation.from_lang.code, translation.to_lang.code))
    return pairs


def get_available_targets(source_lang: str) -> list[tuple[str, str]]:
    source_lang = (source_lang or "").strip().lower()
    pairs = _installed_pairs()
    result = []
    for code, name in SUPPORTED_LANGUAGES:
        if code != source_lang and ((source_lang, code) in pairs):
            result.append((code, name))
    return result


def translate_text(text: str, source: str, target: str) -> str:
    _ensure_runtime()
    source = (source or "").lower()
    target = (target or "").lower()
    if source == target:
        return text
    if source not in SUPPORTED_CODES or target not in SUPPORTED_CODES:
        raise TranslationUnavailable(source, target)

    from argostranslate import translate as argtranslate

    try:
        with _lock:
            out = argtranslate.translate(text, source, target)
    except Exception as exc:
        raise TranslationUnavailable(source, target, detail=str(exc)) from exc
    if out is None:
        raise TranslationUnavailable(source, target, detail="translate zwrocilo None")
    return str(out).strip()


class TranslationUnavailable(Exception):
    def __init__(self, source: str, target: str, detail: str = ""):
        msg = f"Brak pakietu tłumaczeń {source}->{target}"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg)
        self.source = source
        self.target = target
        self.detail = detail


def install_language_pair(source: str, target: str, progress_callback=None) -> bool:
    _ensure_runtime()
    source = (source or "").lower()
    target = (target or "").lower()
    with _lock:
        try:
            from argostranslate import package as argpackage

            if progress_callback:
                progress_callback(f"Pobieranie pakietu tłumaczeń {source}->{target}...")
            argpackage.update_package_index()
            available = argpackage.get_available_packages()

            plan = [(source, target)]
            if source != "en" and target != "en":
                plan = [(source, "en"), ("en", target)]

            installed_any = False
            for pair_source, pair_target in plan:
                if (pair_source, pair_target) in _installed_pairs():
                    installed_any = True
                    continue

                package = next(
                    (
                        item
                        for item in available
                        if item.from_code == pair_source and item.to_code == pair_target
                    ),
                    None,
                )
                if not package:
                    return False

                if progress_callback:
                    progress_callback(f"Pobieranie pakietu {pair_source}->{pair_target}...")
                argpackage.install_from_path(package.download())
                installed_any = True

            return installed_any
        except Exception:
            return False
