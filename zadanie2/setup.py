"""Prepare local runtime assets for the app and portable EXE."""

from __future__ import annotations

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRANSLATIONS_DIR = os.path.join(BASE_DIR, "translation_packages")
DATA_DIR = os.path.join(BASE_DIR, "data")
REQUIRED_TRANSLATION_PAIRS = {
    "translate-pl_en-1_9",
    "translate-en_pl-1_9",
    "en_de",
    "de_en",
    "translate-en_fr-1_9",
    "translate-fr_en-1_9",
    "en_es",
    "translate-es_en-1_9",
}


def whisper_model_ready() -> bool:
    model_path = os.path.join(MODELS_DIR, "small.pt")
    return os.path.exists(model_path) and os.path.getsize(model_path) > 300_000_000


VOSK_MODEL_ZIPS = [
    ("https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip", "pl"),
    ("https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", "en"),
]


def vosk_model_ready(lang: str) -> bool:
    """True if a Vosk model is unpacked under ``models/vosk/<lang>`` (``am/final.mdl``)."""
    return os.path.isfile(
        os.path.join(MODELS_DIR, "vosk", lang, "am", "final.mdl")
    )


def ensure_vosk_models() -> None:
    """Download and unpack small Vosk PL+EN models into ``models/vosk/``."""
    import shutil
    import tempfile
    import urllib.request
    import zipfile

    os.makedirs(os.path.join(MODELS_DIR, "vosk"), exist_ok=True)
    for url, lang in VOSK_MODEL_ZIPS:
        if vosk_model_ready(lang):
            print(f"  Vosk model ({lang}) already present at models/vosk/{lang}.")
            continue
        print(f"  Downloading Vosk model ({lang})...")
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".zip")
            os.close(fd)
            urllib.request.urlretrieve(url, tmp_path)
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    zf.extractall(td)
                entries = [
                    e
                    for e in os.listdir(td)
                    if not e.startswith(".") and os.path.isdir(os.path.join(td, e))
                ]
                if len(entries) != 1:
                    raise RuntimeError(f"Unexpected archive layout: {url}")
                inner = os.path.join(td, entries[0])
                dest = os.path.join(MODELS_DIR, "vosk", lang)
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                shutil.copytree(inner, dest)
            print(f"  OK: Vosk ({lang}).")
        except Exception as exc:
            print(f"  Vosk error ({lang}): {exc}")
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def translation_packages_ready() -> bool:
    if not os.path.isdir(TRANSLATIONS_DIR):
        return False
    installed = {
        name
        for name in os.listdir(TRANSLATIONS_DIR)
        if os.path.isdir(os.path.join(TRANSLATIONS_DIR, name))
    }
    return REQUIRED_TRANSLATION_PAIRS.issubset(installed)


def run(cmd: str, desc: str = "") -> bool:
    if desc:
        print(f"\n>>> {desc}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: {cmd}")
        return False
    return True


def main():
    print("=" * 55)
    print("  Recipe Voice Filter - Setup")
    print("=" * 55)

    run(
        f'{sys.executable} -m pip install --upgrade pip "setuptools<82" wheel',
        "Upgrading pip and setuptools...",
    )

    ok = run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages...",
    )
    if not ok:
        print("  Trying to install packages one by one...")
        packages = [
            "openai-whisper",
            "transformers",
            "customtkinter",
            "sounddevice",
            "soundfile",
            "numpy",
            "scipy",
            "scikit-learn",
            "rapidfuzz",
            "spacy",
            "argostranslate",
            "Pillow",
            "vosk",
        ]
        for package in packages:
            run(f"{sys.executable} -m pip install {package}", f"  Installing {package}...")

    run(
        f"{sys.executable} -m spacy download pl_core_news_sm",
        "Downloading spaCy model (Polish)...",
    )

    print("\n>>> Downloading Whisper 'small' model (~460 MB)...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        if whisper_model_ready():
            print("  Whisper model already in models directory.")
        else:
            model_path = os.path.join(MODELS_DIR, "small.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
            import whisper

            whisper.load_model("small", download_root=MODELS_DIR)
            print("  Whisper model ready.")
    except Exception as exc:
        print(f"  Whisper error: {exc}")

    print("\n>>> Downloading Vosk models (PL + EN, ~90 MB total)...")
    try:
        ensure_vosk_models()
    except Exception as exc:
        print(f"  Vosk error: {exc}")

    print("\n>>> Hugging Face (Wav2Vec2 XLS-R + Meta MMS-1B)")
    print("  The transformers package is listed in requirements.")
    print("  Model weights download on first transcription to models/hf/ (several GB).")

    print("\n>>> Installing offline translation packages into the project directory...")
    try:
        os.makedirs(TRANSLATIONS_DIR, exist_ok=True)
        os.environ["ARGOS_PACKAGES_DIR"] = TRANSLATIONS_DIR

        if translation_packages_ready():
            print("  Translation packages already available in translation_packages.")
            print("  Skipping download.")
        else:
            from argostranslate import package as argpackage

            argpackage.update_package_index()
            available = argpackage.get_available_packages()
            pairs = [
                ("pl", "en"),
                ("en", "pl"),
                ("en", "de"),
                ("de", "en"),
                ("en", "fr"),
                ("fr", "en"),
                ("en", "es"),
                ("es", "en"),
            ]
            for source, target in pairs:
                package = next(
                    (
                        item
                        for item in available
                        if item.from_code == source and item.to_code == target
                    ),
                    None,
                )
                if package:
                    print(f"  Downloading {source}->{target}...")
                    argpackage.install_from_path(package.download())
                    print(f"  OK: {source}->{target}")
                else:
                    print(f"  Package not found: {source}->{target}")
    except Exception as exc:
        print(f"  Translation error: {exc}")

    print("\n>>> Building recipe index cache...")
    try:
        from recipe_matcher import RecipeMatcher

        matcher = RecipeMatcher(DATA_DIR)
        count = matcher.load(progress_callback=lambda msg: print(f"  {msg}"))
        print(f"  Cache ready. Recipes: {count}")
    except Exception as exc:
        print(f"  Index cache error: {exc}")

    print("\n" + "=" * 55)
    print("  Setup finished!")
    print("  Uruchom: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
