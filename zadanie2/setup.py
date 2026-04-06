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
        print(f"BLAD: {cmd}")
        return False
    return True


def main():
    print("=" * 55)
    print("  Recipe Voice Filter - Setup")
    print("=" * 55)

    run(
        f'{sys.executable} -m pip install --upgrade pip "setuptools<82" wheel',
        "Aktualizowanie pip i setuptools...",
    )

    ok = run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalowanie pakietow Python...",
    )
    if not ok:
        print("  Probuje instalowac pakiety pojedynczo...")
        packages = [
            "openai-whisper",
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
        ]
        for package in packages:
            run(f"{sys.executable} -m pip install {package}", f"  Instalowanie {package}...")

    run(
        f"{sys.executable} -m spacy download pl_core_news_sm",
        "Pobieranie modelu spaCy (jezyk polski)...",
    )

    print("\n>>> Pobieranie modelu Whisper 'small' (~460 MB)...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        if whisper_model_ready():
            print("  Model Whisper juz jest w katalogu models.")
        else:
            model_path = os.path.join(MODELS_DIR, "small.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
            import whisper

            whisper.load_model("small", download_root=MODELS_DIR)
            print("  Model Whisper gotowy.")
    except Exception as exc:
        print(f"  BLAD Whisper: {exc}")

    print("\n>>> Instalowanie pakietow tlumaczen offline do katalogu projektu...")
    try:
        os.makedirs(TRANSLATIONS_DIR, exist_ok=True)
        os.environ["ARGOS_PACKAGES_DIR"] = TRANSLATIONS_DIR

        if translation_packages_ready():
            print("  Pakiety tlumaczen sa juz dostepne w translation_packages.")
            print("  Pomijam pobieranie.")
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
                    print(f"  Pobieranie {source}->{target}...")
                    argpackage.install_from_path(package.download())
                    print(f"  OK: {source}->{target}")
                else:
                    print(f"  Nie znaleziono pakietu {source}->{target}")
    except Exception as exc:
        print(f"  BLAD tlumaczen: {exc}")

    print("\n>>> Budowanie cache indeksu przepisow...")
    try:
        from recipe_matcher import RecipeMatcher

        matcher = RecipeMatcher(DATA_DIR)
        count = matcher.load(progress_callback=lambda msg: print(f"  {msg}"))
        print(f"  Cache gotowy. Przepisow: {count}")
    except Exception as exc:
        print(f"  BLAD cache indeksu: {exc}")

    print("\n" + "=" * 55)
    print("  Setup zakonczony!")
    print("  Uruchom: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
