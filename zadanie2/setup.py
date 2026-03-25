"""
Setup script - downloads required models and installs dependencies.
Run this ONCE before first use or before building .exe

Usage: python setup.py
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def run(cmd, desc=""):
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

    # 0. Upgrade pip + setuptools (fixes pkg_resources missing on old miniconda)
    run(
        f"{sys.executable} -m pip install --upgrade pip setuptools wheel",
        "Aktualizowanie pip i setuptools..."
    )

    # 1. Install Python packages
    ok = run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Instalowanie pakietow Python..."
    )
    if not ok:
        # Fallback: install one by one skipping failures
        print("  Probuje instalowac pakiety pojedynczo...")
        packages = [
            "openai-whisper", "customtkinter", "sounddevice", "soundfile",
            "numpy", "scipy", "scikit-learn", "rapidfuzz", "spacy",
            "argostranslate", "Pillow",
        ]
        for pkg in packages:
            run(f"{sys.executable} -m pip install {pkg}", f"  Instalowanie {pkg}...")

    # 2. Download spaCy Polish model
    run(
        f"{sys.executable} -m spacy download pl_core_news_sm",
        "Pobieranie modelu spaCy (jezyk polski)..."
    )

    # 3. Pre-download Whisper model
    print("\n>>> Pobieranie modelu Whisper 'small' (~460 MB)...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    try:
        import whisper
        whisper.load_model("small", download_root=MODELS_DIR)
        print("  Model Whisper gotowy.")
    except Exception as e:
        print(f"  BLAD Whisper: {e}")

    # 4. Install argostranslate packages (pl<->en, pl<->de)
    print("\n>>> Instalowanie pakietow tlumaczen offline (pl<->en, pl<->de)...")
    try:
        from argostranslate import package as argpkg
        argpkg.update_package_index()
        available = argpkg.get_available_packages()
        pairs = [("pl", "en"), ("en", "pl"), ("pl", "de"), ("de", "pl")]
        for src, tgt in pairs:
            pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
            if pkg:
                print(f"  Pobieranie {src}->{tgt}...")
                argpkg.install_from_path(pkg.download())
                print(f"  OK: {src}->{tgt}")
            else:
                print(f"  Nie znaleziono pakietu {src}->{tgt}")
    except Exception as e:
        print(f"  BLAD tlumaczen: {e}")

    print("\n" + "=" * 55)
    print("  Setup zakończony!")
    print("  Uruchom: python app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
