# Konwerter XLSX Desktop

Projekt został przerobiony z aplikacji webowej Flask na aplikację desktopową opartą o `tkinter`. Teraz nie uruchamiasz już lokalnego serwera ani przeglądarki. Zamiast tego otwierasz jedno okno programu, wybierasz plik `.xlsx`, ustawiasz parametry dokumentu i zapisujesz wynik bezpośrednio jako `DOCX` albo `PDF`.

## Co teraz robi aplikacja

1. Wybierasz plik wejściowy `.xlsx`.
2. Ustawiasz format wyjściowy: `DOCX` albo `PDF`.
3. Opcjonalnie wpisujesz tytuł dokumentu.
4. Ustawiasz wyrównanie, interlinię, odstęp po akapicie i numerację stron.
5. Klikasz `Konwertuj i zapisz`, a aplikacja pyta, gdzie zapisać gotowy plik.

Dodatkowo przycisk `Zapisz ustawienia` zapisuje domyślne ustawienia użytkownika, które będą automatycznie wczytywane przy następnym uruchomieniu.

## Wymagania

- Windows z Pythonem 3.11+ lub 3.13
- `tkinter` dostępny w standardowej instalacji Pythona na Windows

## Uruchomienie z Pythona

### Szybki start przez skrypt

Uruchom:

```bat
run.bat
```

Skrypt:

- utworzy lokalne środowisko `.venv`, jeśli go jeszcze nie ma,
- zainstaluje zależności,
- uruchomi aplikację desktopową.

### Ręcznie

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Budowanie `.exe`

Do budowania gotowego pliku wykonywalnego przygotowany jest skrypt:

```bat
build_exe.bat
```

Skrypt:

- aktywuje `.venv`,
- doinstaluje zależności potrzebne do budowy,
- uruchomi `PyInstaller`,
- zapisze gotowy plik w katalogu `dist\KonwerterXLSX.exe`.

Jeżeli chcesz wykonać to ręcznie:

```bat
.venv\Scripts\activate
pip install -r requirements-build.txt
python -m PyInstaller --noconfirm --clean --onefile --windowed --name KonwerterXLSX --collect-data docx app.py
```

## Gdzie zapisywane są ustawienia

Ustawienia użytkownika są zapisywane poza katalogiem projektu, dzięki czemu działają także po zbudowaniu `.exe`.

Na Windows trafiają do:

```text
%APPDATA%\KonwerterXLSX\settings.json
```

W tym samym katalogu aplikacja zapisuje też log:

```text
%APPDATA%\KonwerterXLSX\app.log
```

## Ważne uwagi

- Wejściem musi być plik `.xlsx`.
- Dla `DOCX` obowiązuje limit 63 kolumn w najszerszym arkuszu. Szersze arkusze zapisuj jako `PDF`.
- Wynikowy plik zapisujesz samodzielnie w wybranej lokalizacji przez okno systemowe.
- Aplikacja nie korzysta już z katalogu `templates/`, przeglądarki ani endpointów HTTP.

## Najważniejsze pliki projektu

- `app.py` - główna aplikacja desktopowa `tkinter`
- `converter_service.py` - wspólna logika ustawień, walidacji i konwersji plików
- `converters/` - logika generowania `DOCX`, `PDF` i odczytu `XLSX`
- `run.bat` - szybkie uruchomienie aplikacji
- `build_exe.bat` - budowanie gotowego pliku `.exe`
- `requirements.txt` - zależności uruchomieniowe
- `requirements-build.txt` - zależności do budowania `.exe`
