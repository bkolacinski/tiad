# Recipe Voice Filter

Aplikacja do filtrowania przepisów kulinarnych na podstawie mowy. Działa **w pełni offline**.

## Wymagania

- Python 3.10+ (Windows)
- ~1.5 GB wolnego miejsca (modele AI)

## Pierwsze uruchomienie

```bat
# 1. Zainstaluj zależności i pobierz modele
python setup.py

# 2. (Opcjonalnie) Zescrapuj więcej przepisów
python scraper.py

# 3. Uruchom aplikację
python app.py
```

## Budowanie .exe

```bat
build.bat
```

Gotowy plik: `dist\RecipeVoiceFilter\RecipeVoiceFilter.exe`

## Jak używać

1. Kliknij **Nagraj** i wymów składniki, np. *"kurczak, ziemniaki, czosnek"*
2. Kliknij **Zatrzymaj**
3. Aplikacja transkrybuje mowę, wykrywa składniki i wyświetla pasujące przepisy
4. Kliknij przepis, aby zobaczyć szczegóły i listę składników
5. (Opcjonalnie) Kliknij **Tłumacz**, aby przetłumaczyć transkrypcję na angielski

Możesz też wczytać plik audio (wav, mp3, ogg) przyciskiem **Wczytaj plik audio**.

## Tryby wyszukiwania

| Tryb | Opis |
|------|------|
| **Najlepsze** | Przepisy posortowane wg trafności (TF-IDF cosine similarity) |
| **Wszystkie** | Tylko przepisy zawierające **wszystkie** wymienione składniki |

## Technologie (w pełni offline)

| Komponent | Technologia |
|-----------|-------------|
| UI | customtkinter |
| STT | OpenAI Whisper `small` |
| Detekcja języka | Wbudowana w Whisper |
| NLP / lemmatyzacja | spaCy `pl_core_news_sm` |
| Fuzzy matching | rapidfuzz |
| Dopasowanie przepisów | TF-IDF + cosine similarity (scikit-learn) |
| Tłumaczenie | argostranslate (offline) |
| Pakowanie | PyInstaller |
