"""
Ingredient extraction from transcribed text using spaCy Polish model.
Extracts nouns, lemmatizes them, and filters against a known ingredient vocabulary.
"""

import re
import threading
from typing import List, Set

_nlp = None
_nlp_lock = threading.Lock()

# Common Polish stop-words and non-ingredient words to filter out
NON_INGREDIENTS = {
    "składnik", "przepis", "danie", "potrawa", "jedzenie", "gotowanie",
    "chcę", "chciałbym", "chciałabym", "potrzebuję", "mam", "lubię",
    "zrobić", "ugotować", "przygotować", "lista", "coś", "czegoś",
    "dodać", "użyć", "zawierać", "potrzeba", "proszę", "szukam",
    "przepisy", "składniki", "pokazać", "znaleźć", "jak", "to", "się",
    "który", "która", "które", "tego", "tej", "temu",
}


def _load_nlp():
    global _nlp
    with _nlp_lock:
        if _nlp is None:
            try:
                import spacy
                _nlp = spacy.load("pl_core_news_sm")
            except OSError:
                # Model not downloaded - fall back to regex-based extraction
                _nlp = None
    return _nlp


def extract_ingredients(text: str, known_vocab: Set[str] = None) -> List[str]:
    """
    Extract ingredient words from Polish transcription text.

    Strategy:
    1. Try spaCy NLP (lemmatization + POS tagging)
    2. Fall back to simple tokenization with known vocab matching
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\sąćęłńóśźżĄĆĘŁŃÓŚŹŻ-]", " ", text)

    nlp = _load_nlp()
    if nlp:
        return _extract_with_spacy(text, nlp, known_vocab)
    return _extract_fallback(text, known_vocab)


def _extract_with_spacy(text: str, nlp, known_vocab: Set[str]) -> List[str]:
    """Use spaCy for lemmatization and noun extraction."""
    doc = nlp(text)
    candidates = []

    for token in doc:
        # Take nouns and proper nouns
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemma = token.lemma_.lower()
            if lemma not in NON_INGREDIENTS and len(lemma) > 2:
                candidates.append(lemma)

    return _deduplicate(candidates, known_vocab)


def _extract_fallback(text: str, known_vocab: Set[str]) -> List[str]:
    """Simple tokenization fallback when spaCy is unavailable."""
    words = text.split()
    if known_vocab:
        # Match against known ingredient vocabulary with fuzzy fallback
        from rapidfuzz import process, fuzz
        found = []
        for word in words:
            if len(word) <= 2:
                continue
            if word in known_vocab:
                found.append(word)
                continue
            match = process.extractOne(
                word, known_vocab,
                scorer=fuzz.ratio,
                score_cutoff=80
            )
            if match:
                found.append(match[0])
        return _deduplicate(found, known_vocab)

    # Last resort: return all words longer than 3 chars not in stop list
    return [w for w in words if len(w) > 3 and w not in NON_INGREDIENTS]


def _deduplicate(items: List[str], known_vocab: Set[str] = None) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
