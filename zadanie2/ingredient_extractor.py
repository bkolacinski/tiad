"""Ingredient extraction from transcription text."""

from __future__ import annotations

import logging
import re
import threading
from typing import Iterable

from rapidfuzz import fuzz, process

from text_utils import canonical_token, normalize_text, tokenize_words, tokens_match

logger = logging.getLogger("zadanie2.extractor")

# Shared with recipe_matcher.FUZZY_MIN_RATIO. Kept as its own constant
# so this module is importable without the matcher.
FUZZY_MIN_RATIO = 78

# Drop reason labels — used by the GUI to explain why a token was filtered.
DROP_SHORT = "too_short"
DROP_STOP = "stop_word"
DROP_NON_NOUN = "non_noun"
DROP_NON_INGREDIENT = "non_ingredient"
DROP_DUPLICATE = "duplicate"

# Human-readable PL labels for each drop reason — shown in the ingredient
# pills area so the user understands why a word was ignored.
DROP_REASON_LABELS_PL = {
    DROP_SHORT: "zbyt krótkie",
    DROP_STOP: "słowo pomocnicze",
    DROP_NON_NOUN: "nie jest rzeczownikiem",
    DROP_NON_INGREDIENT: "nie jest składnikiem",
    DROP_DUPLICATE: "duplikat",
}

_nlp = None
_nlp_lock = threading.Lock()

NON_INGREDIENTS = {
    "składnik",
    "składniki",
    "przepis",
    "przepisy",
    "danie",
    "potrawa",
    "jedzenie",
    "chce",
    "chcę",
    "chcialbym",
    "chciałbym",
    "chcialabym",
    "chciałabym",
    "prosze",
    "proszę",
    "szukam",
    "poprosze",
    "poproszę",
    "pokaz",
    "pokaż",
    "znajdz",
    "znajdź",
    "lista",
    "dodac",
    "dodać",
    "zrobic",
    "zrobić",
    "ugotowac",
    "ugotować",
    "przygotowac",
    "przygotować",
}


def _load_nlp():
    global _nlp
    with _nlp_lock:
        if _nlp is None:
            try:
                import spacy

                _nlp = spacy.load("pl_core_news_sm")
            except OSError as exc:
                logger.info("spaCy pl_core_news_sm unavailable, fallback tokenizer: %s", exc)
                _nlp = False
    return _nlp or None


def extract_ingredients(text: str, known_vocab: set[str] | None = None) -> list[str]:
    """Backwards-compatible wrapper — returns only the kept ingredients."""
    kept, _ = extract_ingredients_with_drops(text, known_vocab)
    return kept


def extract_ingredients_with_drops(
    text: str, known_vocab: set[str] | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    """Return (kept_ingredients, dropped) where dropped is [(word, reason), ...].

    Reason codes use the DROP_* constants above. The GUI consumes this to
    explain to the user why a spoken word did not become a filter term.
    """
    text = normalize_text(text)
    text = re.sub(r"[^\w\sąćęłńóśźż-]", " ", text)
    vocab = known_vocab or set()

    nlp = _load_nlp()
    if nlp:
        candidates, dropped = _extract_with_spacy(text, nlp, vocab)
    else:
        candidates, dropped = _extract_fallback(text, vocab)

    kept, dedup_drops = _deduplicate(candidates, vocab)
    dropped.extend(dedup_drops)
    return kept, dropped


def _extract_with_spacy(
    text: str, nlp, known_vocab: set[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    candidates: list[str] = []
    dropped: list[tuple[str, str]] = []
    doc = nlp(text)

    for token in doc:
        original = normalize_text(token.text)
        lemma = normalize_text(token.lemma_ or token.text)
        if not original:
            continue
        if token.is_space or token.is_punct:
            continue
        display = lemma or original
        if len(original) <= 2 and len(lemma) <= 2:
            if display:
                dropped.append((display, DROP_SHORT))
            continue
        if lemma in NON_INGREDIENTS or original in NON_INGREDIENTS:
            dropped.append((display, DROP_NON_INGREDIENT))
            continue
        if token.is_stop:
            dropped.append((display, DROP_STOP))
            continue

        token_forms = [item for item in (lemma, original, canonical_token(lemma)) if item]
        in_vocab = _best_vocab_match(token_forms, known_vocab)
        if in_vocab:
            candidates.append(in_vocab)
            continue

        if token.pos_ in ("NOUN", "PROPN"):
            candidates.append(lemma)
        else:
            dropped.append((display, DROP_NON_NOUN))

    return candidates, dropped


def _extract_fallback(
    text: str, known_vocab: set[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    found: list[str] = []
    dropped: list[tuple[str, str]] = []
    for word in tokenize_words(text):
        if len(word) <= 2:
            dropped.append((word, DROP_SHORT))
            continue
        if word in NON_INGREDIENTS:
            dropped.append((word, DROP_NON_INGREDIENT))
            continue
        match = _best_vocab_match([word, canonical_token(word)], known_vocab)
        found.append(match or word)
    return found, dropped


def _best_vocab_match(candidates: Iterable[str], known_vocab: set[str]) -> str | None:
    if not known_vocab:
        return None

    for candidate in candidates:
        for vocab_item in known_vocab:
            if tokens_match(candidate, vocab_item):
                return vocab_item

    for candidate in candidates:
        best = process.extractOne(
            candidate, known_vocab, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN_RATIO
        )
        if best:
            return best[0]
    return None


def _deduplicate(
    items: list[str], known_vocab: set[str] | None = None
) -> tuple[list[str], list[tuple[str, str]]]:
    result: list[str] = []
    dropped: list[tuple[str, str]] = []
    for item in items:
        if not item:
            continue

        normalized = normalize_text(item)
        if normalized in NON_INGREDIENTS:
            dropped.append((normalized, DROP_NON_INGREDIENT))
            continue

        canonical = None
        if known_vocab:
            canonical = _best_vocab_match([item, normalized, canonical_token(normalized)], known_vocab)
        chosen = canonical or normalized

        if any(tokens_match(chosen, existing) for existing in result):
            dropped.append((chosen, DROP_DUPLICATE))
            continue
        result.append(chosen)
    return result, dropped
