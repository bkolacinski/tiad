"""Ingredient extraction from transcription text."""

from __future__ import annotations

import re
import threading
from typing import Iterable

from rapidfuzz import fuzz, process

from text_utils import canonical_token, normalize_text, tokenize_words, tokens_match

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
            except OSError:
                _nlp = None
    return _nlp


def extract_ingredients(text: str, known_vocab: set[str] | None = None) -> list[str]:
    text = normalize_text(text)
    text = re.sub(r"[^\w\sąćęłńóśźż-]", " ", text)

    nlp = _load_nlp()
    if nlp:
        return _extract_with_spacy(text, nlp, known_vocab or set())
    return _extract_fallback(text, known_vocab or set())


def _extract_with_spacy(text: str, nlp, known_vocab: set[str]) -> list[str]:
    candidates: list[str] = []
    doc = nlp(text)

    for token in doc:
        original = normalize_text(token.text)
        lemma = normalize_text(token.lemma_ or token.text)
        if not original:
            continue
        if token.is_space or token.is_punct:
            continue
        if len(original) <= 2 and len(lemma) <= 2:
            continue
        if lemma in NON_INGREDIENTS or original in NON_INGREDIENTS or token.is_stop:
            continue

        token_forms = [item for item in (lemma, original, canonical_token(lemma)) if item]
        in_vocab = _best_vocab_match(token_forms, known_vocab)
        if in_vocab:
            candidates.append(in_vocab)
            continue

        if token.pos_ in ("NOUN", "PROPN"):
            candidates.append(lemma)

    return _deduplicate(candidates, known_vocab)


def _extract_fallback(text: str, known_vocab: set[str]) -> list[str]:
    found: list[str] = []
    for word in tokenize_words(text):
        if len(word) <= 2 or word in NON_INGREDIENTS:
            continue
        match = _best_vocab_match([word, canonical_token(word)], known_vocab)
        found.append(match or word)
    return _deduplicate(found, known_vocab)


def _best_vocab_match(candidates: Iterable[str], known_vocab: set[str]) -> str | None:
    if not known_vocab:
        return None

    for candidate in candidates:
        for vocab_item in known_vocab:
            if tokens_match(candidate, vocab_item):
                return vocab_item

    for candidate in candidates:
        best = process.extractOne(candidate, known_vocab, scorer=fuzz.ratio, score_cutoff=78)
        if best:
            return best[0]
    return None


def _deduplicate(items: list[str], known_vocab: set[str] | None = None) -> list[str]:
    result: list[str] = []
    for item in items:
        if not item:
            continue

        normalized = normalize_text(item)
        if normalized in NON_INGREDIENTS:
            continue

        canonical = None
        if known_vocab:
            canonical = _best_vocab_match([item, normalized, canonical_token(normalized)], known_vocab)
        chosen = canonical or normalized

        if any(tokens_match(chosen, existing) for existing in result):
            continue
        result.append(chosen)
    return result
