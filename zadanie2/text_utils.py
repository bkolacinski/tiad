"""Utilities for normalizing Polish ingredient text and comparing token forms."""

from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+", re.UNICODE)

COMMON_POLISH_SUFFIXES = (
    "owie",
    "ami",
    "owego",
    "owej",
    "owych",
    "owym",
    "ego",
    "emu",
    "ach",
    "ów",
    "om",
    "em",
    "ie",
    "u",
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text).lower())
    normalized = normalized.replace("/", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_words(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(normalize_text(text))]


def canonical_token(token: str) -> str:
    token = normalize_text(token)
    if len(token) <= 3:
        return token
    for suffix in COMMON_POLISH_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokens_match(left: str, right: str) -> bool:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if min(len(left_norm), len(right_norm)) >= 5 and (
        left_norm in right_norm or right_norm in left_norm
    ):
        return True

    left_stem = canonical_token(left_norm)
    right_stem = canonical_token(right_norm)
    if left_stem == right_stem and len(left_stem) >= 3:
        return True

    prefix_len = len(os_common_prefix(left_stem, right_stem))
    min_len = min(len(left_stem), len(right_stem))
    len_diff = abs(len(left_stem) - len(right_stem))
    # When words differ greatly in length (≥4 chars), require the prefix to cover
    # the full shorter stem — prevents "ziemniak" matching "ziemniaczana" etc.
    if len_diff >= 4:
        return prefix_len >= min_len
    return prefix_len >= max(4, min_len - 1)


def os_common_prefix(left: str, right: str) -> str:
    max_len = min(len(left), len(right))
    idx = 0
    while idx < max_len and left[idx] == right[idx]:
        idx += 1
    return left[:idx]
