"""Recipe matching engine based on normalized ingredient terms."""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import threading
from collections import Counter
from typing import Dict

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from text_utils import canonical_token, normalize_text, tokenize_words, tokens_match

_NLP = None
_NLP_LOCK = threading.Lock()

UNIT_WORDS = {
    "g",
    "kg",
    "dag",
    "ml",
    "l",
    "łyżka",
    "łyżki",
    "łyżeczka",
    "łyżeczki",
    "szklanka",
    "szklanki",
    "szczypta",
    "szczypty",
    "opakowanie",
    "opakowania",
    "puszka",
    "puszki",
    "kostka",
    "kostki",
    "ząbek",
    "ząbki",
    "sztuka",
    "sztuki",
    "sztuk",
    "pęczek",
    "pęczki",
}

NOISE_WORDS = {
    "około",
    "niecałe",
    "niecała",
    "dowolny",
    "dowolnych",
    "ulubionej",
    "ulubionego",
    "mały",
    "mała",
    "małe",
    "duży",
    "duża",
    "duże",
    "średni",
    "średnia",
    "średnie",
    "świeży",
    "świeża",
    "świeże",
    "mrożony",
    "mrożona",
    "mrożone",
    "drobno",
    "starty",
    "starta",
    "startego",
    "gotowej",
    "gotowy",
    "gotowe",
    "obrany",
    "obrana",
    "obrane",
    "przyprawy",
    "można",
    "pominąć",
    "lub",
    "albo",
    "oraz",
    "i",
    "po",
    "do",
    "ze",
    "z",
    "na",
    "bez",
}


def _load_nlp():
    global _NLP
    with _NLP_LOCK:
        if _NLP is None:
            try:
                import spacy

                _NLP = spacy.load("pl_core_news_sm")
            except OSError:
                _NLP = False
    return _NLP or None


class RecipeMatcher:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.cache_path = os.path.join(self.data_dir, "recipe_index_cache.pkl")
        self.recipes: list[Dict] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None
        self.ingredient_vocab: set[str] = set()
        self._lemmatized: list[str] = []
        self._recipe_terms: list[set[str]] = []
        self._lock = threading.Lock()
        self._loaded = False

    def load(self, progress_callback=None) -> int:
        with self._lock:
            self.recipes = []
            self.ingredient_vocab = set()
            self._lemmatized = []
            self._recipe_terms = []

            if self._load_cache():
                if progress_callback:
                    progress_callback("Wczytano gotowy indeks przepisów.")
                self._loaded = True
                return len(self.recipes)

            pattern = os.path.join(self.data_dir, "recipes*.json")
            files = sorted(glob.glob(pattern))
            if not files:
                single = os.path.join(self.data_dir, "recipes.json")
                if os.path.exists(single):
                    files = [single]
            if not files:
                return 0

            for filepath in files:
                try:
                    with open(filepath, encoding="utf-8") as handle:
                        data = json.load(handle)
                except Exception:
                    continue

                if isinstance(data, list):
                    self.recipes.extend(data)
                elif isinstance(data, dict) and "recipes" in data:
                    self.recipes.extend(data["recipes"])

            if not self.recipes:
                return 0

            if progress_callback:
                progress_callback(f"Indeksowanie {len(self.recipes)} przepisów...")

            self._build_index()
            self._save_cache()
            self._loaded = True
            return len(self.recipes)

    def _source_state(self) -> list[tuple[str, int]]:
        """Stan zrodel JSON: (nazwa_pliku, rozmiar). Bez mtime — po spakowaniu do EXE
        PyInstaller ustawia nowe znaczniki czasu i cache z buildu nigdy by nie pasowal."""
        files = sorted(glob.glob(os.path.join(self.data_dir, "recipes*.json")))
        state: list[tuple[str, int]] = []
        for filepath in files:
            try:
                stat = os.stat(filepath)
            except OSError:
                continue
            state.append((os.path.basename(filepath), int(stat.st_size)))
        return state

    def _canonical_state_from_stored(self, stored) -> list[tuple[str, int]]:
        if not stored:
            return []
        out: list[tuple[str, int]] = []
        for item in stored:
            if len(item) == 3:
                out.append((item[0], int(item[2])))
            elif len(item) == 2:
                out.append((item[0], int(item[1])))
        return sorted(out)

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "rb") as handle:
                payload = pickle.load(handle)
            current = sorted(self._source_state())
            if self._canonical_state_from_stored(payload.get("source_state")) != current:
                return False

            self.recipes = payload["recipes"]
            self.vectorizer = payload["vectorizer"]
            self.tfidf_matrix = payload["tfidf_matrix"]
            self.ingredient_vocab = payload["ingredient_vocab"]
            self._lemmatized = payload["lemmatized"]
            self._recipe_terms = payload["recipe_terms"]
            return True
        except Exception:
            return False

    def _save_cache(self):
        payload = {
            "source_state": self._source_state(),
            "recipes": self.recipes,
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "ingredient_vocab": self.ingredient_vocab,
            "lemmatized": self._lemmatized,
            "recipe_terms": self._recipe_terms,
        }
        with open(self.cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _build_index(self):
        raw_terms_per_recipe: list[set[str]] = []
        previews_per_recipe: list[list[str]] = []
        counter = Counter()

        for recipe in self.recipes:
            raw_ingredients = recipe.get("ingredients", [])
            if not isinstance(raw_ingredients, list):
                raw_ingredients = [raw_ingredients]

            terms = self._extract_recipe_terms(raw_ingredients)
            preview = self._build_preview_labels(raw_ingredients)
            raw_terms_per_recipe.append(terms)
            previews_per_recipe.append(preview)
            counter.update(terms)

        aliases = self._build_term_aliases(counter)

        corpus: list[str] = []
        for recipe, raw_terms, preview in zip(self.recipes, raw_terms_per_recipe, previews_per_recipe):
            canonical_terms = {aliases.get(term, term) for term in raw_terms}
            recipe["_search_terms"] = sorted(canonical_terms)
            recipe["_preview_ingredients"] = preview

            search_text = " ".join(sorted(canonical_terms))
            self._recipe_terms.append(canonical_terms)
            self._lemmatized.append(search_text)
            corpus.append(search_text or normalize_text(" ".join(str(i) for i in recipe.get("ingredients", []))))
            self.ingredient_vocab.update(canonical_terms)

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=100000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def _extract_recipe_terms(self, ingredients: list[str]) -> set[str]:
        terms: set[str] = set()
        nlp = _load_nlp()

        for line in ingredients:
            cleaned = self._clean_ingredient_line(line)
            if not cleaned:
                continue

            if nlp:
                doc = nlp(cleaned)
                noun_found = False
                for token in doc:
                    if token.pos_ not in ("NOUN", "PROPN"):
                        continue
                    token_text = normalize_text(token.lemma_ or token.text)
                    if not self._keep_token(token_text):
                        continue
                    noun_found = True
                    terms.add(token_text)
                if noun_found:
                    continue
            for token in tokenize_words(cleaned):
                if self._keep_token(token):
                    terms.add(canonical_token(token))

        return terms

    def _build_preview_labels(self, ingredients: list[str]) -> list[str]:
        labels: list[str] = []
        for line in ingredients:
            label = self._short_ingredient_label(line)
            if label and label not in labels:
                labels.append(label)
        return labels

    def _build_term_aliases(self, counter: Counter) -> dict[str, str]:
        aliases: dict[str, str] = {}
        representatives: list[str] = []

        sorted_terms = sorted(
            counter.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        for term, _ in sorted_terms:
            matched_rep = next((rep for rep in representatives if tokens_match(term, rep)), None)
            if matched_rep:
                aliases[term] = matched_rep
            else:
                aliases[term] = term
                representatives.append(term)
        return aliases

    def _clean_ingredient_line(self, line: str) -> str:
        text = normalize_text(line)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.split(r"\s[-;]\s", text, maxsplit=1)[0]
        text = re.sub(r"\b\d+[.,]?\d*\b", " ", text)
        text = re.sub(r"%", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _short_ingredient_label(self, line: str) -> str:
        text = self._clean_ingredient_line(line)
        words = []
        for word in tokenize_words(text):
            if self._keep_token(word, allow_units=False):
                words.append(word)
            if len(words) >= 5:
                break
        if not words:
            return str(line).strip()
        label = " ".join(words)
        return label[:46].rstrip() + ("…" if len(label) > 46 else "")

    def _keep_token(self, token: str, allow_units: bool = False) -> bool:
        token = normalize_text(token)
        if len(token) <= 2:
            return False
        if token in NOISE_WORDS:
            return False
        if not allow_units and token in UNIT_WORDS:
            return False
        if token.endswith(("ać", "eć", "ić", "yć", "ować")):
            return False
        return True

    def search(self, ingredients: list[str], mode: str = "any", top_n: int = 20) -> list[Dict]:
        if not self._loaded or not self.recipes:
            return []

        normalized = self._normalize_query_ingredients(ingredients)
        if not normalized:
            return []

        if mode == "all":
            return self._filter_all(normalized, top_n)
        return self._rank_by_similarity(normalized, top_n)

    def _normalize_query_ingredients(self, ingredients: list[str]) -> list[str]:
        result: list[str] = []
        for ingredient in ingredients:
            ingredient = normalize_text(ingredient)
            if not ingredient:
                continue
            if any(tokens_match(ingredient, existing) for existing in result):
                continue
            result.append(ingredient)
        return result

    def _filter_all(self, ingredients: list[str], top_n: int) -> list[Dict]:
        results = []
        for idx, recipe in enumerate(self.recipes):
            if all(self._ingredient_matches_recipe(ing, idx) for ing in ingredients):
                score = self._similarity_score(ingredients, idx)
                results.append({**recipe, "_score": score, "_idx": idx})
        results.sort(key=lambda item: (item["_score"], item.get("title", "")), reverse=True)
        if results:
            return results[:top_n]

        minimum_matches = max(2, min(len(ingredients) - 1, (len(ingredients) + 1) // 2))
        relaxed = []
        for idx, recipe in enumerate(self.recipes):
            matched_count = self.match_count(ingredients, idx)
            if matched_count < minimum_matches:
                continue
            relaxed.append(
                {
                    **recipe,
                    "_score": matched_count / len(ingredients),
                    "_idx": idx,
                    "_strict_relaxed": True,
                }
            )
        relaxed.sort(key=lambda item: (item["_score"], item.get("title", "")), reverse=True)
        return relaxed[:top_n]

    def _rank_by_similarity(self, ingredients: list[str], top_n: int) -> list[Dict]:
        query = " ".join(ingredients)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            matched_count = self.match_count(ingredients, idx)
            if matched_count == 0:
                continue
            results.append({**self.recipes[idx], "_score": float(scores[idx]), "_idx": int(idx)})
        return results

    def match_count(self, ingredients: list[str], recipe_idx: int) -> int:
        return len(self.matched_ingredients(ingredients, recipe_idx))

    def matched_ingredients(self, ingredients: list[str], recipe_idx: int) -> list[str]:
        if not (0 <= recipe_idx < len(self._recipe_terms)):
            return []
        matched = []
        for ingredient in self._normalize_query_ingredients(ingredients):
            if self._ingredient_matches_recipe(ingredient, recipe_idx):
                matched.append(ingredient)
        return matched

    def _ingredient_matches_recipe(self, ingredient: str, recipe_idx: int) -> bool:
        if not (0 <= recipe_idx < len(self._recipe_terms)):
            return False

        recipe_terms = self._recipe_terms[recipe_idx]
        for recipe_term in recipe_terms:
            if tokens_match(ingredient, recipe_term):
                return True
            if fuzz.ratio(ingredient, recipe_term) >= 78:
                return True
        return False

    def lemmatized_text(self, recipe_idx: int) -> str:
        if 0 <= recipe_idx < len(self._lemmatized):
            return self._lemmatized[recipe_idx]
        return ""

    def preview_ingredients(self, recipe: Dict) -> list[str]:
        return recipe.get("_preview_ingredients", recipe.get("ingredients", []))

    def _ingredient_in_text(self, ingredient: str, recipe_text: str) -> bool:
        for recipe_term in tokenize_words(recipe_text):
            if tokens_match(ingredient, recipe_term):
                return True
            if fuzz.ratio(normalize_text(ingredient), recipe_term) >= 78:
                return True
        return False

    def _similarity_score(self, ingredients: list[str], recipe_idx: int) -> float:
        if not ingredients:
            return 0.0
        matched = self.match_count(ingredients, recipe_idx)
        return matched / len(ingredients)

    @property
    def vocabulary(self) -> set[str]:
        if self.ingredient_vocab:
            return self.ingredient_vocab
        counter = Counter()
        for recipe in self.recipes:
            for item in recipe.get("_search_terms", []):
                counter[item] += 1
        return set(counter.keys())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def recipe_count(self) -> int:
        return len(self.recipes)
