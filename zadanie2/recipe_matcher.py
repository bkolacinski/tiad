"""
Recipe matching engine using TF-IDF + cosine similarity.
No LLM - pure classical IR/NLP approach.
"""

import json
import os
import glob
import threading
from typing import List, Dict, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process


class RecipeMatcher:
    """
    Matches spoken ingredients against a recipe database using TF-IDF.

    Pipeline:
    1. Load recipes from JSON files
    2. Build TF-IDF index on ingredient strings
    3. For a query (list of ingredients), compute cosine similarity
    4. Return ranked list of matching recipes
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.recipes: List[Dict] = []
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix = None
        self.ingredient_vocab: Set[str] = set()
        self._lemmatized: List[str] = []   # lemmatized ingredient text per recipe
        self._lock = threading.Lock()
        self._loaded = False

    def load(self, progress_callback=None) -> int:
        """Load all recipe JSON files from data_dir. Returns recipe count."""
        with self._lock:
            self.recipes = []
            pattern = os.path.join(self.data_dir, "recipes*.json")
            files = sorted(glob.glob(pattern))

            if not files:
                # Try a single recipes.json
                single = os.path.join(self.data_dir, "recipes.json")
                if os.path.exists(single):
                    files = [single]

            if not files:
                return 0

            for filepath in files:
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.recipes.extend(data)
                    elif isinstance(data, dict) and "recipes" in data:
                        self.recipes.extend(data["recipes"])
                except Exception:
                    continue

            if not self.recipes:
                return 0

            if progress_callback:
                progress_callback(f"Indeksowanie {len(self.recipes)} przepisów...")

            self._build_index()
            self._loaded = True
            return len(self.recipes)

    def _lemmatize_corpus(self, texts: List[str]) -> List[str]:
        """
        Lemmatize ingredient strings using spaCy.
        "2 szklanki mąki" -> "2 szklanka mąka"
        "dwie marchewki" -> "dwa marchewka"
        Falls back to original text if spaCy unavailable.
        """
        try:
            import spacy
            nlp = spacy.load("pl_core_news_sm", disable=["ner", "parser", "senter"])
            result = []
            for doc in nlp.pipe(texts, batch_size=256):
                lemmas = " ".join(t.lemma_.lower() for t in doc if not t.is_space)
                result.append(lemmas)
            return result
        except Exception:
            return texts

    def _build_index(self):
        """Build TF-IDF index from lemmatized recipe ingredients."""
        raw_corpus = []
        for recipe in self.recipes:
            ingredients = recipe.get("ingredients", [])
            if isinstance(ingredients, list):
                text = " ".join(str(i).lower() for i in ingredients)
            else:
                text = str(ingredients).lower()
            raw_corpus.append(text)

        # Lemmatize: "marchewki" -> "marchewka", "wody" -> "woda"
        self._lemmatized = self._lemmatize_corpus(raw_corpus)

        # Collect vocabulary from lemmatized text
        for text in self._lemmatized:
            for word in text.split():
                if len(word) > 2:
                    self.ingredient_vocab.add(word)

        # char n-grams on lemmatized text - robust to remaining morphology
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=100000,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self._lemmatized)

    def search(
        self,
        ingredients: List[str],
        mode: str = "any",   # "all" = must have all, "any" = ranked by count
        top_n: int = 20,
    ) -> List[Dict]:
        """
        Search recipes by ingredient list.

        mode="all": only recipes containing ALL ingredients (strict filter)
        mode="any": rank by how many ingredients match (TF-IDF cosine)
        """
        if not self._loaded or not self.recipes:
            return []

        ingredients = [i.lower() for i in ingredients if i]

        if mode == "all":
            return self._filter_all(ingredients, top_n)
        return self._rank_by_similarity(ingredients, top_n)

    def _filter_all(self, ingredients: List[str], top_n: int) -> List[Dict]:
        """Return recipes that contain ALL queried ingredients (using lemmatized text)."""
        results = []
        for idx, recipe in enumerate(self.recipes):
            lem_text = self._lemmatized[idx] if idx < len(self._lemmatized) else ""
            if all(self._ingredient_in_text(ing, lem_text) for ing in ingredients):
                score = self._similarity_score(ingredients, lem_text)
                results.append({**recipe, "_score": score})

        results.sort(key=lambda r: r["_score"], reverse=True)
        return results[:top_n]

    def _rank_by_similarity(self, ingredients: List[str], top_n: int) -> List[Dict]:
        """Rank recipes by TF-IDF cosine similarity to ingredient query."""
        query = " ".join(ingredients)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({**self.recipes[idx], "_score": float(scores[idx])})
        return results

    def _ingredient_in_text(self, ingredient: str, recipe_text: str) -> bool:
        """
        Check if ingredient (already lemmatized) appears in lemmatized recipe text.
        Handles: exact match, substring (kurczak in kurczaka), fuzzy.
        """
        if ingredient in recipe_text:
            return True
        words = recipe_text.split()
        # Substring: "kurczak" matches "kurczaka", "marchewk" matches "marchewka"
        if any(ingredient in word or word in ingredient for word in words if len(word) > 2):
            return True
        # Fuzzy: catches remaining morphological variants and slight typos
        match = process.extractOne(ingredient, words, scorer=fuzz.partial_ratio, score_cutoff=72)
        return match is not None

    def _similarity_score(self, ingredients: List[str], recipe_text: str) -> float:
        """Fraction of queried ingredients found in recipe."""
        count = sum(1 for ing in ingredients if self._ingredient_in_text(ing, recipe_text))
        return count / max(len(ingredients), 1)

    @property
    def vocabulary(self) -> Set[str]:
        return self.ingredient_vocab

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def recipe_count(self) -> int:
        return len(self.recipes)
