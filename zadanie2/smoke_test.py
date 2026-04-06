"""Quick local validation before demo/build."""

from pathlib import Path

from ingredient_extractor import extract_ingredients
from recipe_matcher import RecipeMatcher
from translator import configure_translation_packages, get_available_targets, translate_text


def _translation_dir() -> str:
    root = Path(__file__).resolve().parent
    candidates = [
        root / "translation_packages",
        root / "release" / "RecipeVoiceFilter" / "_internal" / "translation_packages",
        root / "release" / "RecipeVoiceFilter" / "translation_packages",
    ]
    for path in candidates:
        if path.exists() and any(path.iterdir()):
            return str(path)
    return str(root / "translation_packages")


def main():
    configure_translation_packages(_translation_dir())

    matcher = RecipeMatcher("data")
    count = matcher.load()
    assert count > 0, "Nie wczytano bazy przepisow"

    cases = {
        "chce kurczaka z cebula i czosnkiem": {"kurczak", "cebula", "czosnek"},
        "poprosze przepis z pomidorami, makaronem i serem": {"pomidor", "makaron", "ser"},
        "marchewki ziemniakow cebuli": {"marchewka", "ziemniak", "cebula"},
    }
    for text, expected in cases.items():
        found = set(extract_ingredients(text, matcher.vocabulary))
        missing = expected - found
        assert not missing, f"Brak skladnikow dla '{text}': {sorted(missing)}"

    ranked_results = matcher.search(["kurczak", "cebula", "czosnek"], top_n=10)
    assert ranked_results, "Ranking przepisow nie zwrocil wynikow"
    assert matcher.match_count(["kurczak", "cebula", "czosnek"], ranked_results[0]["_idx"]) > 0, (
        "Najlepiej dopasowany wynik nie zawiera zadanych skladnikow"
    )

    available_pl = {code for code, _ in get_available_targets("pl")}
    assert {"en", "de", "fr", "es"}.issubset(available_pl), "Brak pakietow tlumaczen dla pl"
    assert translate_text("mam ziemniaki i kurczaka", "pl", "en"), "Tlumaczenie pl->en nie dziala"
    assert translate_text("i have potatoes and chicken", "en", "pl"), "Tlumaczenie en->pl nie dziala"

    print("OK: smoke test przeszedl pomyslnie.")


if __name__ == "__main__":
    main()
