#!/usr/bin/env python3
"""
Scraper for aniagotuje.pl recipes.
Uses only Python stdlib (urllib, html.parser, json, re).
"""

import urllib.request
import urllib.error
import re
import json
import os
import time
import html
from html.parser import HTMLParser

OUTPUT_DIR = "/home/nikodem/uni/tiad/zadanie2/data"
BATCH_SIZE = 50
SAVE_EVERY = 50
DELAY = 0.5  # seconds between requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'pl-PL,pl;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


def fetch(url, retries=3, timeout=20):
    """Fetch URL with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=timeout)
            return resp.read().decode('utf-8', errors='replace')
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  HTTP error {e.code} for {url}")
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Error fetching {url}: {e}")
                return None


def clean_html(text):
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_all_recipe_urls():
    """Get all recipe URLs from the sitemap."""
    print("Fetching sitemap...")
    content = fetch("https://aniagotuje.pl/sitemap.xml")
    if not content:
        raise Exception("Could not fetch sitemap")

    urls = re.findall(r'<loc>(https://aniagotuje\.pl/przepis/[^<]+)</loc>', content)
    print(f"Found {len(urls)} recipe URLs in sitemap")
    return urls


def parse_recipe(url, content):
    """Parse a recipe page and return recipe dict."""
    recipe = {
        "title": "",
        "url": url,
        "category": "",
        "ingredients": [],
        "instructions": ""
    }

    # Title from <h1>
    title_m = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.DOTALL)
    if title_m:
        recipe["title"] = clean_html(title_m.group(1))

    # Category from breadcrumb: href="/przepisy/CATEGORY"
    cat_m = re.search(r'href="/przepisy/([^"]+)"[^>]*><span[^>]*>([^<]+)</span>', content)
    if cat_m:
        recipe["category"] = cat_m.group(2).strip()
    else:
        # Try alternative - extract category from URL path pattern
        # Sometimes breadcrumb has different structure
        cat_m2 = re.search(r'href="/przepisy/([^"]+)"', content)
        if cat_m2:
            recipe["category"] = cat_m2.group(1).replace('-', ' ')

    # Ingredients: itemprop="recipeIngredient"
    ingredients = re.findall(
        r'itemprop="recipeIngredient"[^>]*>\s*<span class="ingredient">(.*?)</span>',
        content, re.DOTALL
    )
    if not ingredients:
        # Fallback: class="ingredient"
        ingredients = re.findall(r'class="ingredient">(.*?)</span>', content)

    recipe["ingredients"] = [clean_html(i) for i in ingredients if clean_html(i)]

    # Instructions: extract paragraphs from article-content-body,
    # keeping only those that contain actual cooking steps.
    # aniagotuje.pl uses flowing prose (not numbered steps), so we
    # filter by Polish cooking imperative verbs and skip blog/SEO filler.

    COOKING_VERBS = re.compile(
        r'\b(dodaj|smaż|gotuj|wymieszaj|pokrój|zagotuj|wlej|wsyp|odcedź|'
        r'podgrzej|upiecz|nałóż|posyp|przykryj|wyjmij|odstaw|odczekaj|'
        r'rozgrzej|obtocz|marynuj|blenduj|zetrzyj|wyciśnij|obierz|umyj|'
        r'osusz|dopraw|przypraw|ugotuj|podsmaż|podduś|zacznij|przełóż|'
        r'wyłóż|ułóż|zanurz|odlej|wyłącz|zmniejsz|zwiększ|sprawdź|'
        r'siekaj|posiekaj|zblenduj|utłucz|roztop|rozbij|ubij|ugniataj|'
        r'wyrób|podziel|pokrusz|zetrzyj|zamarynuj|zalewaj|namocz|'
        r'nakładaj|porcjuj|kroić|smażyć|gotować|piec|dusić)\b',
        re.IGNORECASE
    )

    SKIP_PATTERNS = [
        r'^Czas przygotowania',
        r'^Czas gotowania',
        r'^Czas pieczenia',
        r'^Czas smażenia',
        r'^Czas duszenia',
        r'^Liczba porcji',
        r'^W 100 g',
        r'^Wartość energetyczna',
        r'^Węglowodany',
        r'^Białko',
        r'^Tłuszcze',
        r'^Dieta:',
        r'Polecam też',
        r'Polecam wypróbuj',
        r'Sprawdź też',
        r'Zapraszam też',
        r'zapraszam po przepis',
        r'przepisy znajdziesz',
        r'^Smacznego',
    ]

    FILLER_PHRASES = re.compile(
        r'(Uwielbiam szyko|Jak podkreślałam|satysfakcj[ęą] z tworzenia|'
        r'w restauracji kosztowałoby|znakomitym sposobem na|kuchnia to przestrzeń|'
        r'Niech ten przepis będzie|Pamiętaj, że kuchnia|Gotowanie w grupie|'
        r'cieszenie się świeżymi|sztucznych dodatków)',
        re.IGNORECASE
    )

    instructions_parts = []

    m = re.search(r'class="article-content-body"[^>]*>(.*?)(?=class="col-12 related-posts"|class="seo-box")', content, re.DOTALL)
    if m:
        body = m.group(1)
        paras = re.findall(r'<p[^>]*>(.*?)</p>', body, re.DOTALL)

        for p in paras:
            text = clean_html(p)
            if len(text) < 60:
                continue
            if any(re.search(pat, text, re.IGNORECASE) for pat in SKIP_PATTERNS):
                continue
            if FILLER_PHRASES.search(text):
                continue
            # Keep if it has cooking verbs OR is a long paragraph (>200 chars) describing preparation
            if COOKING_VERBS.search(text) or (len(text) > 200 and re.search(r'\b(sos|mięso|warzywa|składnik|masa|ciasto|farsz|nadzienie)\b', text, re.IGNORECASE)):
                instructions_parts.append(text)

    if instructions_parts:
        recipe["instructions"] = "\n\n".join(instructions_parts)

    return recipe


def save_batch(recipes, batch_num):
    """Save a batch of recipes to a JSON file."""
    filename = os.path.join(OUTPUT_DIR, f"recipes_{batch_num:03d}.json")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(recipes)} recipes to {filename}")
    return filename


def load_progress():
    """Load list of already scraped URLs."""
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_progress(scraped_urls):
    """Save progress."""
    progress_file = os.path.join(OUTPUT_DIR, "progress.json")
    with open(progress_file, 'w') as f:
        json.dump(list(scraped_urls), f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get all recipe URLs
    all_urls = get_all_recipe_urls()

    # Load progress
    scraped_urls = load_progress()
    print(f"Already scraped: {len(scraped_urls)} recipes")

    # Filter out already scraped
    pending_urls = [u for u in all_urls if u not in scraped_urls]
    print(f"Pending: {len(pending_urls)} recipes to scrape")

    # Determine next batch number
    existing_batches = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('recipes_') and f.endswith('.json')]
    batch_num = len(existing_batches) + 1

    current_batch = []
    total_scraped = len(scraped_urls)
    failed = 0

    for i, url in enumerate(pending_urls):
        print(f"[{i+1}/{len(pending_urls)}] Scraping: {url}")

        content = fetch(url)
        if content is None:
            print(f"  FAILED to fetch {url}")
            failed += 1
            scraped_urls.add(url)  # Mark as processed to avoid retry loops
            continue

        recipe = parse_recipe(url, content)

        if recipe["title"]:
            current_batch.append(recipe)
            scraped_urls.add(url)
            total_scraped += 1
            print(f"  OK: {recipe['title']} [{recipe['category']}] - {len(recipe['ingredients'])} ingredients")
        else:
            print(f"  WARNING: Could not parse title for {url}")
            failed += 1
            scraped_urls.add(url)

        # Save batch when full
        if len(current_batch) >= BATCH_SIZE:
            save_batch(current_batch, batch_num)
            save_progress(scraped_urls)
            batch_num += 1
            current_batch = []

        # Rate limiting
        time.sleep(DELAY)

    # Save final partial batch
    if current_batch:
        save_batch(current_batch, batch_num)
        save_progress(scraped_urls)

    print(f"\n{'='*50}")
    print(f"DONE! Total scraped: {total_scraped}, Failed: {failed}")

    # Count total recipes in all files
    total_in_files = 0
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith('recipes_') and f.endswith('.json'):
            with open(os.path.join(OUTPUT_DIR, f)) as fp:
                data = json.load(fp)
                total_in_files += len(data)
    print(f"Total recipes in files: {total_in_files}")


if __name__ == "__main__":
    main()
