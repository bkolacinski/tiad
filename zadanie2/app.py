import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Recipe Voice Filter - Main Application
Zadanie 2: Filtrowanie przepisów na podstawie mowy

UI: customtkinter
STT: OpenAI Whisper (offline)
Matching: TF-IDF + cosine similarity (no LLM)
"""

import os
import sys
import threading
import webbrowser

import customtkinter as ctk
from PIL import Image, ImageTk

# Resolve paths when running as PyInstaller bundle
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    DATA_DIR = os.path.join(os.path.dirname(sys.executable), "data")
    MODELS_DIR = os.path.join(os.path.dirname(sys.executable), "models")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

from stt import AudioRecorder, load_model, transcribe, transcribe_file
from ingredient_extractor import extract_ingredients
from recipe_matcher import RecipeMatcher
from translator import get_available_targets, translate_text


# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLOR_BG = "#1a1a2e"
COLOR_PANEL = "#16213e"
COLOR_ACCENT = "#0f3460"
COLOR_HIGHLIGHT = "#e94560"
COLOR_TEXT = "#eaeaea"
COLOR_MUTED = "#888"
COLOR_GREEN = "#27ae60"
COLOR_ORANGE = "#f39c12"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Recipe Voice Filter")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.configure(fg_color=COLOR_BG)

        # State
        self.recorder = AudioRecorder()
        self.is_recording = False
        self.matcher = RecipeMatcher(DATA_DIR)
        self.current_ingredients: list = []
        self.current_results: list = []
        self.current_language: str = "pl"
        self.search_mode = ctk.StringVar(value="any")

        self._build_ui()
        self._start_loading()

    # ─── UI Construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=5)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        left = ctk.CTkFrame(self, fg_color=COLOR_PANEL, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        left.grid_rowconfigure(6, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Header
        ctk.CTkLabel(
            left, text="🎤 Recipe Voice Filter",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLOR_HIGHLIGHT
        ).grid(row=0, column=0, padx=20, pady=(20, 4), sticky="w")

        ctk.CTkLabel(
            left, text="Powiedz składniki, a znajdziemy przepisy",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_MUTED
        ).grid(row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        # Record button
        self.btn_record = ctk.CTkButton(
            left,
            text="⏺  Nagraj",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=60,
            fg_color=COLOR_HIGHLIGHT,
            hover_color="#c0392b",
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=2, column=0, padx=20, pady=8, sticky="ew")

        # Upload file button
        self.btn_file = ctk.CTkButton(
            left,
            text="📂  Wczytaj plik audio",
            font=ctk.CTkFont(size=13),
            height=38,
            fg_color=COLOR_ACCENT,
            hover_color="#1a5276",
            command=self._load_file,
        )
        self.btn_file.grid(row=3, column=0, padx=20, pady=4, sticky="ew")

        # Status label
        self.lbl_status = ctk.CTkLabel(
            left, text="Ładowanie...",
            font=ctk.CTkFont(size=12),
            text_color=COLOR_ORANGE,
            wraplength=280,
        )
        self.lbl_status.grid(row=4, column=0, padx=20, pady=8, sticky="w")

        # Transcription box
        ctk.CTkLabel(
            left, text="Transkrypcja:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT
        ).grid(row=5, column=0, padx=20, pady=(12, 2), sticky="w")

        self.txt_transcription = ctk.CTkTextbox(
            left, height=90, font=ctk.CTkFont(size=12),
            fg_color=COLOR_ACCENT, text_color=COLOR_TEXT
        )
        self.txt_transcription.grid(row=6, column=0, padx=20, pady=(0, 8), sticky="nsew")

        # Detected language
        self.lbl_language = ctk.CTkLabel(
            left, text="Język: —",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_MUTED
        )
        self.lbl_language.grid(row=7, column=0, padx=20, pady=(0, 4), sticky="w")

        # Ingredients panel
        ctk.CTkLabel(
            left, text="Wykryte składniki:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_TEXT
        ).grid(row=8, column=0, padx=20, pady=(8, 2), sticky="w")

        self.ingredients_frame = ctk.CTkScrollableFrame(
            left, height=100, fg_color=COLOR_ACCENT
        )
        self.ingredients_frame.grid(row=9, column=0, padx=20, pady=(0, 8), sticky="ew")

        # Search mode
        mode_frame = ctk.CTkFrame(left, fg_color="transparent")
        mode_frame.grid(row=10, column=0, padx=20, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(
            mode_frame, text="Tryb wyszukiwania:",
            font=ctk.CTkFont(size=11), text_color=COLOR_MUTED
        ).pack(side="left")

        ctk.CTkRadioButton(
            mode_frame, text="Wszystkie", variable=self.search_mode, value="all",
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=(8, 4))
        ctk.CTkRadioButton(
            mode_frame, text="Ranking", variable=self.search_mode, value="any",
            font=ctk.CTkFont(size=11)
        ).pack(side="left")

        # Translation section
        trans_frame = ctk.CTkFrame(left, fg_color=COLOR_ACCENT, corner_radius=8)
        trans_frame.grid(row=11, column=0, padx=20, pady=(0, 16), sticky="ew")
        trans_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            trans_frame, text="Tłumaczenie transkrypcji:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_TEXT
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        self.combo_translate = ctk.CTkComboBox(
            trans_frame,
            values=["angielski", "niemiecki", "francuski", "hiszpański"],
            font=ctk.CTkFont(size=11),
            width=160,
        )
        self.combo_translate.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")

        ctk.CTkButton(
            trans_frame, text="Tłumacz",
            font=ctk.CTkFont(size=11), height=28,
            fg_color=COLOR_HIGHLIGHT, hover_color="#c0392b",
            command=self._translate,
        ).grid(row=1, column=1, padx=(4, 10), pady=(0, 8))

        self.lbl_translation = ctk.CTkLabel(
            trans_frame, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_MUTED,
            wraplength=260,
        )
        self.lbl_translation.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="w")

    def _build_right_panel(self):
        right = ctk.CTkFrame(self, fg_color=COLOR_BG, corner_radius=0)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Results header
        header = ctk.CTkFrame(right, fg_color=COLOR_PANEL, corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)

        self.lbl_results_count = ctk.CTkLabel(
            header, text="Przepisy — powiedz składniki, aby wyszukać",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.lbl_results_count.grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Results list
        self.results_scroll = ctk.CTkScrollableFrame(
            right, fg_color=COLOR_PANEL, corner_radius=8
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        # Placeholder
        self.lbl_placeholder = ctk.CTkLabel(
            self.results_scroll,
            text="🎤 Nagraj głos lub wczytaj plik audio\naby wyszukać przepisy według składników",
            font=ctk.CTkFont(size=14),
            text_color=COLOR_MUTED,
        )
        self.lbl_placeholder.grid(row=0, column=0, pady=80)

        # Detail panel
        self.detail_frame = ctk.CTkFrame(right, fg_color=COLOR_PANEL, corner_radius=8, height=200)
        self.detail_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.detail_frame.grid_columnconfigure(0, weight=1)
        self.detail_frame.grid_propagate(False)

        self.detail_title = ctk.CTkLabel(
            self.detail_frame, text="Kliknij przepis, aby zobaczyć szczegóły",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT
        )
        self.detail_title.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")

        self.detail_scroll = ctk.CTkScrollableFrame(
            self.detail_frame, fg_color=COLOR_ACCENT, corner_radius=6
        )
        self.detail_scroll.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.detail_scroll.grid_columnconfigure(0, weight=1)
        self.detail_frame.grid_rowconfigure(1, weight=1)

        self.detail_body = ctk.CTkLabel(
            self.detail_scroll, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_TEXT,
            wraplength=550,
            justify="left",
            anchor="nw"
        )
        self.detail_body.grid(row=0, column=0, padx=10, pady=8, sticky="nw")

    # ─── Loading ───────────────────────────────────────────────────────────────

    def _start_loading(self):
        def load():
            # Load recipe database
            self._set_status("Ładowanie bazy przepisów...", COLOR_ORANGE)
            count = self.matcher.load(
                progress_callback=lambda msg: self._set_status(msg, COLOR_ORANGE)
            )
            if count == 0:
                self._set_status(
                    "⚠ Brak przepisów w bazie. Uruchom scraper.py",
                    COLOR_ORANGE
                )
            else:
                self._set_status(f"✓ Załadowano {count} przepisów", COLOR_GREEN)

            # Pre-load Whisper model
            self._set_status("Ładowanie modelu mowy (Whisper)...", COLOR_ORANGE)
            try:
                load_model(MODELS_DIR, progress_callback=lambda m: self._set_status(m, COLOR_ORANGE))
                self._set_status(f"✓ Gotowy · {count} przepisów", COLOR_GREEN)
            except Exception as e:
                self._set_status(f"⚠ Błąd modelu: {e}", COLOR_HIGHLIGHT)

        threading.Thread(target=load, daemon=True).start()

    # ─── Recording ────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.is_recording = True
        self.btn_record.configure(
            text="⏹  Zatrzymaj",
            fg_color=COLOR_GREEN,
            hover_color="#1e8449"
        )
        self._set_status("● Nagrywanie...", COLOR_HIGHLIGHT)
        self.recorder.start()

    def _stop_recording(self):
        self.is_recording = False
        self.btn_record.configure(
            text="⏺  Nagraj",
            fg_color=COLOR_HIGHLIGHT,
            hover_color="#c0392b"
        )
        self._set_status("Przetwarzanie mowy...", COLOR_ORANGE)

        def process():
            audio = self.recorder.stop()
            self._process_audio(audio)

        threading.Thread(target=process, daemon=True).start()

    def _load_file(self):
        from tkinter import filedialog
        filepath = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[
                ("Pliki audio", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                ("Wszystkie pliki", "*.*"),
            ]
        )
        if not filepath:
            return
        self._set_status("Przetwarzanie pliku...", COLOR_ORANGE)

        def process():
            import soundfile as sf
            import numpy as np
            from scipy.signal import resample as scipy_resample

            audio, sr = sf.read(filepath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                n = int(len(audio) * 16000 / sr)
                audio = scipy_resample(audio, n)
            self._process_audio(audio)

        threading.Thread(target=process, daemon=True).start()

    def _process_audio(self, audio):
        try:
            result = transcribe(audio, MODELS_DIR)
            text = result["text"]
            lang = result["language"]
            lang_name = result["language_name"]

            self.current_language = lang

            # Update UI - transcription
            self.after(0, lambda: self._update_transcription(text, lang_name))

            # Extract ingredients
            ingredients = extract_ingredients(text, self.matcher.vocabulary)
            self.current_ingredients = ingredients
            self.after(0, lambda: self._update_ingredients(ingredients))

            # Search recipes
            mode = self.search_mode.get()
            results = self.matcher.search(ingredients, mode=mode)
            self.current_results = results
            self.after(0, lambda: self._update_results(results))

            self._set_status(f"✓ Znaleziono {len(results)} przepisów", COLOR_GREEN)

        except Exception as e:
            self._set_status(f"Błąd: {e}", COLOR_HIGHLIGHT)

    # ─── UI Updates ───────────────────────────────────────────────────────────

    def _update_transcription(self, text: str, lang_name: str):
        self.txt_transcription.delete("1.0", "end")
        self.txt_transcription.insert("1.0", text)
        self.lbl_language.configure(text=f"Język: {lang_name}")

    def _update_ingredients(self, ingredients: list):
        for w in self.ingredients_frame.winfo_children():
            w.destroy()

        if not ingredients:
            ctk.CTkLabel(
                self.ingredients_frame, text="Nie wykryto składników",
                font=ctk.CTkFont(size=11), text_color=COLOR_MUTED
            ).pack(anchor="w")
            return

        wrap = ctk.CTkFrame(self.ingredients_frame, fg_color="transparent")
        wrap.pack(fill="x")

        for ing in ingredients:
            tag = ctk.CTkFrame(wrap, fg_color=COLOR_HIGHLIGHT, corner_radius=12)
            tag.pack(side="left", padx=3, pady=3)
            ctk.CTkLabel(
                tag, text=ing,
                font=ctk.CTkFont(size=11),
                text_color="white"
            ).pack(padx=8, pady=3)

    def _update_results(self, results: list):
        for w in self.results_scroll.winfo_children():
            w.destroy()

        if not results:
            self.lbl_results_count.configure(text="Nie znaleziono przepisów")
            ctk.CTkLabel(
                self.results_scroll,
                text="🔍 Brak przepisów dla podanych składników\nSpróbuj trybu 'Ranking' lub dodaj więcej składników",
                font=ctk.CTkFont(size=13),
                text_color=COLOR_MUTED,
            ).grid(row=0, column=0, pady=60)
            return

        self.lbl_results_count.configure(
            text=f"Znaleziono {len(results)} przepisów"
        )

        for i, recipe in enumerate(results):
            self._add_recipe_card(i, recipe)

    def _add_recipe_card(self, idx: int, recipe: dict):
        score = recipe.get("_score", 0)
        score_pct = int(score * 100)

        card = ctk.CTkFrame(
            self.results_scroll,
            fg_color=COLOR_ACCENT,
            corner_radius=8
        )
        card.grid(row=idx, column=0, sticky="ew", padx=4, pady=4)
        card.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(card, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row,
            text=recipe.get("title", "Bez nazwy"),
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
            anchor="w"
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_row,
            text=f"{score_pct}%",
            font=ctk.CTkFont(size=11),
            text_color=COLOR_GREEN if score_pct > 60 else COLOR_ORANGE,
            anchor="e"
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        # Ingredients preview
        ings = recipe.get("ingredients", [])
        if isinstance(ings, list):
            preview = ", ".join(str(i) for i in ings[:5])
            if len(ings) > 5:
                preview += f" (+{len(ings) - 5})"
        else:
            preview = str(ings)[:80]

        ctk.CTkLabel(
            card,
            text=preview,
            font=ctk.CTkFont(size=10),
            text_color=COLOR_MUTED,
            anchor="w",
            wraplength=450,
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        card.bind("<Button-1>", lambda e, r=recipe: self._show_detail(r))
        for child in card.winfo_children():
            child.bind("<Button-1>", lambda e, r=recipe: self._show_detail(r))
            for grandchild in child.winfo_children():
                grandchild.bind("<Button-1>", lambda e, r=recipe: self._show_detail(r))

    def _show_detail(self, recipe: dict):
        self.detail_title.configure(text=recipe.get("title", "Przepis"))

        ings = recipe.get("ingredients", [])
        if isinstance(ings, list):
            ings_text = "\n".join(f"  • {i}" for i in ings)
        else:
            ings_text = str(ings)

        instructions = recipe.get("instructions", recipe.get("steps", "Brak opisu."))

        # Highlight matched ingredients
        matched = [i for i in self.current_ingredients
                   if i.lower() in str(recipe.get("ingredients", "")).lower()]

        text = f"SKŁADNIKI ({len(ings) if isinstance(ings, list) else '?'}):\n{ings_text}\n\n"
        if matched:
            text += f"DOPASOWANE: {', '.join(matched)}\n\n"
        text += f"PRZYGOTOWANIE:\n{instructions}"

        url = recipe.get("url", "")
        if url:
            text += f"\n\nŹRÓDŁO: {url}"

        self.detail_body.configure(text=text)

    # ─── Translation ──────────────────────────────────────────────────────────

    def _translate(self):
        text = self.txt_transcription.get("1.0", "end").strip()
        if not text:
            return

        target_name = self.combo_translate.get()
        name_to_code = {
            "angielski": "en", "niemiecki": "de",
            "francuski": "fr", "hiszpański": "es",
        }
        target = name_to_code.get(target_name, "en")

        self.lbl_translation.configure(text="Tłumaczenie...", text_color=COLOR_ORANGE)

        def do_translate():
            translated = translate_text(text, self.current_language, target)
            if translated:
                self.after(0, lambda: self.lbl_translation.configure(
                    text=translated, text_color=COLOR_TEXT
                ))
            else:
                self.after(0, lambda: self.lbl_translation.configure(
                    text="⚠ Pakiet tłumaczeń nie jest zainstalowany.\nUruchom: python setup_translation.py",
                    text_color=COLOR_ORANGE
                ))

        threading.Thread(target=do_translate, daemon=True).start()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = COLOR_TEXT):
        self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=color))


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
