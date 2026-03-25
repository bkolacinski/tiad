import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import customtkinter as ctk
from tkinter import filedialog

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

from stt import AudioRecorder, load_model, transcribe
from ingredient_extractor import extract_ingredients
from recipe_matcher import RecipeMatcher
from translator import translate_text, install_language_pair, TranslationUnavailable

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0d0f18"
PANEL   = "#151929"
CARD    = "#1e2235"
CARD_H  = "#252b42"
ACCENT  = "#7c6af7"
REC     = "#e84560"
REC2    = "#b52d46"
SUCCESS = "#26d97f"
WARN    = "#f0a500"
TEXT    = "#eef0f7"
MUTED   = "#6b7280"
PILL_BG = "#252b42"
BORDER  = "#2a2f47"


# ── Detail popup ──────────────────────────────────────────────────────────────
class DetailWindow(ctk.CTkToplevel):
    def __init__(self, parent, recipe: dict, matched: list):
        super().__init__(parent)
        title = recipe.get("title", "Przepis")
        self.title(title)
        self.geometry("780x660")
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grab_set()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Header ──
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=80)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        hdr.grid_propagate(False)

        ctk.CTkLabel(hdr, text=title,
                     font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=TEXT, anchor="w"
                     ).grid(row=0, column=0, padx=24, pady=(16, 2), sticky="w")

        cat = recipe.get("category", "")
        url = recipe.get("url", "")
        meta = "  ·  ".join(x for x in [cat, url] if x)
        ctk.CTkLabel(hdr, text=meta, font=ctk.CTkFont(size=11),
                     text_color=MUTED, anchor="w"
                     ).grid(row=1, column=0, padx=24, pady=(0, 14), sticky="w")

        # ── Body ──
        body = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        ings = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", recipe.get("steps", ""))
        row = 0

        # Matched banner
        if matched:
            banner = ctk.CTkFrame(body, fg_color="#1a3328", corner_radius=10)
            banner.grid(row=row, column=0, sticky="ew", padx=24, pady=(20, 0))
            banner.grid_columnconfigure(0, weight=1)
            row += 1

            ctk.CTkLabel(banner,
                         text=f"✓  Znaleziono {len(matched)} z {len(matched) + (0)} składników",
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=SUCCESS, anchor="w"
                         ).grid(row=0, column=0, padx=14, pady=(10, 6), sticky="w")

            chip_wrap = ctk.CTkFrame(banner, fg_color="transparent")
            chip_wrap.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
            for m in matched:
                pill = ctk.CTkFrame(chip_wrap, fg_color=SUCCESS, corner_radius=10)
                pill.pack(side="left", padx=(0, 6))
                ctk.CTkLabel(pill, text=m, font=ctk.CTkFont(size=11),
                             text_color=BG).pack(padx=9, pady=3)

        # Ingredients
        ctk.CTkLabel(body,
                     text=f"Składniki  ({len(ings) if isinstance(ings, list) else '?'})",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT
                     ).grid(row=row, column=0, padx=24, pady=(20, 8), sticky="w")
        row += 1

        ing_frame = ctk.CTkFrame(body, fg_color=CARD, corner_radius=10)
        ing_frame.grid(row=row, column=0, padx=24, sticky="ew")
        ing_frame.grid_columnconfigure(0, weight=1)
        row += 1

        if isinstance(ings, list):
            for i, ing in enumerate(ings):
                bg = CARD if i % 2 == 0 else CARD_H
                r = ctk.CTkFrame(ing_frame, fg_color=bg, corner_radius=0, height=32)
                r.grid(row=i, column=0, sticky="ew")
                r.grid_columnconfigure(0, weight=1)
                r.grid_propagate(False)

                # Highlight matched ingredients in list
                ing_str = str(ing)
                is_matched = any(
                    m.lower() in ing_str.lower() or ing_str.lower().startswith(m[:4].lower())
                    for m in matched
                )
                color = SUCCESS if is_matched else TEXT
                ctk.CTkLabel(r, text=f"  {'✓' if is_matched else '·'}  {ing_str}",
                             font=ctk.CTkFont(size=12), text_color=color, anchor="w"
                             ).grid(row=0, column=0, sticky="w", padx=8)

        # Instructions
        if instructions:
            ctk.CTkLabel(body, text="Przygotowanie",
                         font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT
                         ).grid(row=row, column=0, padx=24, pady=(24, 8), sticky="w")
            row += 1

            box = ctk.CTkTextbox(body, font=ctk.CTkFont(size=12),
                                 fg_color=CARD, text_color=TEXT,
                                 corner_radius=10, wrap="word", height=200)
            box.grid(row=row, column=0, padx=24, pady=(0, 28), sticky="ew")
            box.insert("1.0", instructions)
            box.configure(state="disabled")


# ── Main App ──────────────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Recipe Voice Filter")
        self.geometry("1280x800")
        self.minsize(960, 640)
        self.configure(fg_color=BG)
        self._center_window()

        self.recorder = AudioRecorder()
        self.is_recording = False
        self._pulse_job = None
        self.matcher = RecipeMatcher(DATA_DIR)
        self.current_ingredients: list = []
        self.current_results: list = []
        self.current_language: str = "pl"
        self.search_mode = ctk.StringVar(value="any")

        self._build_ui()
        self._start_loading()

    def _center_window(self):
        self.update_idletasks()
        w, h = 1280, 800
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self._build_topbar()
        self._build_sidebar()
        self._build_main()

    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=56)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        bar.grid_propagate(False)

        ctk.CTkLabel(bar, text="🍽  Recipe Voice Filter",
                     font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT
                     ).grid(row=0, column=0, padx=24, pady=14, sticky="w")

        self.lbl_status = ctk.CTkLabel(bar, text="",
                                        font=ctk.CTkFont(size=12), text_color=MUTED)
        self.lbl_status.grid(row=0, column=1, padx=16, pady=14, sticky="e")

        ctk.CTkFrame(bar, fg_color=BORDER, width=2, height=30).grid(row=0, column=2)

        mode_frame = ctk.CTkFrame(bar, fg_color="transparent")
        mode_frame.grid(row=0, column=3, padx=(12, 24), pady=14)
        ctk.CTkLabel(mode_frame, text="Tryb:",
                     font=ctk.CTkFont(size=12), text_color=MUTED
                     ).pack(side="left", padx=(0, 8))

        self.seg_mode = ctk.CTkSegmentedButton(
            mode_frame, values=["Ranking", "Wszystkie"],
            command=self._on_mode_change,
            font=ctk.CTkFont(size=12), height=30,
            selected_color=ACCENT, selected_hover_color="#6254d4",
            unselected_color=CARD, unselected_hover_color=CARD_H,
            text_color=TEXT, fg_color=CARD,
        )
        self.seg_mode.set("Ranking")
        self.seg_mode.pack(side="left")

    def _on_mode_change(self, val):
        self.search_mode.set("any" if val == "Ranking" else "all")

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, width=360)
        side.grid(row=1, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(4, weight=1)

        # Mic
        mic_area = ctk.CTkFrame(side, fg_color="transparent")
        mic_area.grid(row=0, column=0, padx=24, pady=(28, 0), sticky="ew")
        mic_area.grid_columnconfigure(0, weight=1)

        self.btn_record = ctk.CTkButton(
            mic_area, text="⏺   Nagraj",
            font=ctk.CTkFont(size=17, weight="bold"),
            height=64, corner_radius=32,
            fg_color=REC, hover_color=REC2,
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=0, column=0, sticky="ew")

        self.btn_file = ctk.CTkButton(
            mic_area, text="📂  Wczytaj plik audio",
            font=ctk.CTkFont(size=13), height=36, corner_radius=18,
            fg_color=CARD, hover_color=CARD_H,
            border_width=1, border_color=BORDER, text_color=TEXT,
            command=self._load_file,
        )
        self.btn_file.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        ctk.CTkFrame(side, fg_color=BORDER, height=1).grid(
            row=1, column=0, sticky="ew", padx=24, pady=20)

        # Transcription
        tb = ctk.CTkFrame(side, fg_color="transparent")
        tb.grid(row=2, column=0, padx=24, sticky="ew")
        tb.grid_columnconfigure(0, weight=1)

        hrow = ctk.CTkFrame(tb, fg_color="transparent")
        hrow.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hrow, text="Transkrypcja",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
                     ).pack(side="left")
        self.lbl_language = ctk.CTkLabel(hrow, text="",
                                          font=ctk.CTkFont(size=11), text_color=MUTED)
        self.lbl_language.pack(side="right")

        self.txt_transcription = ctk.CTkTextbox(
            tb, height=80, font=ctk.CTkFont(size=12),
            fg_color=CARD, text_color=TEXT,
            corner_radius=10, border_width=1, border_color=BORDER,
        )
        self.txt_transcription.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Ingredients
        ib = ctk.CTkFrame(side, fg_color="transparent")
        ib.grid(row=3, column=0, padx=24, pady=(16, 0), sticky="ew")
        ib.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(ib, text="Wykryte składniki",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
                     ).grid(row=0, column=0, sticky="w")

        self.ingredients_frame = ctk.CTkScrollableFrame(
            ib, height=68, fg_color=CARD, corner_radius=10,
            border_width=1, border_color=BORDER,
        )
        self.ingredients_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # Translation panel
        tr = ctk.CTkFrame(side, fg_color=CARD, corner_radius=12)
        tr.grid(row=5, column=0, padx=24, pady=(0, 24), sticky="ew")
        tr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(tr, text="Tłumaczenie transkrypcji",
                     font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
                     ).grid(row=0, column=0, columnspan=3, padx=14, pady=(12, 8), sticky="w")

        self.combo_translate = ctk.CTkComboBox(
            tr, values=["angielski", "niemiecki", "francuski", "hiszpański"],
            font=ctk.CTkFont(size=12), width=140,
            fg_color=PANEL, border_color=BORDER, button_color=ACCENT,
            dropdown_fg_color=PANEL,
        )
        self.combo_translate.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")

        self.btn_translate = ctk.CTkButton(
            tr, text="Tłumacz", font=ctk.CTkFont(size=12),
            height=32, width=80, corner_radius=8,
            fg_color=ACCENT, hover_color="#6254d4",
            command=self._translate,
        )
        self.btn_translate.grid(row=1, column=1, padx=(4, 4), pady=(0, 8))

        self.btn_copy = ctk.CTkButton(
            tr, text="Kopiuj", font=ctk.CTkFont(size=11),
            height=32, width=60, corner_radius=8,
            fg_color=CARD_H, hover_color=BORDER, text_color=MUTED,
            command=self._copy_translation,
        )
        self.btn_copy.grid(row=1, column=2, padx=(0, 14), pady=(0, 8))

        self.txt_translation = ctk.CTkTextbox(
            tr, height=64, font=ctk.CTkFont(size=11),
            fg_color=PANEL, text_color=TEXT,
            corner_radius=8, wrap="word",
        )
        self.txt_translation.grid(row=2, column=0, columnspan=3,
                                   padx=14, pady=(0, 8), sticky="ew")
        self.txt_translation.insert("1.0", "Transkrypcja pojawi się tutaj po tłumaczeniu...")
        self.txt_translation.configure(state="disabled", text_color=MUTED)

        self.btn_install_pkg = ctk.CTkButton(
            tr, text="⬇  Pobierz pakiety tłumaczeń",
            font=ctk.CTkFont(size=11), height=28, corner_radius=8,
            fg_color="#2a1f00", hover_color="#3a2c00", text_color=WARN,
            border_width=1, border_color=WARN,
            command=self._install_translation_packages,
        )
        self.btn_install_pkg.grid(row=3, column=0, columnspan=3,
                                   padx=14, pady=(0, 12), sticky="ew")
        self.btn_install_pkg.grid_remove()

    def _build_main(self):
        self._main_frame = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self._main_frame.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        self._main_frame.grid_rowconfigure(1, weight=1)
        self._main_frame.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 0))
        hdr.grid_columnconfigure(0, weight=1)

        self.lbl_results_count = ctk.CTkLabel(
            hdr, text="Powiedz składniki, aby wyszukać przepisy",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT, anchor="w"
        )
        self.lbl_results_count.grid(row=0, column=0, sticky="w")

        self.results_scroll = ctk.CTkScrollableFrame(
            self._main_frame, fg_color="transparent", corner_radius=0
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(12, 20))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.results_scroll,
            text="🎤  Nagraj głos lub wczytaj plik audio\naby wyszukać przepisy według składników",
            font=ctk.CTkFont(size=15), text_color=MUTED,
        ).grid(row=0, column=0, pady=120)

        # Loading overlay on top of main frame
        self._overlay = ctk.CTkFrame(self._main_frame, fg_color=BG, corner_radius=0)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._overlay.grid_columnconfigure(0, weight=1)
        self._overlay.grid_rowconfigure(0, weight=1)

        center = ctk.CTkFrame(self._overlay, fg_color="transparent")
        center.place(relx=0.5, rely=0.42, anchor="center")

        ctk.CTkLabel(center, text="🍽",
                     font=ctk.CTkFont(size=52)).pack()
        ctk.CTkLabel(center, text="Recipe Voice Filter",
                     font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT
                     ).pack(pady=(10, 4))

        self.lbl_loading_step = ctk.CTkLabel(
            center, text="Inicjalizacja...",
            font=ctk.CTkFont(size=13), text_color=MUTED
        )
        self.lbl_loading_step.pack(pady=(0, 20))

        self.progress_bar = ctk.CTkProgressBar(
            center, width=300, height=6,
            mode="indeterminate",
            progress_color=ACCENT,
            fg_color=CARD,
        )
        self.progress_bar.pack()
        self.progress_bar.start()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _start_loading(self):
        def load():
            self._set_loading("Ładowanie bazy przepisów...")
            count = self.matcher.load(
                progress_callback=lambda m: self._set_loading(m)
            )
            if count == 0:
                self._set_status("⚠ Brak przepisów", WARN)
                self._hide_overlay()
                return

            self._set_loading(f"Ładowanie modelu mowy Whisper...")
            try:
                load_model(MODELS_DIR,
                           progress_callback=lambda m: self._set_loading(m))
                self._set_status(f"✓ Gotowy  ·  {count} przepisów", SUCCESS)
            except Exception as e:
                self._set_status(f"⚠ Błąd modelu: {e}", REC)

            self._hide_overlay()

        threading.Thread(target=load, daemon=True).start()

    def _set_loading(self, msg: str):
        self.after(0, lambda: self.lbl_loading_step.configure(text=msg))

    def _hide_overlay(self):
        def hide():
            self.progress_bar.stop()
            self._overlay.place_forget()
        self.after(0, hide)

    # ── Recording ─────────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.is_recording = True
        self.btn_record.configure(text="⏹   Zatrzymaj nagranie")
        self._pulse()
        self._set_status("● Nagrywanie...", REC)
        self.recorder.start()

    def _stop_recording(self):
        self.is_recording = False
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job = None
        self.btn_record.configure(text="⏺   Nagraj", fg_color=REC, hover_color=REC2)
        self._set_status("Przetwarzanie mowy...", WARN)
        threading.Thread(
            target=lambda: self._process_audio(self.recorder.stop()),
            daemon=True
        ).start()

    def _pulse(self):
        if not self.is_recording:
            return
        cur = self.btn_record.cget("fg_color")
        self.btn_record.configure(fg_color=REC2 if cur == REC else REC)
        self._pulse_job = self.after(550, self._pulse)

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a"), ("Wszystkie", "*.*")]
        )
        if not path:
            return
        self._set_status("Wczytywanie pliku...", WARN)
        threading.Thread(target=self._process_file, args=(path,), daemon=True).start()

    def _process_file(self, path):
        import soundfile as sf
        import numpy as np
        from scipy.signal import resample as sp_resample
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = sp_resample(audio, int(len(audio) * 16000 / sr))
        self._process_audio(audio)

    def _process_audio(self, audio):
        try:
            result = transcribe(audio, MODELS_DIR)
            text = result["text"]
            lang = result["language"]
            lang_name = result["language_name"]
            self.current_language = lang
            self.after(0, lambda: self._update_transcription(text, lang_name))

            # Extract + deduplicate ingredients
            raw = extract_ingredients(text, self.matcher.vocabulary)
            ingredients = self._deduplicate_ingredients(raw)
            self.current_ingredients = ingredients
            self.after(0, lambda: self._update_ingredients(ingredients))

            # Search
            results = self.matcher.search(ingredients, mode=self.search_mode.get())

            # Sort by matched_count DESC, TF-IDF as tiebreaker
            def match_score(r):
                idx = r.get("_idx", -1)
                lem = self.matcher.lemmatized_text(idx)
                return sum(1 for i in ingredients
                           if self.matcher._ingredient_in_text(i, lem))

            results.sort(key=lambda r: (match_score(r), r.get("_score", 0)), reverse=True)
            self.current_results = results
            self.after(0, lambda: self._update_results(results))
            self._set_status(f"✓ Znaleziono {len(results)} przepisów", SUCCESS)

        except Exception as e:
            self._set_status(f"Błąd: {e}", REC)

    def _deduplicate_ingredients(self, ingredients: list) -> list:
        """Remove near-duplicate ingredients (e.g. 'kurczak' and 'kurczaka')."""
        from rapidfuzz import fuzz
        result = []
        for ing in ingredients:
            if not any(fuzz.ratio(ing, existing) >= 82 for existing in result):
                result.append(ing)
        return result

    # ── UI updates ────────────────────────────────────────────────────────────

    def _update_transcription(self, text, lang_name):
        self.txt_transcription.configure(state="normal")
        self.txt_transcription.delete("1.0", "end")
        self.txt_transcription.insert("1.0", text)
        self.lbl_language.configure(text=f"  {lang_name}")

    def _update_ingredients(self, ingredients):
        for w in self.ingredients_frame.winfo_children():
            w.destroy()
        if not ingredients:
            ctk.CTkLabel(self.ingredients_frame, text="Brak wykrytych składników",
                         font=ctk.CTkFont(size=11), text_color=MUTED
                         ).pack(anchor="w", padx=8)
            return
        wrap = ctk.CTkFrame(self.ingredients_frame, fg_color="transparent")
        wrap.pack(fill="x", padx=4, pady=4)
        for ing in ingredients:
            pill = ctk.CTkFrame(wrap, fg_color=ACCENT, corner_radius=12)
            pill.pack(side="left", padx=(0, 6), pady=2)
            ctk.CTkLabel(pill, text=ing, font=ctk.CTkFont(size=11),
                         text_color="white").pack(padx=10, pady=3)

    def _update_results(self, results):
        for w in self.results_scroll.winfo_children():
            w.destroy()

        if not results:
            self.lbl_results_count.configure(text="Nie znaleziono przepisów")
            ctk.CTkLabel(self.results_scroll,
                         text="🔍  Brak przepisów dla podanych składników",
                         font=ctk.CTkFont(size=14), text_color=MUTED
                         ).grid(row=0, column=0, pady=80)
            return

        total_asked = len(self.current_ingredients)
        self.lbl_results_count.configure(
            text=f"Znaleziono {len(results)} przepisów"
        )
        for display_idx, recipe in enumerate(results):
            recipe_idx = recipe.get("_idx", -1)
            lem_text = self.matcher.lemmatized_text(recipe_idx)
            matched_count = sum(
                1 for i in self.current_ingredients
                if self.matcher._ingredient_in_text(i, lem_text)
            )
            self._add_card(display_idx, recipe, matched_count, total_asked)

    def _add_card(self, display_idx: int, recipe: dict, matched_count: int, total_asked: int):
        ings = recipe.get("ingredients", [])
        ing_count = len(ings) if isinstance(ings, list) else 0

        card = ctk.CTkFrame(self.results_scroll, fg_color=CARD,
                             corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=display_idx, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
        inner.grid_columnconfigure(0, weight=1)

        # Title + score
        title_row = ctk.CTkFrame(inner, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(title_row,
                     text=recipe.get("title", "Bez nazwy"),
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TEXT, anchor="w"
                     ).grid(row=0, column=0, sticky="w")

        if total_asked > 0:
            score_text = f"{matched_count}/{total_asked}"
            score_color = (SUCCESS if matched_count == total_asked
                           else WARN if matched_count > 0 else MUTED)
        else:
            score_text, score_color = "", MUTED

        ctk.CTkLabel(title_row, text=score_text,
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=score_color
                     ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        # Meta
        cat = recipe.get("category", "")
        meta = f"{cat}  ·  {ing_count} składników" if cat else f"{ing_count} składników"
        ctk.CTkLabel(inner, text=meta, font=ctk.CTkFont(size=11),
                     text_color=MUTED, anchor="w"
                     ).grid(row=1, column=0, sticky="w", pady=(2, 6))

        # Ingredient chips preview
        if isinstance(ings, list) and ings:
            chip_row = ctk.CTkFrame(inner, fg_color="transparent")
            chip_row.grid(row=2, column=0, sticky="w")
            for ing_text in ings[:6]:
                chip = ctk.CTkFrame(chip_row, fg_color=PILL_BG, corner_radius=8)
                chip.pack(side="left", padx=(0, 4), pady=2)
                ctk.CTkLabel(chip, text=str(ing_text)[:26],
                             font=ctk.CTkFont(size=10), text_color=MUTED
                             ).pack(padx=7, pady=2)
            if ing_count > 6:
                ctk.CTkLabel(chip_row, text=f"+{ing_count-6}",
                             font=ctk.CTkFont(size=10), text_color=MUTED
                             ).pack(side="left", padx=4)

        # Hover + click bindings
        def on_enter(e, c=card): c.configure(fg_color=CARD_H, border_color=ACCENT)
        def on_leave(e, c=card): c.configure(fg_color=CARD, border_color=BORDER)
        def on_click(e, r=recipe): self._show_detail(r)

        all_widgets = [card, inner, title_row]
        for w in inner.winfo_children():
            all_widgets.append(w)
            all_widgets += list(w.winfo_children())
        for w in title_row.winfo_children():
            all_widgets.append(w)

        for widget in all_widgets:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

    def _show_detail(self, recipe):
        recipe_idx = recipe.get("_idx", -1)
        lem_text = self.matcher.lemmatized_text(recipe_idx)
        matched = [i for i in self.current_ingredients
                   if self.matcher._ingredient_in_text(i, lem_text)]
        DetailWindow(self, recipe, matched).focus()

    # ── Translation ───────────────────────────────────────────────────────────

    def _translate(self):
        text = self.txt_transcription.get("1.0", "end").strip()
        if not text or text == "":
            return
        lang_map = {"angielski": "en", "niemiecki": "de",
                    "francuski": "fr", "hiszpański": "es"}
        target = lang_map.get(self.combo_translate.get(), "en")

        self._set_translation("Tłumaczę...", MUTED)
        self.btn_install_pkg.grid_remove()

        def do():
            try:
                result = translate_text(text, self.current_language, target)
                self.after(0, lambda: self._set_translation(result, TEXT))
            except TranslationUnavailable:
                self.after(0, lambda: self._set_translation(
                    "Pakiety tłumaczeń nie są zainstalowane.", WARN))
                self.after(0, self.btn_install_pkg.grid)
            except Exception as e:
                self.after(0, lambda: self._set_translation(f"Błąd: {e}", REC))

        threading.Thread(target=do, daemon=True).start()

    def _set_translation(self, text: str, color: str):
        self.txt_translation.configure(state="normal", text_color=color)
        self.txt_translation.delete("1.0", "end")
        self.txt_translation.insert("1.0", text)
        self.txt_translation.configure(state="disabled")

    def _copy_translation(self):
        text = self.txt_translation.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.btn_copy.configure(text="✓ Skopiowano")
            self.after(1500, lambda: self.btn_copy.configure(text="Kopiuj"))

    def _install_translation_packages(self):
        self.btn_install_pkg.configure(
            text="Pobieranie... (wymaga internetu)", state="disabled")
        self._set_translation("Pobieranie pakietów tłumaczeń...", WARN)

        def do():
            pairs = [("pl", "en"), ("en", "pl"), ("pl", "de"), ("de", "pl")]
            ok = 0
            for src, tgt in pairs:
                if install_language_pair(
                    src, tgt,
                    progress_callback=lambda m: self.after(
                        0, lambda msg=m: self._set_translation(msg, WARN))
                ):
                    ok += 1
            if ok > 0:
                self.after(0, lambda: self._set_translation(
                    f"✓ Zainstalowano {ok} pakietów. Spróbuj tłumaczyć ponownie.", SUCCESS))
                self.after(0, self.btn_install_pkg.grid_remove)
            else:
                self.after(0, lambda: self._set_translation(
                    "⚠ Nie udało się pobrać. Sprawdź internet.", REC))
            self.after(0, lambda: self.btn_install_pkg.configure(
                text="⬇  Pobierz pakiety tłumaczeń", state="normal"))

        threading.Thread(target=do, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = TEXT):
        self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=color))


if __name__ == "__main__":
    App().mainloop()
