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
REC2    = "#b52d46"   # pulse alt
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
        self.title(recipe.get("title", "Przepis"))
        self.geometry("760x640")
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=72)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        hdr.grid_propagate(False)

        title = recipe.get("title", "Przepis")
        ctk.CTkLabel(
            hdr, text=title,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT, anchor="w"
        ).grid(row=0, column=0, padx=24, pady=(14, 2), sticky="w")

        cat = recipe.get("category", "")
        url = recipe.get("url", "")
        meta = f"{cat}  ·  {url}" if cat and url else (cat or url)
        ctk.CTkLabel(
            hdr, text=meta,
            font=ctk.CTkFont(size=11),
            text_color=MUTED, anchor="w"
        ).grid(row=1, column=0, padx=24, pady=(0, 10), sticky="w")

        # Body
        body = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        body.grid_columnconfigure(0, weight=1)

        ings = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", recipe.get("steps", ""))

        # Matched chips
        if matched:
            chip_row = ctk.CTkFrame(body, fg_color="transparent")
            chip_row.grid(row=0, column=0, sticky="w", padx=24, pady=(20, 0))
            ctk.CTkLabel(
                chip_row, text="Znalezione składniki: ",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=SUCCESS
            ).pack(side="left")
            for m in matched:
                pill = ctk.CTkFrame(chip_row, fg_color=SUCCESS, corner_radius=10)
                pill.pack(side="left", padx=(4, 0))
                ctk.CTkLabel(pill, text=m, font=ctk.CTkFont(size=11),
                              text_color=BG).pack(padx=8, pady=2)

        # Ingredients section
        ctk.CTkLabel(
            body,
            text=f"Składniki  ({len(ings) if isinstance(ings, list) else ''})",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT
        ).grid(row=1, column=0, padx=24, pady=(20, 8), sticky="w")

        ings_frame = ctk.CTkFrame(body, fg_color=CARD, corner_radius=10)
        ings_frame.grid(row=2, column=0, padx=24, sticky="ew")
        ings_frame.grid_columnconfigure(0, weight=1)

        if isinstance(ings, list):
            for i, ing in enumerate(ings):
                bg = CARD if i % 2 == 0 else CARD_H
                row_frame = ctk.CTkFrame(ings_frame, fg_color=bg, corner_radius=0, height=30)
                row_frame.grid(row=i, column=0, sticky="ew", padx=0)
                row_frame.grid_columnconfigure(0, weight=1)
                row_frame.grid_propagate(False)
                ctk.CTkLabel(
                    row_frame, text=f"  ·  {ing}",
                    font=ctk.CTkFont(size=12), text_color=TEXT, anchor="w"
                ).grid(row=0, column=0, sticky="w", padx=8)

        # Instructions section
        if instructions:
            ctk.CTkLabel(
                body, text="Przygotowanie",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=TEXT
            ).grid(row=3, column=0, padx=24, pady=(24, 8), sticky="w")

            inst_box = ctk.CTkTextbox(
                body, font=ctk.CTkFont(size=12),
                fg_color=CARD, text_color=TEXT,
                corner_radius=10, wrap="word",
                height=220,
            )
            inst_box.grid(row=4, column=0, padx=24, pady=(0, 24), sticky="ew")
            inst_box.insert("1.0", instructions)
            inst_box.configure(state="disabled")


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
        self._selected_card = None

        self._build_ui()
        self._start_loading()

    def _center_window(self):
        self.update_idletasks()
        w, h = 1280, 800
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
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

        ctk.CTkLabel(
            bar, text="🍽  Recipe Voice Filter",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT
        ).grid(row=0, column=0, padx=24, pady=14, sticky="w")

        self.lbl_status = ctk.CTkLabel(
            bar, text="Ładowanie...",
            font=ctk.CTkFont(size=12),
            text_color=WARN,
        )
        self.lbl_status.grid(row=0, column=1, padx=16, pady=14, sticky="e")

        ctk.CTkFrame(bar, fg_color=BORDER, width=2, height=30).grid(
            row=0, column=2, padx=(0, 0)
        )

        # Mode toggle in topbar
        mode_frame = ctk.CTkFrame(bar, fg_color="transparent")
        mode_frame.grid(row=0, column=3, padx=(12, 24), pady=14)

        ctk.CTkLabel(
            mode_frame, text="Tryb:",
            font=ctk.CTkFont(size=12), text_color=MUTED
        ).pack(side="left", padx=(0, 8))

        self.seg_mode = ctk.CTkSegmentedButton(
            mode_frame,
            values=["Ranking", "Wszystkie"],
            command=self._on_mode_change,
            font=ctk.CTkFont(size=12),
            height=30,
            selected_color=ACCENT,
            selected_hover_color="#6254d4",
            unselected_color=CARD,
            unselected_hover_color=CARD_H,
            text_color=TEXT,
            fg_color=CARD,
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

        # ── Mic area ──
        mic_area = ctk.CTkFrame(side, fg_color="transparent")
        mic_area.grid(row=0, column=0, padx=24, pady=(28, 0), sticky="ew")
        mic_area.grid_columnconfigure(0, weight=1)

        self.btn_record = ctk.CTkButton(
            mic_area,
            text="⏺   Nagraj",
            font=ctk.CTkFont(size=17, weight="bold"),
            height=64,
            corner_radius=32,
            fg_color=REC,
            hover_color=REC2,
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=0, column=0, sticky="ew")

        self.btn_file = ctk.CTkButton(
            mic_area,
            text="📂  Wczytaj plik audio",
            font=ctk.CTkFont(size=13),
            height=36,
            corner_radius=18,
            fg_color=CARD,
            hover_color=CARD_H,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            command=self._load_file,
        )
        self.btn_file.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        # ── Divider ──
        ctk.CTkFrame(side, fg_color=BORDER, height=1).grid(
            row=1, column=0, sticky="ew", padx=24, pady=20
        )

        # ── Transcription ──
        trans_block = ctk.CTkFrame(side, fg_color="transparent")
        trans_block.grid(row=2, column=0, padx=24, sticky="ew")
        trans_block.grid_columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(trans_block, fg_color="transparent")
        header_row.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            header_row, text="Transkrypcja",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
        ).pack(side="left")
        self.lbl_language = ctk.CTkLabel(
            header_row, text="",
            font=ctk.CTkFont(size=11), text_color=MUTED
        )
        self.lbl_language.pack(side="right")

        self.txt_transcription = ctk.CTkTextbox(
            trans_block,
            height=80,
            font=ctk.CTkFont(size=12),
            fg_color=CARD,
            text_color=TEXT,
            corner_radius=10,
            border_width=1,
            border_color=BORDER,
        )
        self.txt_transcription.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # ── Ingredients ──
        ing_block = ctk.CTkFrame(side, fg_color="transparent")
        ing_block.grid(row=3, column=0, padx=24, pady=(16, 0), sticky="ew")
        ing_block.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            ing_block, text="Wykryte składniki",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
        ).grid(row=0, column=0, sticky="w")

        self.ingredients_frame = ctk.CTkScrollableFrame(
            ing_block, height=68,
            fg_color=CARD, corner_radius=10,
            border_width=1, border_color=BORDER,
        )
        self.ingredients_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # ── Translation (bottom) ──
        tr_block = ctk.CTkFrame(side, fg_color=CARD, corner_radius=12)
        tr_block.grid(row=5, column=0, padx=24, pady=(0, 24), sticky="ew")
        tr_block.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tr_block, text="Tłumaczenie",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 6), sticky="w")

        self.combo_translate = ctk.CTkComboBox(
            tr_block,
            values=["angielski", "niemiecki", "francuski", "hiszpański"],
            font=ctk.CTkFont(size=12),
            fg_color=PANEL, border_color=BORDER, button_color=ACCENT,
            dropdown_fg_color=PANEL,
        )
        self.combo_translate.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="ew")

        self.btn_translate = ctk.CTkButton(
            tr_block, text="Tłumacz",
            font=ctk.CTkFont(size=12), height=32,
            corner_radius=8,
            fg_color=ACCENT, hover_color="#6254d4",
            command=self._translate,
        )
        self.btn_translate.grid(row=1, column=1, padx=(4, 14), pady=(0, 10))

        self.lbl_translation = ctk.CTkLabel(
            tr_block, text="",
            font=ctk.CTkFont(size=11), text_color=MUTED,
            wraplength=290, justify="left"
        )
        self.lbl_translation.grid(row=2, column=0, columnspan=2,
                                   padx=14, pady=(0, 4), sticky="w")

        self.btn_install_pkg = ctk.CTkButton(
            tr_block, text="⬇ Pobierz pakiety tłumaczeń",
            font=ctk.CTkFont(size=11), height=28,
            corner_radius=8,
            fg_color=CARD_H, hover_color=CARD,
            text_color=MUTED,
            command=self._install_translation_packages,
        )
        self.btn_install_pkg.grid(row=3, column=0, columnspan=2,
                                   padx=14, pady=(0, 12), sticky="ew")
        self.btn_install_pkg.grid_remove()  # hidden by default

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        main.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # Results header bar
        hdr = ctk.CTkFrame(main, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 0))
        hdr.grid_columnconfigure(0, weight=1)

        self.lbl_results_count = ctk.CTkLabel(
            hdr, text="Powiedz składniki, aby wyszukać przepisy",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT, anchor="w"
        )
        self.lbl_results_count.grid(row=0, column=0, sticky="w")

        # Scrollable recipe list
        self.results_scroll = ctk.CTkScrollableFrame(
            main, fg_color="transparent", corner_radius=0
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew",
                                  padx=20, pady=(12, 20))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        # Placeholder
        self.lbl_placeholder = ctk.CTkLabel(
            self.results_scroll,
            text="🎤  Nagraj głos lub wczytaj plik audio\naby wyszukać przepisy według składników",
            font=ctk.CTkFont(size=15),
            text_color=MUTED,
        )
        self.lbl_placeholder.grid(row=0, column=0, pady=120)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _start_loading(self):
        def load():
            self._set_status("Ładowanie przepisów...", WARN)
            count = self.matcher.load(
                progress_callback=lambda m: self._set_status(m, WARN)
            )
            if count == 0:
                self._set_status("⚠ Brak przepisów — uruchom scraper.py", WARN)
            else:
                self._set_status(f"Ładowanie modelu mowy...", WARN)
                try:
                    load_model(MODELS_DIR,
                               progress_callback=lambda m: self._set_status(m, WARN))
                    self._set_status(f"✓ Gotowy  ·  {count} przepisów", SUCCESS)
                except Exception as e:
                    self._set_status(f"⚠ Błąd modelu: {e}", REC)
        threading.Thread(target=load, daemon=True).start()

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
        threading.Thread(target=lambda: self._process_audio(self.recorder.stop()),
                         daemon=True).start()

    def _pulse(self):
        if not self.is_recording:
            return
        cur = self.btn_record.cget("fg_color")
        nxt = REC2 if cur == REC else REC
        self.btn_record.configure(fg_color=nxt)
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
            text, lang, lang_name = result["text"], result["language"], result["language_name"]
            self.current_language = lang
            self.after(0, lambda: self._update_transcription(text, lang_name))

            ingredients = extract_ingredients(text, self.matcher.vocabulary)
            self.current_ingredients = ingredients
            self.after(0, lambda: self._update_ingredients(ingredients))

            results = self.matcher.search(ingredients, mode=self.search_mode.get())
            self.current_results = results
            self.after(0, lambda: self._update_results(results))
            self._set_status(f"✓ Znaleziono {len(results)} przepisów", SUCCESS)
        except Exception as e:
            self._set_status(f"Błąd: {e}", REC)

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
                         font=ctk.CTkFont(size=11), text_color=MUTED).pack(anchor="w", padx=8)
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
        self._selected_card = None

        if not results:
            self.lbl_results_count.configure(text="Nie znaleziono przepisów")
            ctk.CTkLabel(
                self.results_scroll,
                text="🔍  Brak przepisów dla podanych składników",
                font=ctk.CTkFont(size=14), text_color=MUTED
            ).grid(row=0, column=0, pady=80)
            return

        self.lbl_results_count.configure(
            text=f"Znaleziono {len(results)} przepisów"
        )
        for i, recipe in enumerate(results):
            self._add_card(i, recipe)

    def _add_card(self, idx, recipe):
        score = recipe.get("_score", 0)
        score_pct = int(score * 100)
        ings = recipe.get("ingredients", [])
        ing_count = len(ings) if isinstance(ings, list) else 0

        card = ctk.CTkFrame(self.results_scroll, fg_color=CARD,
                             corner_radius=12, border_width=1, border_color=BORDER)
        card.grid(row=idx, column=0, sticky="ew", pady=(0, 8))
        card.grid_columnconfigure(0, weight=1)

        # Inner layout
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
        inner.grid_columnconfigure(0, weight=1)

        # Title row
        title_row = ctk.CTkFrame(inner, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row,
            text=recipe.get("title", "Bez nazwy"),
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT, anchor="w"
        ).grid(row=0, column=0, sticky="w")

        score_color = SUCCESS if score_pct > 55 else (WARN if score_pct > 25 else MUTED)
        ctk.CTkLabel(
            title_row,
            text=f"{score_pct}%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=score_color
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

        # Meta: category + ingredient count
        cat = recipe.get("category", "")
        meta = f"{cat}  ·  {ing_count} składników" if cat else f"{ing_count} składników"
        ctk.CTkLabel(
            inner, text=meta,
            font=ctk.CTkFont(size=11), text_color=MUTED, anchor="w"
        ).grid(row=1, column=0, sticky="w", pady=(2, 6))

        # Ingredient preview chips
        if isinstance(ings, list) and ings:
            chip_row = ctk.CTkFrame(inner, fg_color="transparent")
            chip_row.grid(row=2, column=0, sticky="w")
            for ing_text in ings[:6]:
                short = str(ing_text)[:28]
                chip = ctk.CTkFrame(chip_row, fg_color=PILL_BG, corner_radius=8)
                chip.pack(side="left", padx=(0, 4), pady=2)
                ctk.CTkLabel(chip, text=short, font=ctk.CTkFont(size=10),
                              text_color=MUTED).pack(padx=7, pady=2)
            if ing_count > 6:
                ctk.CTkLabel(chip_row, text=f"+{ing_count-6}",
                             font=ctk.CTkFont(size=10), text_color=MUTED).pack(side="left", padx=4)

        # Hover + click
        def on_enter(e, c=card): c.configure(fg_color=CARD_H, border_color=ACCENT)
        def on_leave(e, c=card): c.configure(fg_color=CARD, border_color=BORDER)
        def on_click(e, r=recipe): self._show_detail(r)

        for widget in [card, inner, title_row] + list(inner.winfo_children()) + list(title_row.winfo_children()):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

    def _show_detail(self, recipe):
        idx = recipe.get("_idx", -1)
        lem_text = self.matcher.lemmatized_text(idx)
        matched = [i for i in self.current_ingredients
                   if self.matcher._ingredient_in_text(i, lem_text)]
        w = DetailWindow(self, recipe, matched)
        w.focus()

    # ── Translation ───────────────────────────────────────────────────────────

    def _translate(self):
        text = self.txt_transcription.get("1.0", "end").strip()
        if not text:
            return
        lang_map = {"angielski": "en", "niemiecki": "de", "francuski": "fr", "hiszpański": "es"}
        target = lang_map.get(self.combo_translate.get(), "en")
        self.lbl_translation.configure(text="Tłumaczę...", text_color=WARN)
        self.btn_install_pkg.grid_remove()

        def do():
            try:
                result = translate_text(text, self.current_language, target)
                self.after(0, lambda: self.lbl_translation.configure(
                    text=result, text_color=TEXT))
            except TranslationUnavailable:
                self.after(0, lambda: self.lbl_translation.configure(
                    text="⚠ Pakiety tłumaczeń nie są zainstalowane.",
                    text_color=WARN))
                self.after(0, self.btn_install_pkg.grid)
            except Exception as e:
                self.after(0, lambda: self.lbl_translation.configure(
                    text=f"Błąd: {e}", text_color=REC))

        threading.Thread(target=do, daemon=True).start()

    def _install_translation_packages(self):
        self.btn_install_pkg.configure(text="Pobieranie... (wymaga internetu)", state="disabled")
        self.lbl_translation.configure(text="Pobieranie pakietów tłumaczeń...", text_color=WARN)

        def do():
            pairs = [("pl", "en"), ("en", "pl"), ("pl", "de"), ("de", "pl")]
            ok = 0
            for src, tgt in pairs:
                success = install_language_pair(
                    src, tgt,
                    progress_callback=lambda m: self.after(0, lambda msg=m:
                        self.lbl_translation.configure(text=msg, text_color=WARN))
                )
                if success:
                    ok += 1
            if ok > 0:
                self.after(0, lambda: self.lbl_translation.configure(
                    text=f"✓ Zainstalowano {ok} pakietów. Spróbuj tłumaczyć ponownie.",
                    text_color=SUCCESS))
                self.after(0, self.btn_install_pkg.grid_remove)
            else:
                self.after(0, lambda: self.lbl_translation.configure(
                    text="⚠ Nie udało się pobrać pakietów. Sprawdź połączenie z internetem.",
                    text_color=REC))
            self.after(0, lambda: self.btn_install_pkg.configure(
                text="⬇ Pobierz pakiety tłumaczeń", state="normal"))

        threading.Thread(target=do, daemon=True).start()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg, color=TEXT):
        self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=color))


if __name__ == "__main__":
    App().mainloop()
