"""Pantry — Recipe Voice Filter GUI (offline: audio → STT → ingredients → recipes)."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import time
import unicodedata

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog


def _asset_ready(asset_name: str, path: str) -> bool:
    """Return whether an asset directory (data/models/translations) looks complete."""
    if not os.path.isdir(path):
        return False
    if asset_name == "data":
        return os.path.exists(os.path.join(path, "recipe_index_cache.pkl"))
    if asset_name == "models":
        return os.path.exists(os.path.join(path, "small.pt"))
    if asset_name == "translation_packages":
        try:
            for name in os.listdir(path):
                package_dir = os.path.join(path, name)
                if os.path.isdir(package_dir) and os.path.exists(
                    os.path.join(package_dir, "metadata.json")
                ):
                    return True
            return False
        except OSError:
            return False
    return os.path.exists(path)


def _runtime_asset_dir(asset_name: str) -> str:
    """Resolve path to data/models/translation_packages (next to EXE or under _MEIPASS when frozen)."""
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        bundled_dir = getattr(sys, "_MEIPASS", exe_dir)
        external_path = os.path.join(exe_dir, asset_name)
        bundled_path = os.path.join(bundled_dir, asset_name)
        if _asset_ready(asset_name, external_path):
            return external_path
        if _asset_ready(asset_name, bundled_path):
            return bundled_path
        return bundled_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(base_dir, asset_name)
    if _asset_ready(asset_name, source_path):
        return source_path
    bundled_path = os.path.join(base_dir, "release", "RecipeVoiceFilter", "_internal", asset_name)
    if _asset_ready(asset_name, bundled_path):
        return bundled_path
    return source_path


if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
    DATA_DIR = _runtime_asset_dir("data")
    MODELS_DIR = _runtime_asset_dir("models")
    TRANSLATIONS_DIR = _runtime_asset_dir("translation_packages")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = _runtime_asset_dir("data")
    MODELS_DIR = _runtime_asset_dir("models")
    TRANSLATIONS_DIR = _runtime_asset_dir("translation_packages")

for path in [DATA_DIR, MODELS_DIR, TRANSLATIONS_DIR]:
    if not getattr(sys, "frozen", False) or path.startswith(os.path.dirname(sys.executable)):
        os.makedirs(path, exist_ok=True)
os.environ.setdefault("ARGOS_PACKAGES_DIR", TRANSLATIONS_DIR)

from stt import (
    AudioRecorder,
    LANG_NAMES,
    RECORDING_SAMPLE_RATE,
    detect_language_code,
    load_model,
    transcribe_whisper,
)
from stt_hf_wav2vec import hf_wav2vec_transformers_available, transcribe_hf_wav2vec
from stt_mms import mms_transformers_available, transcribe_mms
from stt_vosk import transcribe_vosk, vosk_available
from ingredient_extractor import extract_ingredients
from recipe_matcher import RecipeMatcher
from text_utils import normalize_text, tokenize_words, tokens_match
from translator import (
    TranslationUnavailable,
    configure_translation_packages,
    get_available_targets,
    translate_text,
)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# ── Palette: sage + warm cream ────────────────────────────────────────────
BG = "#f0eeea"
PANEL = "#fefefe"
CARD = "#ffffff"
CARD_H = "#f7f5f0"
ACCENT = "#3d6b50"      # sage herb green
ACCENT_H = "#2e5440"
ACCENT_SOFT = "#e2eeea"
RED = "#b83030"
RED_H = "#912525"
GREEN = "#1a6640"
GREEN_SOFT = "#e0f2e9"
AMBER = "#9b6820"
AMBER_SOFT = "#fef3d0"
TEXT = "#1a1814"
TEXT2 = "#3a3530"
MUTED = "#7a736a"
DIM = "#b0a89e"
BORDER = "#d8d0c4"
TAG_BG = "#ece7de"
MATCH_FULL = "#1a5c35"
MATCH_PART = "#7a4e10"

TRANSLATION_LABELS = {
    "pl": "polski",
    "en": "angielski",
    "de": "niemiecki",
    "fr": "francuski",
    "es": "hiszpanski",
}
TRANSLATION_LABEL_TO_CODE = {label: code for code, label in TRANSLATION_LABELS.items()}
# Argos may return a label with a Polish character — map it to the same code
TRANSLATION_LABEL_TO_CODE["hiszpański"] = "es"

def _fold_label(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()


LOW_SIGNAL_INGREDIENTS = {"woda", "olej", "ocet", "sól", "pieprz", "cukier"}
GENERIC_INGREDIENTS = {"pierś", "filet", "udko", "noga", "skrzydło", "mięso"}
SPECIFIC_PROTEINS = {
    "kurczak", "indyk", "wołowina", "wieprzowina",
    "ryba", "łosoś", "mintaj", "dorsz",
}


# ── Helpers ───────────────────────────────────────────────────────────────

def _detail_match_reject(query_norm: str, word_norm: str) -> bool:
    """Reject spurious fuzzy matches (e.g. potato vs potato starch/flour)."""
    if not query_norm or not word_norm:
        return False
    if query_norm.startswith("ziemniak") and "ziemniaczan" in word_norm:
        return True
    return False


def _detail_line_matches_query_term(query: str, ingredient_line: str) -> bool:
    """Match words in an ingredient line stricter than tokens_match alone (avoids potato vs starch confusion)."""
    qn = normalize_text(query)
    for word in tokenize_words(ingredient_line):
        if not tokens_match(query, word):
            continue
        if _detail_match_reject(qn, normalize_text(word)):
            continue
        return True
    return False


# ── Animated status label ────────────────────────────────────────────────
class StatusDots:
    """Animates trailing dots on a label to show activity."""

    def __init__(self, label: ctk.CTkLabel, base_text: str, color: str):
        self._label = label
        self._base = base_text
        self._color = color
        self._count = 0
        self._job = None
        self._running = False

    def start(self):
        self._running = True
        self._tick()

    def stop(self):
        self._running = False
        if self._job:
            try:
                self._label.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def update_text(self, base_text: str):
        self._base = base_text

    def _tick(self):
        if not self._running:
            return
        self._count = (self._count + 1) % 4
        dots = "." * self._count
        try:
            self._label.configure(text=f"{self._base}{dots}", text_color=self._color)
        except Exception:
            return
        self._job = self._label.after(400, self._tick)


# ── Detail popup ─────────────────────────────────────────────────────────
def _detail_translate_long(text: str, source: str, target: str) -> str:
    """Argos can fail on very long blocks — split on paragraph breaks."""
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) < 2800:
        return translate_text(t, source, target)
    parts = [p.strip() for p in t.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if len(parts) <= 1:
        parts = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if not parts:
        return translate_text(t, source, target)
    out = []
    for p in parts:
        out.append(translate_text(p, source, target))
    return "\n\n".join(out)


class DetailWindow(ctk.CTkToplevel):
    """Recipe detail window: ingredients, query match, instructions, translation."""

    def __init__(
        self,
        parent,
        recipe: dict,
        matched: list,
        total_asked: int = 0,
        matched_supplement: list | None = None,
    ):
        super().__init__(parent)
        title_str = recipe.get("title", "Przepis")
        self.title(title_str)
        self.geometry("720x700")
        self.minsize(540, 480)
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.transient(parent)
        self._dcanvas = None
        self._detail_win_id = None

        matched_for_lines = list(matched) + list(matched_supplement or [])
        ings = recipe.get("ingredients", [])
        if not isinstance(ings, list):
            ings = []
        instructions = recipe.get("instructions", recipe.get("steps", "")) or ""
        cat = (recipe.get("category") or "").strip()
        ing_count = len(ings)

        self._recipe = recipe
        self._matched_core = list(matched)
        self._matched_for_lines = matched_for_lines
        self._total_asked = total_asked
        self._orig_title = title_str
        self._orig_cat = cat
        self._orig_ings = [str(x) for x in ings]
        self._orig_instructions = instructions if isinstance(instructions, str) else str(instructions)
        self._row_match_flags = [
            any(_detail_line_matches_query_term(m, str(ings[i])) for m in matched_for_lines)
            for i in range(len(ings))
        ]
        self._detail_lang_labels: list[str] = []
        self._detail_lang_to_code: dict[str, str | None] = {}
        self._detail_translating = False

        # ── Canvas-based scroll (no CTkScrollableFrame lag) ──────────────────
        outer = tk.Frame(self, bg=BG)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        self._dcanvas = canvas

        sb = ctk.CTkScrollbar(outer, command=canvas.yview,
                               button_color=ACCENT, button_hover_color=ACCENT_H)
        sb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=sb.set)

        content = tk.Frame(canvas, bg=BG)
        win_id = canvas.create_window((0, 0), window=content, anchor="nw")
        self._detail_win_id = win_id

        def _on_content_resize(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        content.bind("<Configure>", _on_content_resize)

        def _on_canvas_resize(e):
            canvas.itemconfig(win_id, width=e.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.bind_all("<MouseWheel>", self._on_wheel)

        P = 20

        # ── HEADER ─────────────────────────────────────────────────────────
        hdr = tk.Frame(content, bg=PANEL, relief="flat")
        hdr.pack(fill="x", padx=P, pady=(18, 0))

        self._detail_title_lbl = tk.Label(
            hdr, text=title_str, bg=PANEL, fg=TEXT,
            font=("Segoe UI", 20, "bold"),
            anchor="w", justify="left", wraplength=640,
        )
        self._detail_title_lbl.pack(anchor="w", padx=18, pady=(16, 4))

        meta_parts = []
        if cat:
            meta_parts.append(cat)
        meta_parts.append(f"{ing_count} składników")
        self._detail_meta_lbl = tk.Label(
            hdr, text="  ·  ".join(meta_parts), bg=PANEL, fg=MUTED,
            font=("Segoe UI", 12), anchor="w",
        )
        self._detail_meta_lbl.pack(anchor="w", padx=18, pady=(0, 8))

        # ── Recipe display language (Argos offline) ───────────────────────
        tr_wrap = ctk.CTkFrame(hdr, fg_color=PANEL, corner_radius=0)
        tr_wrap.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkLabel(
            tr_wrap, text="Język wyświetlania:",
            font=ctk.CTkFont(size=12), text_color=TEXT2,
        ).pack(side="left", padx=(4, 8))

        self._detail_lang_labels = ["Polski (oryginał)"]
        self._detail_lang_to_code = {"Polski (oryginał)": None}
        try:
            for code, name in get_available_targets("pl"):
                self._detail_lang_labels.append(name)
                self._detail_lang_to_code[name] = code
        except Exception:
            pass

        self._detail_lang_var = ctk.StringVar(value=self._detail_lang_labels[0])
        self._detail_lang_menu = ctk.CTkOptionMenu(
            tr_wrap,
            variable=self._detail_lang_var,
            values=self._detail_lang_labels,
            font=ctk.CTkFont(size=12),
            fg_color=ACCENT,
            button_color=ACCENT_H,
            button_hover_color=ACCENT_H,
            dropdown_fg_color=CARD,
            dropdown_hover_color=CARD_H,
            text_color="white",
            width=200,
            command=self._on_detail_language_selected,
        )
        self._detail_lang_menu.pack(side="left", padx=(0, 10))

        self._detail_tr_status = ctk.CTkLabel(
            tr_wrap, text="", font=ctk.CTkFont(size=11), text_color=MUTED,
        )
        self._detail_tr_status.pack(side="left", fill="x", expand=True)

        self._recipe_body = tk.Frame(content, bg=BG)
        self._recipe_body.pack(fill="both", expand=True)

        self._render_recipe_body(
            title_str=title_str,
            cat=cat,
            ing_count=ing_count,
            ings_display=self._orig_ings,
            instructions_display=self._orig_instructions,
            matched_for_lines=matched_for_lines,
            matched_core=list(matched),
            total_asked=total_asked,
            ui_lang="pl",
        )

        self.bind("<Destroy>", self._on_destroy, add="+")
        self.after(60, self._finalize_window)

    def _on_detail_language_selected(self, choice: str):
        if self._detail_translating:
            return
        code = self._detail_lang_to_code.get(choice)
        if code is None:
            self._apply_detail_original()
            return
        self._detail_translating = True
        self._detail_lang_menu.configure(state="disabled")
        self._detail_tr_status.configure(text="Tłumaczenie…", text_color=ACCENT)
        threading.Thread(target=self._detail_translate_worker, args=(code,), daemon=True).start()

    def _apply_detail_original(self):
        self._detail_tr_status.configure(text="", text_color=MUTED)
        self.title(self._orig_title)
        self._detail_title_lbl.configure(text=self._orig_title)
        meta_parts = []
        if self._orig_cat:
            meta_parts.append(self._orig_cat)
        meta_parts.append(f"{len(self._orig_ings)} składników")
        self._detail_meta_lbl.configure(text="  ·  ".join(meta_parts))
        for w in self._recipe_body.winfo_children():
            w.destroy()
        self._render_recipe_body(
            title_str=self._orig_title,
            cat=self._orig_cat,
            ing_count=len(self._orig_ings),
            ings_display=self._orig_ings,
            instructions_display=self._orig_instructions,
            matched_for_lines=self._matched_for_lines,
            matched_core=self._matched_core,
            total_asked=self._total_asked,
            ui_lang="pl",
        )
        self._refresh_detail_scroll()

    def _detail_translate_worker(self, target_code: str):
        err: str | None = None
        bundle: dict | None = None
        try:
            tgt = target_code.lower()
            title_t = translate_text(self._orig_title, "pl", tgt)
            cat_t = translate_text(self._orig_cat, "pl", tgt) if self._orig_cat else ""
            ings_t = [translate_text(line, "pl", tgt) for line in self._orig_ings]
            inst_t = _detail_translate_long(self._orig_instructions, "pl", tgt)
            meta_count_t = translate_text(
                f"{len(self._orig_ings)} składników", "pl", tgt
            )
            sk_h = translate_text("Składniki", "pl", tgt)
            pr_h = translate_text("Przygotowanie", "pl", tgt)
            matched = self._matched_core
            total_asked = self._total_asked
            matched_lines = self._matched_for_lines
            if matched_lines and total_asked > 0:
                sym = "✓" if len(matched) == total_asked else "~"
                banner_line = translate_text(
                    f"{sym}  {len(matched)} z {total_asked} składników dopasowanych",
                    "pl",
                    tgt,
                )
                pills_t = [translate_text(str(x), "pl", tgt) for x in matched_lines]
            else:
                banner_line = ""
                pills_t = []
            bundle = {
                "title": title_t,
                "cat": cat_t,
                "meta_count": meta_count_t,
                "ings": ings_t,
                "instructions": inst_t,
                "sk_header": f"{sk_h} ({len(self._orig_ings)})",
                "prep_header": pr_h,
                "banner_line": banner_line,
                "pills": pills_t,
                "is_full": len(matched) == total_asked if total_asked > 0 else True,
                "target_code": tgt,
            }
        except TranslationUnavailable as exc:
            err = str(exc)
        except Exception as exc:
            err = str(exc)

        def done():
            self._detail_translating = False
            self._detail_lang_menu.configure(state="normal")
            if err:
                self._detail_tr_status.configure(text=err[:120], text_color=RED)
                return
            if bundle:
                self._apply_detail_translated(bundle)

        self.after(0, done)

    def _apply_detail_translated(self, b: dict):
        self._detail_tr_status.configure(text="", text_color=MUTED)
        title_t = b["title"]
        self.title(title_t)
        self._detail_title_lbl.configure(text=title_t)
        meta_parts = []
        if (b.get("cat") or "").strip():
            meta_parts.append(b["cat"].strip())
        meta_parts.append(b["meta_count"])
        self._detail_meta_lbl.configure(text="  ·  ".join(meta_parts))
        for w in self._recipe_body.winfo_children():
            w.destroy()
        self._render_recipe_body(
            title_str=title_t,
            cat=b.get("cat") or "",
            ing_count=len(self._orig_ings),
            ings_display=b["ings"],
            instructions_display=b["instructions"],
            matched_for_lines=self._matched_for_lines,
            matched_core=self._matched_core,
            total_asked=self._total_asked,
            ui_lang=b.get("target_code", "pl"),
            banner_line_t=b.get("banner_line") or "",
            pills_t=b.get("pills") or [],
            is_full_banner=b.get("is_full", True),
            ing_header_override=b.get("sk_header"),
            prep_header_override=b.get("prep_header"),
        )
        self._refresh_detail_scroll()

    def _refresh_detail_scroll(self):
        try:
            self.update_idletasks()
            if self._dcanvas and self._detail_win_id:
                self._dcanvas.configure(scrollregion=self._dcanvas.bbox("all"))
        except Exception:
            pass

    def _render_recipe_body(
        self,
        title_str: str,
        cat: str,
        ing_count: int,
        ings_display: list[str],
        instructions_display: str,
        matched_for_lines: list,
        matched_core: list,
        total_asked: int,
        ui_lang: str,
        banner_line_t: str | None = None,
        pills_t: list | None = None,
        is_full_banner: bool = True,
        ing_header_override: str | None = None,
        prep_header_override: str | None = None,
    ):
        """Render recipe body into ``self._recipe_body`` (original or translated)."""
        _ = (title_str, cat)
        parent = self._recipe_body
        P = 20
        matched_for_lines = list(matched_for_lines)
        pills_t = pills_t or []
        banner_line_t = banner_line_t if banner_line_t is not None else None

        # ── MATCH BANNER ────────────────────────────────────────────────────
        if matched_for_lines and total_asked > 0:
            if ui_lang == "pl":
                is_full = len(matched_core) == total_asked
                b_bg = "#e8f5ee" if is_full else "#fdf5e0"
                b_fg = MATCH_FULL if is_full else MATCH_PART
                sym = "✓" if is_full else "~"
                banner_txt = f"{sym}  {len(matched_core)} z {total_asked} składników dopasowanych"
                pill_items = [str(m) for m in matched_for_lines]
            else:
                is_full = is_full_banner
                b_bg = "#e8f5ee" if is_full else "#fdf5e0"
                b_fg = MATCH_FULL if is_full else MATCH_PART
                banner_txt = banner_line_t or ""
                pill_items = [str(x) for x in pills_t]

            banner = tk.Frame(parent, bg=b_bg, relief="flat")
            banner.pack(fill="x", padx=P, pady=(10, 0))

            tk.Label(
                banner, text=banner_txt,
                bg=b_bg, fg=b_fg,
                font=("Segoe UI", 13, "bold"), anchor="w",
            ).pack(anchor="w", padx=16, pady=(12, 8))

            pills_row = tk.Frame(banner, bg=b_bg)
            pills_row.pack(anchor="w", padx=14, pady=(0, 12))
            for m in pill_items:
                tk.Label(
                    pills_row, text=str(m),
                    bg="#c8eed9" if is_full else "#f5e4b0",
                    fg=b_fg,
                    font=("Segoe UI", 11, "bold"), padx=10, pady=5,
                ).pack(side="left", padx=(0, 6))

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=P, pady=(14, 0))

        ing_heading = (
            ing_header_override
            if ing_header_override
            else f"Składniki ({ing_count})"
        )
        tk.Label(
            parent, text=ing_heading,
            bg=BG, fg=TEXT,
            font=("Segoe UI", 14, "bold"), anchor="w",
        ).pack(anchor="w", padx=P + 4, pady=(16, 8))

        ing_card = tk.Frame(parent, bg=CARD, relief="flat",
                            highlightthickness=1, highlightbackground=BORDER)
        ing_card.pack(fill="x", padx=P)

        for i, ing_str in enumerate(ings_display):
            is_m = self._row_match_flags[i] if i < len(self._row_match_flags) else False
            row_bg = "#edf9f2" if is_m else (CARD if i % 2 == 0 else CARD_H)

            row_f = tk.Frame(ing_card, bg=row_bg)
            row_f.pack(fill="x")

            sym = "✓" if is_m else " "
            sym_fg = MATCH_FULL if is_m else DIM
            tk.Label(row_f, text=sym, bg=row_bg, fg=sym_fg,
                     font=("Segoe UI", 13, "bold"), width=3, anchor="center",
                     ).pack(side="left", padx=(10, 4), pady=10)
            tk.Label(row_f, text=ing_str, bg=row_bg,
                     fg=MATCH_FULL if is_m else TEXT2,
                     font=("Segoe UI", 13), anchor="w",
                     ).pack(side="left", fill="x", expand=True, padx=(0, 16), pady=10)

            if i < len(ings_display) - 1:
                tk.Frame(ing_card, bg=BORDER, height=1).pack(fill="x")

        if instructions_display:
            tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=P, pady=(18, 0))
            prep_h = prep_header_override or "Przygotowanie"
            tk.Label(
                parent, text=prep_h,
                bg=BG, fg=TEXT,
                font=("Segoe UI", 14, "bold"), anchor="w",
            ).pack(anchor="w", padx=P + 4, pady=(16, 8))

            inst_box = ctk.CTkTextbox(
                parent,
                font=ctk.CTkFont(size=13),
                fg_color=CARD,
                text_color=TEXT2,
                corner_radius=10,
                border_width=1,
                border_color=BORDER,
                wrap="word",
                height=min(400, max(120, instructions_display.count("\n") * 22 + 60)),
            )
            inst_box.pack(fill="x", padx=P, pady=(0, 24))
            inst_box.insert("1.0", instructions_display)
            inst_box.configure(state="disabled")
        else:
            tk.Frame(parent, bg=BG, height=24).pack()

    def _on_wheel(self, event):
        if not self._dcanvas:
            return
        try:
            self._dcanvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _on_destroy(self, event):
        if event.widget is not self:
            return
        try:
            self.unbind_all("<MouseWheel>")
        except Exception:
            pass
        try:
            if hasattr(self.master, "_bind_recipe_scroll"):
                self.master.after(80, self.master._bind_recipe_scroll)
        except Exception:
            pass

    def _finalize_window(self):
        try:
            self.update_idletasks()
            if self.winfo_viewable():
                self.grab_set()
        except Exception:
            pass
        try:
            self.lift()
            self.focus_force()
        except Exception:
            pass


# ── Main App ─────────────────────────────────────────────────────────────
class App(ctk.CTk):
    """Main window: record/file audio, STT engine selection, recipe results, translation."""

    def __init__(self):
        super().__init__()
        self.title("Pantry")
        self.geometry("1280x860")
        self.minsize(1040, 700)
        self.configure(fg_color=BG)
        self._center_window()
        configure_translation_packages(TRANSLATIONS_DIR)

        self.recorder = AudioRecorder()
        self.is_recording = False
        self._pulse_job = None
        self._dots_anim = None
        self.matcher = RecipeMatcher(DATA_DIR)
        self.detected_ingredients: list = []
        self.current_ingredients: list = []
        self.current_results: list = []
        self.current_language: str = "pl"
        self._layout_refresh_job = None
        self._layout_mode = None

        self._audio_buffer: np.ndarray | None = None
        self._audio_buffer_hq: np.ndarray | None = None
        self._audio_hq_sr: int = 16000
        self._audio_meta: dict = {}
        self._detected_lang: str | None = None
        self._text_whisper: str = ""
        self._text_vosk: str = ""
        self._text_hf_wv: str = ""
        self._text_mms: str = ""
        self._lang_whisper: str | None = None
        self._lang_vosk: str | None = None
        self._lang_hf_wv: str | None = None
        self._lang_mms: str | None = None
        # stt_engine drives both model selection and recipe search source
        self.stt_engine = ctk.StringVar(value="whisper")

        self._build_ui()
        self._refresh_layout()
        self.bind("<Configure>", self._schedule_layout_refresh)
        self.after(120, self._refresh_layout)
        self._refresh_translation_support()
        self._start_loading()

    def _center_window(self):
        self.update_idletasks()
        w, h = 1280, 860
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    # ── Build UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_header_bar()
        self._build_workspace_shell()

    def _build_header_bar(self):
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, border_width=0, height=50)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        header.grid_rowconfigure(0, weight=1)
        header.grid_propagate(False)

        # Bottom border line
        ctk.CTkFrame(header, fg_color=BORDER, height=1).place(relx=0, rely=1.0, relwidth=1, anchor="sw")

        ctk.CTkLabel(
            header, text="Pantry",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT, anchor="w",
        ).grid(row=0, column=0, padx=18, sticky="w")

        self.status_card = ctk.CTkFrame(
            header, fg_color=ACCENT_SOFT, corner_radius=11,
            border_width=1, border_color=BORDER,
        )
        self.status_card.grid(row=0, column=1, padx=14, sticky="e")

        self.lbl_status = ctk.CTkLabel(
            self.status_card, text="Uruchamianie...",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT2,
        )
        self.lbl_status.pack(padx=11, pady=5)

    def _build_workspace_shell(self):
        self._main_frame = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self._main_frame.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self._main_frame.grid_rowconfigure(0, weight=1)
        self._main_frame.grid_columnconfigure(0, weight=1)

        self.workspace = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        self.workspace.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=0, minsize=242)
        self.workspace.grid_columnconfigure(1, weight=1)

        self.left_col = ctk.CTkFrame(self.workspace, fg_color="transparent", width=248)
        self.right_col = ctk.CTkFrame(self.workspace, fg_color="transparent")

        self.input_card = self._build_input_card(self.left_col)
        self.transcription_card = self._build_transcription_card(self.left_col)

        self.left_col.grid_columnconfigure(0, weight=1)
        self.input_card.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.transcription_card.grid(row=1, column=0, sticky="ew")

        self.right_col.grid_rowconfigure(0, weight=1)
        self.right_col.grid_columnconfigure(0, weight=1)
        self.results_card = self._build_results_card(self.right_col)
        self.results_card.grid(row=0, column=0, sticky="nsew")

        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.right_col.grid(row=0, column=1, sticky="nsew")

        self._overlay = ctk.CTkFrame(self._main_frame, fg_color=BG, corner_radius=0)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        center = ctk.CTkFrame(self._overlay, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)
        center.place(relx=0.5, rely=0.43, anchor="center")

        ctk.CTkLabel(
            center, text="Pantry",
            font=ctk.CTkFont(size=18, weight="bold"), text_color=TEXT,
        ).pack(padx=32, pady=(22, 6))

        self.lbl_loading_step = ctk.CTkLabel(
            center, text="Inicjalizacja",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        )
        self.lbl_loading_step.pack(pady=(0, 14))

        self.progress_bar = ctk.CTkProgressBar(
            center, width=220, height=5, mode="indeterminate",
            progress_color=ACCENT, fg_color=ACCENT_SOFT,
        )
        self.progress_bar.pack(padx=32, pady=(0, 22))
        self.progress_bar.start()

    def _build_input_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            card, text="Audio",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT, anchor="w",
        ).grid(row=0, column=0, columnspan=2, padx=11, pady=(8, 4), sticky="w")

        self.btn_record = ctk.CTkButton(
            card, text="Nagraj",
            font=ctk.CTkFont(size=11, weight="bold"), height=32, corner_radius=12,
            fg_color=ACCENT, hover_color=ACCENT_H, text_color="white",
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=1, column=0, padx=(11, 4), sticky="ew")

        self.btn_file = ctk.CTkButton(
            card, text="Wczytaj plik",
            font=ctk.CTkFont(size=10), height=32, corner_radius=12,
            fg_color=CARD, hover_color=CARD_H, text_color=TEXT2,
            border_width=1, border_color=BORDER, command=self._load_file,
        )
        self.btn_file.grid(row=1, column=1, padx=(4, 11), sticky="ew")

        self.lbl_audio_meta = ctk.CTkLabel(
            card, text="Brak nagrania",
            font=ctk.CTkFont(size=9), text_color=MUTED, anchor="w",
        )
        self.lbl_audio_meta.grid(row=2, column=0, columnspan=2, padx=11, pady=(3, 2), sticky="ew")

        self.btn_save_audio = ctk.CTkButton(
            card, text="Zapisz WAV",
            font=ctk.CTkFont(size=9), height=24, corner_radius=10,
            fg_color=CARD, hover_color=CARD_H, text_color=TEXT2,
            border_width=1, border_color=BORDER,
            command=self._save_audio_buffer, state="disabled",
        )
        self.btn_save_audio.grid(row=3, column=0, padx=(11, 4), pady=(0, 8), sticky="ew")

        self.btn_play_audio = ctk.CTkButton(
            card, text="Odtworz",
            font=ctk.CTkFont(size=9), height=24, corner_radius=10,
            fg_color=CARD, hover_color=CARD_H, text_color=TEXT2,
            border_width=1, border_color=BORDER,
            command=self._play_audio_buffer, state="disabled",
        )
        self.btn_play_audio.grid(row=3, column=1, padx=(4, 11), pady=(0, 8), sticky="ew")
        return card

    def _build_transcription_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        card.grid_columnconfigure(0, weight=1)

        # ── Header ─────────────────────────────────────────────────────────
        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Transkrypcja", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT).grid(row=0, column=0, sticky="w")
        self.lbl_language = ctk.CTkLabel(head, text="", font=ctk.CTkFont(size=9), text_color=MUTED)
        self.lbl_language.grid(row=0, column=1, sticky="e")

        # ── Model selector ─────────────────────────────────────────────────
        model_frame = ctk.CTkFrame(card, fg_color=BG, corner_radius=10)
        model_frame.grid(row=1, column=0, padx=12, pady=(4, 4), sticky="ew")
        model_frame.grid_columnconfigure(0, weight=1)

        self.rb_whisper = ctk.CTkRadioButton(
            model_frame, text="Whisper small", variable=self.stt_engine, value="whisper",
            font=ctk.CTkFont(size=10), text_color=TEXT,
            fg_color=ACCENT, hover_color=ACCENT_H,
            command=self._on_engine_change,
        )
        self.rb_whisper.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")

        self.btn_stt = ctk.CTkButton(
            model_frame, text="Transkrybuj", font=ctk.CTkFont(size=9, weight="bold"),
            height=26, width=90, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_H, text_color="white",
            command=self._transcribe_action, state="disabled",
        )
        self.btn_stt.grid(row=0, column=1, rowspan=4, padx=(4, 8), pady=8, sticky="e")

        self.rb_vosk = ctk.CTkRadioButton(
            model_frame, text="Vosk PL/EN", variable=self.stt_engine, value="vosk",
            font=ctk.CTkFont(size=10), text_color=TEXT,
            fg_color=ACCENT, hover_color=ACCENT_H,
            command=self._on_engine_change,
        )
        self.rb_vosk.grid(row=1, column=0, padx=10, pady=(0, 4), sticky="w")

        self.rb_hf_wv = ctk.CTkRadioButton(
            model_frame,
            text="Wav2Vec2 XLS-R (HF, Meta) PL/EN",
            variable=self.stt_engine,
            value="hf_wav2vec",
            font=ctk.CTkFont(size=10),
            text_color=TEXT,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            command=self._on_engine_change,
        )
        self.rb_hf_wv.grid(row=2, column=0, padx=10, pady=(0, 4), sticky="w")

        self.rb_mms = ctk.CTkRadioButton(
            model_frame,
            text="Meta MMS-1B ASR (HF) PL/EN",
            variable=self.stt_engine,
            value="meta_mms",
            font=ctk.CTkFont(size=10),
            text_color=TEXT,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            command=self._on_engine_change,
        )
        self.rb_mms.grid(row=3, column=0, padx=10, pady=(0, 8), sticky="w")

        self.lbl_vosk_hint = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=8), text_color=AMBER, anchor="w", wraplength=220)
        self.lbl_vosk_hint.grid(row=2, column=0, padx=12, sticky="ew")
        self.lbl_vosk_hint.grid_remove()

        # ── Single transcription text ───────────────────────────────────────
        self.txt_stt = ctk.CTkTextbox(card, height=60, font=ctk.CTkFont(size=10), fg_color=BG, text_color=TEXT2, corner_radius=8, border_width=0, wrap="word")
        self.txt_stt.grid(row=3, column=0, padx=12, pady=(4, 4), sticky="ew")
        self.txt_whisper = self.txt_stt
        self.txt_vosk = self.txt_stt

        # ── Translation section ─────────────────────────────────────────────
        ctk.CTkFrame(card, fg_color=BORDER, height=1).grid(row=4, column=0, padx=12, pady=(6, 8), sticky="ew")

        ctk.CTkLabel(card, text="Tłumaczenie", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT, anchor="w").grid(row=5, column=0, padx=12, pady=(0, 4), sticky="w")

        tr_ctrl = ctk.CTkFrame(card, fg_color="transparent")
        tr_ctrl.grid(row=6, column=0, padx=12, sticky="ew")
        tr_ctrl.grid_columnconfigure(0, weight=1)
        self.combo_translate = ctk.CTkOptionMenu(
            tr_ctrl,
            values=[name for code, name in TRANSLATION_LABELS.items() if code != "pl"],
            font=ctk.CTkFont(size=10),
            fg_color=ACCENT, button_color=ACCENT_H, button_hover_color=ACCENT_H,
            dropdown_fg_color=CARD, dropdown_hover_color=CARD_H,
            text_color="white", dynamic_resizing=False,
        )
        self.combo_translate.grid(row=0, column=0, sticky="ew")
        self.btn_translate = ctk.CTkButton(tr_ctrl, text="Tłumacz", font=ctk.CTkFont(size=9, weight="bold"), height=28, width=64, corner_radius=8, fg_color=TEXT, hover_color=TEXT2, text_color="white", command=self._translate)
        self.btn_translate.grid(row=0, column=1, padx=(5, 0))
        self.btn_copy = ctk.CTkButton(tr_ctrl, text="Kopiuj", font=ctk.CTkFont(size=8), height=28, width=50, corner_radius=8, fg_color=CARD, hover_color=CARD_H, text_color=TEXT2, border_width=1, border_color=BORDER, command=self._copy_translation)
        self.btn_copy.grid(row=0, column=2, padx=(4, 0))

        self.txt_translation = ctk.CTkTextbox(card, height=42, font=ctk.CTkFont(size=10), fg_color=BG, text_color=MUTED, corner_radius=8, border_width=0, wrap="word")
        self.txt_translation.grid(row=7, column=0, padx=12, pady=(4, 2), sticky="ew")
        self.txt_translation.insert("1.0", "Wynik tłumaczenia pojawi się tutaj.")
        self.txt_translation.configure(state="disabled")

        self.lbl_translation_state = ctk.CTkLabel(card, text="Sprawdzanie pakietów...", font=ctk.CTkFont(size=8), text_color=GREEN, anchor="w")
        self.lbl_translation_state.grid(row=8, column=0, padx=12, pady=(0, 10), sticky="ew")
        return card


    def _build_results_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        card.grid_rowconfigure(3, weight=1)
        card.grid_columnconfigure(0, weight=1)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(card, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 2))
        hdr.grid_columnconfigure(0, weight=1)
        self.lbl_results_count = ctk.CTkLabel(hdr, text="Przepisy", font=ctk.CTkFont(size=17, weight="bold"), text_color=TEXT, anchor="w")
        self.lbl_results_count.grid(row=0, column=0, sticky="w")
        self.lbl_results_sub = ctk.CTkLabel(hdr, text="Wyniki pojawia sie po transkrypcji.", font=ctk.CTkFont(size=10), text_color=MUTED, anchor="w")
        self.lbl_results_sub.grid(row=1, column=0, sticky="w")

        # ── Ingredient pills ───────────────────────────────────────────────
        pills_wrap = ctk.CTkFrame(card, fg_color="transparent")
        pills_wrap.grid(row=1, column=0, sticky="ew", padx=14, pady=(2, 2))
        pills_wrap.grid_columnconfigure(0, weight=1)
        self.ingredients_frame = tk.Frame(pills_wrap, bg=PANEL)
        self.ingredients_frame.grid(row=0, column=0, sticky="w")
        self.lbl_filter_info = ctk.CTkLabel(pills_wrap, text="", font=ctk.CTkFont(size=8), text_color=MUTED, anchor="w")
        self.lbl_filter_info.grid(row=1, column=0, sticky="ew")

        ctk.CTkFrame(card, fg_color=BORDER, height=1).grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 6))

        # ── Canvas scroll area ───────────────────────────────────────────────
        scroll_area = tk.Frame(card, bg=PANEL)
        scroll_area.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        scroll_area.grid_rowconfigure(0, weight=1)
        scroll_area.grid_columnconfigure(0, weight=1)

        self._recipe_canvas = tk.Canvas(scroll_area, bg=PANEL, highlightthickness=0, bd=0)
        self._recipe_canvas.grid(row=0, column=0, sticky="nsew")

        _sb = ctk.CTkScrollbar(scroll_area, command=self._recipe_canvas.yview)
        _sb.grid(row=0, column=1, sticky="ns")
        self._recipe_canvas.configure(yscrollcommand=_sb.set)

        self.results_scroll = tk.Frame(self._recipe_canvas, bg=PANEL)
        self._recipe_canvas_win = self._recipe_canvas.create_window((0, 0), window=self.results_scroll, anchor="nw")
        self.results_scroll.grid_columnconfigure(0, weight=1, uniform="rcol")
        self.results_scroll.grid_columnconfigure(1, weight=1, uniform="rcol")

        def _on_inner_configure(_e):
            self._recipe_canvas.configure(scrollregion=self._recipe_canvas.bbox("all"))
        self.results_scroll.bind("<Configure>", _on_inner_configure)

        def _on_canvas_resize(e):
            self._recipe_canvas.itemconfig(self._recipe_canvas_win, width=e.width)
        self._recipe_canvas.bind("<Configure>", _on_canvas_resize)

        self._recipe_canvas.bind("<Enter>", self._bind_recipe_scroll)
        self._recipe_canvas.bind("<Leave>", self._unbind_recipe_scroll)
        self.results_scroll.bind("<Enter>", self._bind_recipe_scroll)

        self._empty_label = ctk.CTkLabel(
            self.results_scroll,
            text="Brak wyników.\nNagraj głos lub wczytaj plik audio.",
            font=ctk.CTkFont(size=13), text_color=DIM, justify="center",
        )
        self._empty_label.grid(row=0, column=0, pady=80)
        return card

    def _bind_recipe_scroll(self, _=None):
        self._recipe_canvas.bind_all("<MouseWheel>", self._on_recipe_scroll)

    def _unbind_recipe_scroll(self, _=None):
        try:
            self._recipe_canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass

    def _on_recipe_scroll(self, event):
        self._recipe_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _schedule_layout_refresh(self, event=None):
        if event is not None and event.widget is not self:
            return
        if self._layout_refresh_job is not None:
            try:
                self.after_cancel(self._layout_refresh_job)
            except Exception:
                pass
        self._layout_refresh_job = self.after(80, self._refresh_layout)

    def _refresh_layout(self):
        self._layout_refresh_job = None
        self.update_idletasks()
        scaling = max(self._get_window_scaling(), 1.0)
        width = max(self.winfo_width(), self._main_frame.winfo_width(), self.winfo_reqwidth()) / scaling
        if width >= 1100:
            mode = "wide"
        else:
            mode = "compact"
        if mode == self._layout_mode:
            return
        self._layout_mode = mode
        self._apply_layout(mode)

    def _reset_grid(self, frame, rows: int, columns: int):
        for row in range(rows):
            frame.grid_rowconfigure(row, weight=0)
        for column in range(columns):
            frame.grid_columnconfigure(column, weight=0)

    def _apply_layout(self, mode: str):
        self.left_col.grid_forget()
        self.right_col.grid_forget()
        self._reset_grid(self.workspace, 3, 3)

        if mode == "wide":
            self.workspace.grid_columnconfigure(0, weight=0, minsize=242)
            self.workspace.grid_columnconfigure(1, weight=1)
            self.workspace.grid_rowconfigure(0, weight=1)
            self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
            self.right_col.grid(row=0, column=1, sticky="nsew")
            return

        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=0)
        self.right_col.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.left_col.grid(row=1, column=0, sticky="ew")


    # ── Loading ──────────────────────────────────────────────────────────

    def _start_loading(self):
        self._dots_anim = StatusDots(self.lbl_loading_step, "Inicjalizacja", MUTED)
        self._dots_anim.start()

        def load():
            self._set_loading("Ladowanie bazy przepisow")
            count = self.matcher.load(progress_callback=lambda m: self._set_loading(m))
            if count == 0:
                self._set_status("Brak przepisow w bazie", AMBER)
                self._stop_dots()
                self._hide_overlay()
                return

            self._set_loading("Ladowanie modelu mowy Whisper")
            try:
                load_model(MODELS_DIR, progress_callback=lambda m: self._set_loading(m))
                self._set_status(f"Gotowy - {count} przepisow", TEXT2)
            except Exception as e:
                self._set_status(f"Blad modelu: {e}", RED)

            self._stop_dots()
            self._hide_overlay()
            self.after(0, self._sync_vosk_buttons)

        threading.Thread(target=load, daemon=True).start()

    def _set_loading(self, msg: str):
        def update():
            if self._dots_anim:
                self._dots_anim.update_text(msg)
        self.after(0, update)

    def _stop_dots(self):
        def stop():
            if self._dots_anim:
                self._dots_anim.stop()
                self._dots_anim = None
        self.after(0, stop)

    def _hide_overlay(self):
        def hide():
            self.progress_bar.stop()
            self._overlay.place_forget()
        self.after(0, hide)

    def _show_processing_overlay(self, msg: str = "Przetwarzanie"):
        def show():
            self._proc_overlay = ctk.CTkFrame(self._main_frame, fg_color=BG, corner_radius=0)
            self._proc_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

            center = ctk.CTkFrame(
                self._proc_overlay,
                fg_color=CARD,
                corner_radius=20,
                border_width=1,
                border_color=BORDER,
            )
            center.place(relx=0.5, rely=0.42, anchor="center")

            self._proc_label = ctk.CTkLabel(
                center,
                text=msg,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=TEXT,
            )
            self._proc_label.pack(padx=28, pady=(22, 10))

            ctk.CTkLabel(
                center,
                text="Prosze czekac chwile, aplikacja przygotowuje wyniki.",
                font=ctk.CTkFont(size=11),
                text_color=MUTED,
            ).pack(padx=28, pady=(0, 14))

            self._proc_bar = ctk.CTkProgressBar(
                center,
                width=220,
                height=5,
                mode="indeterminate",
                progress_color=ACCENT, fg_color=CARD,
            )
            self._proc_bar.pack(padx=28, pady=(0, 22))
            self._proc_bar.start()

            self._proc_dots = StatusDots(self._proc_label, msg, TEXT)
            self._proc_dots.start()
        self.after(0, show)

    def _hide_processing_overlay(self):
        def hide():
            if hasattr(self, "_proc_dots"):
                self._proc_dots.stop()
            if hasattr(self, "_proc_bar"):
                self._proc_bar.stop()
            if hasattr(self, "_proc_overlay"):
                self._proc_overlay.place_forget()
        self.after(0, hide)

    # ── Recording ────────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.is_recording = True
        self.btn_record.configure(
            text="Zatrzymaj", fg_color=RED, hover_color=RED_H
        )
        self._pulse()
        self._set_status("Nagrywanie...", RED)
        self.recorder.start()

    def _stop_recording(self):
        self.is_recording = False
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job = None
        self.btn_record.configure(
            text="Nagraj", fg_color=ACCENT, hover_color=ACCENT_H
        )
        self._set_status("Przygotowanie nagrania...", TEXT2)
        audio_hq = self.recorder.stop()  # 44100 Hz for quality playback
        # Resample to 16 kHz for Whisper/Vosk
        from scipy.signal import resample as sp_resample
        audio_16k = sp_resample(audio_hq, int(len(audio_hq) * 16000 / RECORDING_SAMPLE_RATE)).astype(np.float32)
        self._audio_buffer_hq = audio_hq
        self._audio_hq_sr = RECORDING_SAMPLE_RATE
        self._finalize_audio_buffer(audio_16k, {"source": "mic", "source_sr": RECORDING_SAMPLE_RATE})

    def _pulse(self):
        if not self.is_recording:
            return
        cur = self.btn_record.cget("fg_color")
        self.btn_record.configure(fg_color=RED_H if cur == RED else RED)
        self._pulse_job = self.after(550, self._pulse)

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Wybierz plik audio",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                ("Wszystkie", "*.*"),
            ],
        )
        if not path:
            return
        self._set_status("Wczytywanie pliku...", TEXT2)
        self._show_processing_overlay("Przetwarzanie pliku audio")
        threading.Thread(target=self._process_file, args=(path,), daemon=True).start()

    def _process_file(self, path):
        from scipy.signal import resample as sp_resample

        try:
            audio, sr = sf.read(path, dtype="float32")
            ch = int(audio.shape[1]) if audio.ndim > 1 else 1
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            src_sr = int(sr)
            if src_sr != 16000:
                audio = sp_resample(audio, int(len(audio) * 16000 / src_sr))
            meta = {
                "source": "file",
                "path": path,
                "source_sr": src_sr,
                "channels": ch,
            }
        except Exception as exc:
            self.after(0, lambda: self._set_status(f"Blad pliku: {exc}", RED))
            self.after(0, self._hide_processing_overlay)
            return

        def handoff():
            self._hide_processing_overlay()
            self._finalize_audio_buffer(audio, meta)

        self.after(0, handoff)

    # ── Audio buffer + STT (Whisper / Vosk) ────────────────────────────────

    def _finalize_audio_buffer(self, audio: np.ndarray, meta: dict):
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            self._set_status("Puste nagranie", AMBER)
            return
        if float(np.max(np.abs(audio))) > 1.0:
            audio = audio / 32768.0

        self._audio_buffer = audio
        self._audio_meta = dict(meta)
        self._text_whisper = ""
        self._text_vosk = ""
        self._text_hf_wv = ""
        self._text_mms = ""
        self._lang_whisper = None
        self._lang_vosk = None
        self._lang_hf_wv = None
        self._lang_mms = None
        self._detected_lang = None
        if meta.get("source") != "mic":
            self._audio_buffer_hq = None
            self._audio_hq_sr = 16000

        self.txt_stt.configure(state="normal")
        self.txt_stt.delete("1.0", "end")

        self._update_audio_meta_label()
        self.btn_save_audio.configure(state="normal")
        self.btn_play_audio.configure(state="normal")
        # keep btn_stt disabled until language detection finishes
        self.btn_stt.configure(state="disabled")
        self._sync_vosk_buttons()
        self.lbl_language.configure(text="wykrywanie...")
        threading.Thread(target=self._detect_lang_worker, daemon=True).start()

    def _update_audio_meta_label(self):
        if self._audio_buffer is None:
            self.lbl_audio_meta.configure(text="Brak nagrania")
            return
        dur = len(self._audio_buffer) / 16000.0
        src = self._audio_meta.get("source", "")
        extra = ""
        if src == "file" and self._audio_meta.get("source_sr"):
            o = int(self._audio_meta["source_sr"])
            extra = f" | plik {o} Hz -> 16 kHz"
        ch = self._audio_meta.get("channels", 1)
        self.lbl_audio_meta.configure(
            text=f"Dlugosc: {dur:.2f} s | 16 kHz | kanaly: {ch}{extra}",
        )

    def _sync_vosk_buttons(self):
        hints: list[str] = []
        if vosk_available(MODELS_DIR):
            self.rb_vosk.configure(state="normal")
        else:
            self.rb_vosk.configure(state="disabled")
            if self.stt_engine.get() == "vosk":
                self.stt_engine.set("whisper")
            hints.append("Vosk: brak modeli — uruchom setup.py")
        hf_ok = hf_wav2vec_transformers_available() and mms_transformers_available()
        if hf_ok:
            self.rb_hf_wv.configure(state="normal")
            self.rb_mms.configure(state="normal")
        else:
            self.rb_hf_wv.configure(state="disabled")
            self.rb_mms.configure(state="disabled")
            if self.stt_engine.get() in ("hf_wav2vec", "meta_mms"):
                self.stt_engine.set("whisper")
            hints.append("Hugging Face (Wav2Vec2 / MMS): pip install transformers")
        if hints:
            self.lbl_vosk_hint.configure(text=" · ".join(hints))
            self.lbl_vosk_hint.grid()
        else:
            self.lbl_vosk_hint.grid_remove()

    def _detect_lang_worker(self):
        try:
            code = detect_language_code(self._audio_buffer, MODELS_DIR)
            name = LANG_NAMES.get(code, code)
            self._detected_lang = code

            def ok():
                self.lbl_language.configure(text=f"{name} ({code})")
                self.btn_stt.configure(state="normal")
                self._sync_vosk_buttons()
                # auto-transcribe with selected engine now that language is known
                threading.Thread(target=self._thread_transcribe_active, daemon=True).start()

            self.after(0, ok)
        except Exception:
            self._detected_lang = None

            def err():
                self.lbl_language.configure(text="nie wykryto")
                self.btn_stt.configure(state="normal")

            self.after(0, err)

    @staticmethod
    def _vosk_lang_from_detect(code: str | None) -> str | None:
        if not code:
            return None
        c = str(code).lower().strip()
        if c == "pl":
            return "pl"
        if c == "en":
            return "en"
        return None

    def _save_audio_buffer(self):
        if self._audio_buffer is None:
            return
        path = filedialog.asksaveasfilename(
            title="Zapisz nagranie",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav"), ("Wszystkie", "*.*")],
        )
        if not path:
            return
        try:
            buf = self._audio_buffer_hq if self._audio_buffer_hq is not None else self._audio_buffer
            sr = self._audio_hq_sr if self._audio_buffer_hq is not None else 16000
            sf.write(path, buf, sr, subtype="PCM_16")
            self._set_status(f"Zapisano: {path}", GREEN)
        except Exception as exc:
            self._set_status(f"Blad zapisu: {exc}", RED)

    def _play_audio_buffer(self):
        if self._audio_buffer is None:
            return

        def play():
            try:
                self.after(0, lambda: self._set_status("Odtwarzanie...", TEXT2))
                buf = self._audio_buffer_hq if self._audio_buffer_hq is not None else self._audio_buffer
                sr = self._audio_hq_sr if self._audio_buffer_hq is not None else 16000
                sd.play(buf.astype(np.float32), sr)
                sd.wait()
                self.after(0, lambda: self._set_status("Gotowy", TEXT2))
            except Exception as exc:
                self.after(0, lambda: self._set_status(f"Blad odtwarzania: {exc}", RED))

        threading.Thread(target=play, daemon=True).start()

    # ── Engine selection + auto-transcribe ────────────────────────────────

    def _on_engine_change(self):
        """Called when user switches STT engine radio.
        Shows cached result if available; never auto-transcribes on switch."""
        engine = self.stt_engine.get()
        if engine == "whisper":
            cached, lang = self._text_whisper, self._lang_whisper
        elif engine == "vosk":
            cached, lang = self._text_vosk, self._lang_vosk
        elif engine == "hf_wav2vec":
            cached, lang = self._text_hf_wv, self._lang_hf_wv
        elif engine == "meta_mms":
            cached, lang = self._text_mms, self._lang_mms
        else:
            cached, lang = self._text_whisper, self._lang_whisper
        if cached:
            self._update_stt_display(cached)
            if lang:
                self._apply_recipe_pipeline_from_text(cached, lang)
        else:
            # clear display — user needs to click Transkrybuj
            self._update_stt_display("")

    def _transcribe_action(self):
        """Manual retranscribe (or first run) with current engine."""
        if self._audio_buffer is None:
            return
        eng = self.stt_engine.get()
        if eng == "whisper":
            self._show_processing_overlay("Transkrypcja Whisper")
            threading.Thread(target=self._thread_transcribe_whisper, daemon=True).start()
        elif eng == "vosk":
            self._transcribe_vosk_action()
        elif eng == "hf_wav2vec":
            self._transcribe_hf_wav2vec_action()
        elif eng == "meta_mms":
            self._transcribe_mms_action()

    def _thread_transcribe_active(self):
        """Background thread: transcribe with whichever engine is selected."""
        engine = self.stt_engine.get()
        try:
            if engine == "whisper":
                lang_hint = self._detected_lang
                result = transcribe_whisper(self._audio_buffer, MODELS_DIR, language=lang_hint)
                self.after(0, lambda r=result: self._on_whisper_done(r))
            elif engine == "hf_wav2vec":
                vk = self._vosk_lang_from_detect(self._detected_lang)
                if vk and hf_wav2vec_transformers_available():
                    result = transcribe_hf_wav2vec(self._audio_buffer, MODELS_DIR, vk)
                    self.after(0, lambda r=result: self._on_hf_wav2vec_done(r))
                elif vk and not hf_wav2vec_transformers_available():
                    self.after(
                        0,
                        lambda: self._update_stt_display(
                            "Brak pakietu transformers — pip install transformers"
                        ),
                    )
                else:
                    self.after(
                        0,
                        lambda: self._update_stt_display(
                            "Wav2Vec2 HF: wykryty język nieobsługiwany (tylko PL/EN)"
                        ),
                    )
            elif engine == "meta_mms":
                vk = self._vosk_lang_from_detect(self._detected_lang)
                if vk and mms_transformers_available():
                    result = transcribe_mms(self._audio_buffer, MODELS_DIR, vk)
                    self.after(0, lambda r=result: self._on_mms_done(r))
                elif vk and not mms_transformers_available():
                    self.after(
                        0,
                        lambda: self._update_stt_display(
                            "Brak pakietu transformers — pip install transformers"
                        ),
                    )
                else:
                    self.after(
                        0,
                        lambda: self._update_stt_display(
                            "Meta MMS: wykryty język nieobsługiwany (tylko PL/EN)"
                        ),
                    )
            else:
                vk = self._vosk_lang_from_detect(self._detected_lang)
                if vk and vosk_available(MODELS_DIR):
                    result = transcribe_vosk(self._audio_buffer, MODELS_DIR, vk)
                    self.after(0, lambda r=result: self._on_vosk_done(r))
                elif vk and not vosk_available(MODELS_DIR):
                    self.after(0, lambda: self._update_stt_display("Brak modeli Vosk — uruchom setup.py"))
                else:
                    self.after(0, lambda: self._update_stt_display("Vosk: wykryty język nieobsługiwany (tylko PL/EN)"))
        except Exception as exc:
            self.after(0, lambda: self._stt_failed(exc))

    def _update_stt_display(self, text: str):
        """Update the single transcription textbox."""
        self.txt_stt.configure(state="normal")
        self.txt_stt.delete("1.0", "end")
        self.txt_stt.insert("1.0", text)

    def _transcribe_whisper_action(self):
        if self._audio_buffer is None:
            return
        self._show_processing_overlay("Transkrypcja Whisper")
        threading.Thread(target=self._thread_transcribe_whisper, daemon=True).start()

    def _transcribe_hf_wav2vec_action(self):
        if self._audio_buffer is None:
            return
        if not hf_wav2vec_transformers_available():
            self._set_status("Brak pakietu transformers — pip install transformers", AMBER)
            return
        vk = self._vosk_lang_from_detect(self._detected_lang)
        if vk is None:
            self._update_stt_display("Wav2Vec2 HF: wykryty język nieobsługiwany (tylko PL/EN)")
            self._set_status("Wav2Vec2 HF: nieobsługiwany język", AMBER)
            return
        self._show_processing_overlay("Transkrypcja Wav2Vec2 (HF)")
        threading.Thread(
            target=self._thread_transcribe_hf_wav2vec,
            args=(vk,),
            daemon=True,
        ).start()

    def _thread_transcribe_hf_wav2vec(self, vk: str):
        try:
            result = transcribe_hf_wav2vec(self._audio_buffer, MODELS_DIR, vk)
            self.after(0, lambda r=result: self._on_hf_wav2vec_done(r))
        except Exception as exc:
            self.after(0, lambda: self._stt_failed(exc))
        finally:
            self.after(0, self._hide_processing_overlay)

    def _transcribe_mms_action(self):
        if self._audio_buffer is None:
            return
        if not mms_transformers_available():
            self._set_status("Brak pakietu transformers — pip install transformers", AMBER)
            return
        vk = self._vosk_lang_from_detect(self._detected_lang)
        if vk is None:
            self._update_stt_display("Meta MMS: wykryty język nieobsługiwany (tylko PL/EN)")
            self._set_status("Meta MMS: nieobsługiwany język", AMBER)
            return
        self._show_processing_overlay("Transkrypcja Meta MMS")
        threading.Thread(
            target=self._thread_transcribe_mms,
            args=(vk,),
            daemon=True,
        ).start()

    def _thread_transcribe_mms(self, vk: str):
        try:
            result = transcribe_mms(self._audio_buffer, MODELS_DIR, vk)
            self.after(0, lambda r=result: self._on_mms_done(r))
        except Exception as exc:
            self.after(0, lambda: self._stt_failed(exc))
        finally:
            self.after(0, self._hide_processing_overlay)

    def _thread_transcribe_whisper(self):
        try:
            lang_hint = self._detected_lang
            result = transcribe_whisper(
                self._audio_buffer,
                MODELS_DIR,
                language=lang_hint,
            )
            self.after(0, lambda r=result: self._on_whisper_done(r))
        except Exception as exc:
            self.after(0, lambda: self._stt_failed(exc))
        finally:
            self.after(0, self._hide_processing_overlay)

    def _on_whisper_done(self, result: dict):
        self._text_whisper = (result.get("text") or "").strip()
        self._lang_whisper = (result.get("language") or self._detected_lang or "pl").lower()
        if self.stt_engine.get() == "whisper":
            self._update_stt_display(self._text_whisper)
        self.lbl_language.configure(text=result.get("language_name", ""))
        self._refresh_translation_support()
        if self.stt_engine.get() == "whisper":
            self._apply_recipe_pipeline_from_text(self._text_whisper, self._lang_whisper)
        self._set_status("Transkrypcja Whisper gotowa", GREEN)

    def _on_hf_wav2vec_done(self, result: dict):
        text = (result.get("text") or "").strip()
        lang = (result.get("language") or self._detected_lang or "pl").lower()
        self._text_hf_wv = text
        self._lang_hf_wv = lang
        if self.stt_engine.get() == "hf_wav2vec":
            self._update_stt_display(text)
        self.lbl_language.configure(text=result.get("language_name", ""))
        self._refresh_translation_support()
        if self.stt_engine.get() == "hf_wav2vec":
            self._apply_recipe_pipeline_from_text(text, lang)
        self._set_status("Transkrypcja Wav2Vec2 (HF) gotowa", GREEN)

    def _on_mms_done(self, result: dict):
        text = (result.get("text") or "").strip()
        lang = (result.get("language") or self._detected_lang or "pl").lower()
        self._text_mms = text
        self._lang_mms = lang
        if self.stt_engine.get() == "meta_mms":
            self._update_stt_display(text)
        self.lbl_language.configure(text=result.get("language_name", ""))
        self._refresh_translation_support()
        if self.stt_engine.get() == "meta_mms":
            self._apply_recipe_pipeline_from_text(text, lang)
        self._set_status("Transkrypcja Meta MMS gotowa", GREEN)

    def _transcribe_vosk_action(self):
        if self._audio_buffer is None:
            return
        if not vosk_available(MODELS_DIR):
            self._set_status("Brak modeli Vosk — uruchom setup.py", AMBER)
            return
        vk = self._vosk_lang_from_detect(self._detected_lang)
        if vk is None:
            msg = "Vosk: wykryty język nieobsługiwany (tylko PL/EN) — użyj Whisper."
            self._update_stt_display(msg)
            self._text_vosk = ""
            self._lang_vosk = None
            self._set_status("Vosk: nieobsługiwany język", AMBER)
            return
        self._show_processing_overlay("Transkrypcja Vosk")
        threading.Thread(
            target=self._thread_transcribe_vosk,
            args=(vk,),
            daemon=True,
        ).start()

    def _thread_transcribe_vosk(self, vk: str):
        try:
            result = transcribe_vosk(self._audio_buffer, MODELS_DIR, vk)
            self.after(0, lambda r=result: self._on_vosk_done(r))
        except Exception as exc:
            self.after(0, lambda: self._stt_failed(exc))
        finally:
            self.after(0, self._hide_processing_overlay)

    def _on_vosk_done(self, result: dict):
        self._text_vosk = (result.get("text") or "").strip()
        self._lang_vosk = (result.get("language") or "pl").lower()
        if self.stt_engine.get() == "vosk":
            self._update_stt_display(self._text_vosk)
        self._refresh_translation_support()
        if self.stt_engine.get() == "vosk":
            self._apply_recipe_pipeline_from_text(self._text_vosk, self._lang_vosk)
        self._set_status("Transkrypcja Vosk gotowa", GREEN)

    def _set_vosk_placeholder(self, msg: str):
        self._text_vosk = ""
        self._lang_vosk = None
        if self.stt_engine.get() == "vosk":
            self._update_stt_display(msg)

    def _stt_failed(self, exc: Exception):
        self._set_status(f"Błąd STT: {exc}", RED)

    def _apply_recipe_pipeline_from_text(self, text: str, lang_code: str):
        text = (text or "").strip()
        if not text:
            return
        try:
            self.current_language = (str(lang_code).lower().strip() if lang_code else "pl")
            raw = extract_ingredients(text, self.matcher.vocabulary)
            detected_ingredients = self._deduplicate_ingredients(raw)
            ingredients = self._cleanup_search_ingredients(detected_ingredients)
            ignored_ingredients = [
                ingredient
                for ingredient in detected_ingredients
                if not any(tokens_match(ingredient, kept) for kept in ingredients)
            ]
            self.detected_ingredients = detected_ingredients
            self.current_ingredients = ingredients
            self._update_ingredients(detected_ingredients, ignored_ingredients)

            results = self._search_recipes(ingredients)
            self.current_results = results
            self._update_results(results)
            self._set_status(f"Znaleziono {len(results)} przepisow", GREEN)
            self._refresh_translation_support()
        except Exception as exc:
            self._set_status(f"Blad: {exc}", RED)

    def _transcription_text_for_translate(self) -> str:
        e = self.stt_engine.get()
        if e == "vosk":
            return self._text_vosk.strip()
        if e == "hf_wav2vec":
            return self._text_hf_wv.strip()
        if e == "meta_mms":
            return self._text_mms.strip()
        return self._text_whisper.strip()

    def _deduplicate_ingredients(self, ingredients: list) -> list:
        result = []
        for ing in ingredients:
            if not any(tokens_match(ing, existing) for existing in result):
                result.append(ing)
        return result

    def _format_ignored_hint(self, ignored: list[str]) -> str:
        if not ignored:
            return ""
        staples = [
            i
            for i in ignored
            if any(tokens_match(i, s) for s in LOW_SIGNAL_INGREDIENTS)
        ]
        generic = [
            i
            for i in ignored
            if any(tokens_match(i, g) for g in GENERIC_INGREDIENTS)
        ]
        rest = [i for i in ignored if i not in staples and i not in generic]
        parts = []
        if staples:
            parts.append(
                "bez wpływu na ranking (typowe dodatki): "
                + ", ".join(staples)
            )
        if generic:
            parts.append(
                "zastąpione konkretnym składnikiem: " + ", ".join(generic)
            )
        if rest:
            parts.append("pominięto: " + ", ".join(rest))
        return " · ".join(parts)

    def _staple_supplements_for_recipe(self, recipe_idx: int) -> list[str]:
        if recipe_idx < 0:
            return []
        core = self.matcher.matched_ingredients(self.current_ingredients, recipe_idx)
        out: list[str] = []
        for ing in self.detected_ingredients:
            if any(tokens_match(ing, c) for c in core):
                continue
            if not any(tokens_match(ing, s) for s in LOW_SIGNAL_INGREDIENTS):
                continue
            term = normalize_text(ing)
            if term and self.matcher._ingredient_matches_recipe(term, recipe_idx):
                if not any(tokens_match(ing, o) for o in out):
                    out.append(ing)
        return out

    def _cleanup_search_ingredients(self, ingredients: list) -> list:
        cleaned = self._deduplicate_ingredients(ingredients)
        cleaned = [
            ingredient
            for ingredient in cleaned
            if not any(tokens_match(ingredient, ignored) for ignored in LOW_SIGNAL_INGREDIENTS)
        ]
        has_specific_protein = any(
            any(tokens_match(ingredient, protein) for protein in SPECIFIC_PROTEINS)
            for ingredient in cleaned
        )
        if has_specific_protein:
            cleaned = [
                ingredient
                for ingredient in cleaned
                if not any(tokens_match(ingredient, generic) for generic in GENERIC_INGREDIENTS)
            ]
        return cleaned

    def _search_recipes(self, ingredients: list) -> list:
        results = self.matcher.search(ingredients, top_n=50)
        results.sort(
            key=lambda recipe: (
                self.matcher.match_count(ingredients, recipe.get("_idx", -1)),
                recipe.get("_score", 0),
                recipe.get("title", ""),
            ),
            reverse=True,
        )
        return results

    # ── UI updates ───────────────────────────────────────────────────────

    def _update_ingredients(self, ingredients, ignored_ingredients=None):
        for w in self.ingredients_frame.winfo_children():
            w.destroy()
        ignored_ingredients = ignored_ingredients or []
        if not ingredients:
            self.lbl_results_sub.configure(text="nagraj składniki — wyniki pojawią się tutaj")
            self.lbl_filter_info.configure(text="")
            return
        for ing in ingredients:
            ctk.CTkLabel(
                self.ingredients_frame, text=ing,
                font=ctk.CTkFont(size=9),
                fg_color=ACCENT_SOFT, text_color=ACCENT,
                corner_radius=8,
            ).pack(side="left", padx=(0, 4), pady=2)
        used = self.current_ingredients or ingredients
        ignored_ingredients = ignored_ingredients or []
        hint = self._format_ignored_hint(ignored_ingredients)
        if hint:
            self.lbl_filter_info.configure(text=hint)
        else:
            self.lbl_filter_info.configure(text="")

    def _update_results(self, results):
        for w in self.results_scroll.winfo_children():
            w.destroy()
        self._recipe_canvas.yview_moveto(0)

        if not results:
            self.lbl_results_count.configure(text="Przepisy")
            self.lbl_results_sub.configure(text="Nie znaleziono pasujących przepisów")
            ctk.CTkLabel(
                self.results_scroll,
                text="Brak wyników.\nSpróbuj innych składników.",
                font=ctk.CTkFont(size=12), text_color=DIM, justify="center",
            ).grid(row=0, column=0, pady=60)
            return

        total_asked = len(self.current_ingredients)
        self.lbl_results_count.configure(text=f"{len(results)} przepisów")
        if total_asked:
            self.lbl_results_sub.configure(text=f"posortowane wg trafności · {total_asked} skł.")
        else:
            self.lbl_results_sub.configure(text="posortowane według trafności")
        self.results_scroll.grid_columnconfigure(0, weight=1, uniform="rcol")
        self.results_scroll.grid_columnconfigure(1, weight=1, uniform="rcol")
        for display_idx, recipe in enumerate(results):
            recipe_idx = recipe.get("_idx", -1)
            matched_count = self.matcher.match_count(self.current_ingredients, recipe_idx)
            self._add_card(display_idx, recipe, matched_count, total_asked)

    def _add_card(self, display_idx, recipe, matched_count, total_asked):
        recipe_idx = recipe.get("_idx", -1)
        ing_count = len(self.matcher.preview_ingredients(recipe) or [])

        grid_row = display_idx // 2
        grid_col = display_idx % 2
        pad_x = (0, 6) if grid_col == 0 else (6, 0)

        # Score colours
        if total_asked > 0 and matched_count == total_asked:
            accent_bar, inner_bg = GREEN, "#f2fbf5"
            chip_bg, chip_fg = "#c8eed9", MATCH_FULL
        elif total_asked > 0 and matched_count > 0:
            accent_bar, inner_bg = AMBER, "#fdf7e8"
            chip_bg, chip_fg = "#f5e4b0", MATCH_PART
        else:
            accent_bar, inner_bg = BORDER, CARD
            chip_bg, chip_fg = TAG_BG, TEXT2

        card = ctk.CTkFrame(self.results_scroll, fg_color=inner_bg,
                            corner_radius=14, border_width=1, border_color=BORDER)
        card.grid(row=grid_row, column=grid_col, sticky="nsew",
                  padx=pad_x, pady=(0, 10))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        inner = tk.Frame(card, bg=inner_bg)
        inner.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        inner.grid_columnconfigure(0, weight=1)

        # ── Title row + match badge ─────────────────────────────────────────
        title_row = tk.Frame(inner, bg=inner_bg)
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        title_text = recipe.get("title", "Bez nazwy")
        tk.Label(title_row, text=title_text, bg=inner_bg, fg=TEXT,
                 font=("Segoe UI", 18, "bold"),
                 anchor="w", justify="left", wraplength=360,
                 ).grid(row=0, column=0, sticky="ew")

        # match badge top-right
        if total_asked > 0 and matched_count > 0:
            sym = "✓" if matched_count == total_asked else "~"
            badge_text = f"{sym} {matched_count}/{total_asked}"
            badge_bg = "#c8eed9" if matched_count == total_asked else "#f5e4b0"
            badge_fg = MATCH_FULL if matched_count == total_asked else MATCH_PART
            badge = tk.Frame(title_row, bg=badge_bg)
            badge.grid(row=0, column=1, sticky="ne", padx=(10, 0))
            tk.Label(badge, text=badge_text, bg=badge_bg, fg=badge_fg,
                     font=("Segoe UI", 11, "bold"), padx=9, pady=4).pack()

        # ── Category ───────────────────────────────────────────────────────
        cat = (recipe.get("category") or "").strip()
        if cat:
            tk.Label(inner, text=cat, bg=inner_bg, fg=MUTED,
                     font=("Segoe UI", 12), anchor="w",
                     ).grid(row=1, column=0, sticky="w", pady=(5, 10))

        # ── Matched ingredient pills ────────────────────────────────────────
        if total_asked > 0 and matched_count > 0:
            pills_src = self.matcher.matched_ingredients(self.current_ingredients, recipe_idx)
        else:
            pills_src = (self.matcher.preview_ingredients(recipe) or [])[:4]

        if pills_src:
            chips_row = tk.Frame(inner, bg=inner_bg)
            chips_row.grid(row=2, column=0, sticky="w")
            shown = 0
            for t in pills_src:
                label = str(t)
                label = label[:16] + "…" if len(label) > 16 else label
                tk.Label(chips_row, text=label, bg=chip_bg, fg=chip_fg,
                         font=("Segoe UI", 11), padx=9, pady=5,
                         ).pack(side="left", padx=(0, 5), pady=2)
                shown += 1
                if shown >= 5:
                    break
            remaining = ing_count - shown
            if remaining > 0:
                tk.Label(chips_row, text=f"+{remaining}", bg=inner_bg, fg=DIM,
                         font=("Segoe UI", 11)).pack(side="left", padx=(3, 0))

        def on_click(_e, r=recipe):
            self._show_detail(r)

        def on_enter(_e):
            card.configure(border_color=ACCENT)
            self._bind_recipe_scroll()

        def on_leave(_e):
            card.configure(border_color=BORDER)

        try:
            card.configure(cursor="hand2")
        except Exception:
            pass

        all_w = [card, inner] + list(inner.winfo_children())
        for w in all_w:
            w.bind("<Button-1>", on_click)
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<MouseWheel>", self._on_recipe_scroll)
        for child in inner.winfo_children():
            for gc in child.winfo_children():
                gc.bind("<Button-1>", on_click)
                gc.bind("<MouseWheel>", self._on_recipe_scroll)

    def _show_detail(self, recipe):
        recipe_idx = recipe.get("_idx", -1)
        now = time.monotonic()
        last = getattr(self, "_last_detail_open", None)
        if last and last[0] == recipe_idx and (now - last[1]) < 0.45:
            return
        self._last_detail_open = (recipe_idx, now)
        matched = self.matcher.matched_ingredients(self.current_ingredients, recipe_idx)
        total_asked = len(self.current_ingredients)
        supplement = self._staple_supplements_for_recipe(recipe_idx)
        DetailWindow(self, recipe, matched, total_asked, matched_supplement=supplement)

    # ── Translation ──────────────────────────────────────────────────────

    def _resolve_target_lang_code(self) -> str | None:
        label = (self.combo_translate.get() or "").strip()
        if not label:
            return None
        try:
            for code, name in get_available_targets(self.current_language):
                if name == label:
                    return code
                if _fold_label(name) == _fold_label(label):
                    return code
        except Exception:
            pass
        code = TRANSLATION_LABEL_TO_CODE.get(label)
        if code:
            return code
        folded = _fold_label(label)
        for lbl, c in TRANSLATION_LABEL_TO_CODE.items():
            if isinstance(lbl, str) and _fold_label(lbl) == folded:
                return c
        return None

    def _translate(self):
        text = self._transcription_text_for_translate()
        if not text:
            return
        configure_translation_packages(TRANSLATIONS_DIR)
        target = self._resolve_target_lang_code()
        available_targets = {code for code, _ in get_available_targets(self.current_language)}
        if target is None or target not in available_targets:
            self._refresh_translation_support()
            self._translation_show_error(self._translation_unavailable_message())
            return
        source = (self.current_language or "pl").lower()
        if source == target:
            self._translation_apply_success(text)
            self.lbl_translation_state.configure(
                text="Tekst jest juz w wybranym jezyku docelowym.",
                text_color=GREEN,
            )
            return

        self._set_translation_progress("Tlumaczenie...")

        def do():
            try:
                configure_translation_packages(TRANSLATIONS_DIR)
                result = translate_text(text, source, target)
                self.after(
                    0,
                    lambda res=result: self._translation_apply_success(res),
                )
            except TranslationUnavailable as exc:
                err = (str(exc) or self._translation_unavailable_message())[:420]

                def show_translation_error():
                    try:
                        self._refresh_translation_support()
                    except Exception:
                        pass
                    self._translation_show_error(err)

                self.after(0, show_translation_error)
            except Exception as exc:
                self.after(0, lambda e=exc: self._translation_show_error(f"Blad: {e}"))

        threading.Thread(target=do, daemon=True).start()

    def _set_translation_progress(self, msg: str):
        self.txt_translation.configure(state="normal", text_color=MUTED)
        self.txt_translation.delete("1.0", "end")
        self.txt_translation.insert("1.0", msg)
        self.txt_translation.configure(state="disabled")

    def _translation_apply_success(self, result: str):
        text_out = (result or "").strip()
        if not text_out:
            self._translation_show_error(
                "Tlumaczenie zwrocilo pusty tekst. W wersji EXE doinstaluj zaleznosci "
                "(ctranslate2) albo uruchom ponownie build.bat po aktualizacji."
            )
            return
        self.txt_translation.configure(state="normal", text_color=TEXT)
        self.txt_translation.delete("1.0", "end")
        self.txt_translation.insert("1.0", text_out)
        self.txt_translation.configure(state="disabled", text_color=TEXT)
        tgt_label = self.combo_translate.get()
        self.lbl_translation_state.configure(
            text=f"Tlumaczenie gotowe → {tgt_label}.",
            text_color=GREEN,
        )

    def _translation_show_error(self, msg: str):
        self.txt_translation.configure(state="normal")
        self.txt_translation.delete("1.0", "end")
        self.txt_translation.insert("1.0", "")
        self.txt_translation.configure(state="disabled")
        self.lbl_translation_state.configure(text=msg, text_color=AMBER)

    def _copy_translation(self):
        text = self.txt_translation.get("1.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.btn_copy.configure(text="OK!")
            self.after(1500, lambda: self.btn_copy.configure(text="Kopiuj"))

    def _translation_unavailable_message(self) -> str:
        source_name = TRANSLATION_LABELS.get(self.current_language, self.current_language)
        try:
            available = get_available_targets(self.current_language)
        except Exception:
            available = []
        if available:
            names = ", ".join(name for _, name in available)
            return f"Dla jezyka {source_name} gotowe sa tylko: {names}."
        return (
            "Brak wbudowanego tlumaczenia dla wykrytego jezyka. "
            "Dostepne jezyki z pakietami offline: polski, angielski, niemiecki, francuski, hiszpanski."
        )

    def _refresh_translation_support(self):
        try:
            configure_translation_packages(TRANSLATIONS_DIR)
            available = get_available_targets(self.current_language)
        except Exception:
            available = []

        values = [name for _, name in available]
        if values:
            current_value = self.combo_translate.get()
            self.combo_translate.configure(values=values)
            if current_value not in values:
                self.combo_translate.set(values[0])
            self.btn_translate.configure(state="normal")
            self.lbl_translation_state.configure(
                text=f"Pakiety offline gotowe dla jezyka {TRANSLATION_LABELS.get(self.current_language, self.current_language)}.",
                text_color=GREEN,
            )
        else:
            fallback_values = [name for code, name in TRANSLATION_LABELS.items() if code != self.current_language]
            self.combo_translate.configure(values=fallback_values or ["angielski"])
            if fallback_values:
                self.combo_translate.set(fallback_values[0])
            self.btn_translate.configure(state="disabled")
            self.lbl_translation_state.configure(
                text=self._translation_unavailable_message(),
                text_color=AMBER,
            )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = TEXT2):
        badge_color = ACCENT_SOFT
        if color == GREEN:
            badge_color = GREEN_SOFT
        elif color == AMBER:
            badge_color = AMBER_SOFT
        elif color == RED:
            badge_color = "#3b1d22"

        def update():
            self.lbl_status.configure(text=msg, text_color=color)
            if hasattr(self, "status_card"):
                self.status_card.configure(fg_color=badge_color)

        self.after(0, update)


if __name__ == "__main__":
    App().mainloop()
