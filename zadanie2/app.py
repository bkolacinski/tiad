import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import threading
import time
import unicodedata
import customtkinter as ctk
from tkinter import filedialog


def _asset_ready(asset_name: str, path: str) -> bool:
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

from stt import AudioRecorder, load_model, transcribe
from ingredient_extractor import extract_ingredients
from recipe_matcher import RecipeMatcher
from text_utils import tokens_match
from translator import (
    TranslationUnavailable,
    configure_translation_packages,
    get_available_targets,
    translate_text,
)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("dark-blue")

# ── Palette: neutral, minimal ─────────────────────────────────────────────
BG = "#edf2f7"
PANEL = "#f8fbff"
CARD = "#ffffff"
CARD_H = "#f4f8fc"
ACCENT = "#2563eb"
ACCENT_H = "#1d4ed8"
ACCENT_SOFT = "#dbeafe"
RED = "#dc2626"
RED_H = "#b91c1c"
GREEN = "#16a34a"
GREEN_SOFT = "#dcfce7"
AMBER = "#d97706"
AMBER_SOFT = "#fef3c7"
TEXT = "#0f172a"
TEXT2 = "#334155"
MUTED = "#64748b"
DIM = "#94a3b8"
BORDER = "#d9e2ec"
TAG_BG = "#eef4ff"

TRANSLATION_LABELS = {
    "pl": "polski",
    "en": "angielski",
    "de": "niemiecki",
    "fr": "francuski",
    "es": "hiszpanski",
}
TRANSLATION_LABEL_TO_CODE = {label: code for code, label in TRANSLATION_LABELS.items()}
# Argos zwraca też nazwę z polską literą — musi mapować na ten sam kod
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

def _fuzzy_ingredient_match(ingredient: str, text: str) -> bool:
    return any(tokens_match(ingredient, word) for word in text.lower().split())


def _flow_layout_chips(parent, items, bg_color, text_color, font_size=11, max_columns=4):
    row_frame = ctk.CTkFrame(parent, fg_color="transparent")
    row_frame.pack(fill="x", anchor="w", padx=2, pady=2)
    col = 0
    row_idx = 0
    for item in items:
        pill = ctk.CTkFrame(
            row_frame,
            fg_color=bg_color,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        pill.grid(row=row_idx, column=col, padx=(0, 5), pady=2, sticky="w")
        ctk.CTkLabel(
            pill, text=item, font=ctk.CTkFont(size=font_size), text_color=text_color
        ).pack(padx=10, pady=4)
        col += 1
        if col >= max_columns:
            col = 0
            row_idx += 1


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
class DetailWindow(ctk.CTkToplevel):
    def __init__(self, parent, recipe: dict, matched: list, total_asked: int = 0):
        super().__init__(parent)
        title = recipe.get("title", "Przepis")
        self.title(title)
        self.geometry("720x700")
        self.resizable(True, True)
        self.configure(fg_color=BG)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.transient(parent)
        self.after(50, self._finalize_window)

        # Header
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=72)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr, text=title,
            font=ctk.CTkFont(size=17, weight="bold"), text_color=TEXT, anchor="w",
        ).grid(row=0, column=0, padx=24, pady=(16, 2), sticky="w")

        cat = recipe.get("category", "")
        url = recipe.get("url", "")
        meta = "  /  ".join(x for x in [cat, url] if x)
        if meta:
            ctk.CTkLabel(
                hdr, text=meta, font=ctk.CTkFont(size=11), text_color=MUTED, anchor="w"
            ).grid(row=1, column=0, padx=24, pady=(0, 12), sticky="w")

        # Body
        body = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_columnconfigure(0, weight=1)

        ings = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", recipe.get("steps", ""))
        row = 0
        total_display = max(total_asked, len(matched))

        if matched:
            banner = ctk.CTkFrame(body, fg_color="#14291e", corner_radius=8)
            banner.grid(row=row, column=0, sticky="ew", padx=24, pady=(20, 0))
            banner.grid_columnconfigure(0, weight=1)
            row += 1
            ctk.CTkLabel(
                banner,
                text=f"Pasuje {len(matched)} z {total_display} skladnikow",
                font=ctk.CTkFont(size=12, weight="bold"), text_color=GREEN, anchor="w",
            ).grid(row=0, column=0, padx=14, pady=(10, 6), sticky="w")
            chip_wrap = ctk.CTkFrame(banner, fg_color="transparent")
            chip_wrap.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
            _flow_layout_chips(chip_wrap, matched, GREEN, BG)

        # Ingredients section
        ctk.CTkLabel(
            body,
            text=f"Skladniki ({len(ings) if isinstance(ings, list) else '?'})",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT,
        ).grid(row=row, column=0, padx=24, pady=(20, 8), sticky="w")
        row += 1

        ing_frame = ctk.CTkFrame(body, fg_color=CARD, corner_radius=8)
        ing_frame.grid(row=row, column=0, padx=24, sticky="ew")
        ing_frame.grid_columnconfigure(0, weight=1)
        row += 1

        matched_lower = [m.lower() for m in matched]
        if isinstance(ings, list):
            for i, ing in enumerate(ings):
                bg = CARD if i % 2 == 0 else CARD_H
                r = ctk.CTkFrame(ing_frame, fg_color=bg, corner_radius=0, height=30)
                r.grid(row=i, column=0, sticky="ew")
                r.grid_columnconfigure(0, weight=1)
                r.grid_propagate(False)
                ing_str = str(ing)
                is_matched = any(
                    _fuzzy_ingredient_match(m, ing_str.lower()) for m in matched_lower
                )
                color = GREEN if is_matched else TEXT2
                prefix = "+" if is_matched else " "
                ctk.CTkLabel(
                    r, text=f"  {prefix}  {ing_str}",
                    font=ctk.CTkFont(size=12), text_color=color, anchor="w",
                ).grid(row=0, column=0, sticky="w", padx=8)

        if instructions:
            ctk.CTkLabel(
                body, text="Przygotowanie",
                font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT,
            ).grid(row=row, column=0, padx=24, pady=(24, 8), sticky="w")
            row += 1
            box = ctk.CTkTextbox(
                body, font=ctk.CTkFont(size=12), fg_color=CARD, text_color=TEXT2,
                corner_radius=8, wrap="word", height=300,
            )
            box.grid(row=row, column=0, padx=24, pady=(0, 24), sticky="ew")
            box.insert("1.0", instructions)
            box.configure(state="disabled")

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
    def __init__(self):
        super().__init__()
        self.title("Recipe Voice Filter")
        self.geometry("1280x860")
        self.minsize(1040, 720)
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
        header = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Recipe Voice Filter",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, padx=22, pady=(14, 2), sticky="w")

        ctk.CTkLabel(
            header,
            text="Nagranie i transkrypcja po lewej — lista przepisow po prawej.",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            anchor="w",
        ).grid(row=1, column=0, padx=22, pady=(0, 14), sticky="w")

        self.status_card = ctk.CTkFrame(
            header,
            fg_color=ACCENT_SOFT,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        self.status_card.grid(row=0, column=1, rowspan=2, padx=22, pady=18, sticky="e")

        self.lbl_status = ctk.CTkLabel(
            self.status_card,
            text="Uruchamianie...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT2,
        )
        self.lbl_status.pack(padx=12, pady=8)

    def _build_workspace_shell(self):
        self._main_frame = ctk.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self._main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 16))
        self._main_frame.grid_rowconfigure(0, weight=1)
        self._main_frame.grid_columnconfigure(0, weight=1)

        self.workspace = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        self.workspace.grid(row=0, column=0, sticky="nsew")
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_columnconfigure(0, weight=0, minsize=292)
        self.workspace.grid_columnconfigure(1, weight=1)

        self.left_col = ctk.CTkFrame(self.workspace, fg_color="transparent", width=300)
        self.right_col = ctk.CTkFrame(self.workspace, fg_color="transparent")

        self.input_card = self._build_input_card(self.left_col)
        self.transcription_card = self._build_transcription_card(self.left_col)
        self.translation_card = self._build_translation_card(self.left_col)
        self.ingredients_card = self._build_ingredients_card(self.left_col)
        self.results_card = self._build_results_card(self.right_col)

        self.left_col.grid_columnconfigure(0, weight=1)
        self.input_card.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.transcription_card.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.translation_card.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.ingredients_card.grid(row=3, column=0, sticky="ew")

        self.right_col.grid_rowconfigure(0, weight=1)
        self.right_col.grid_columnconfigure(0, weight=1)
        self.results_card.grid(row=0, column=0, sticky="nsew")

        self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.right_col.grid(row=0, column=1, sticky="nsew")

        self._overlay = ctk.CTkFrame(self._main_frame, fg_color=BG, corner_radius=24)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        center = ctk.CTkFrame(
            self._overlay,
            fg_color=CARD,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        center.place(relx=0.5, rely=0.43, anchor="center")

        ctk.CTkLabel(
            center,
            text="Recipe Voice Filter",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
        ).pack(padx=34, pady=(24, 6))

        self.lbl_loading_step = ctk.CTkLabel(
            center,
            text="Inicjalizacja",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
        )
        self.lbl_loading_step.pack(pady=(0, 18))

        self.progress_bar = ctk.CTkProgressBar(
            center,
            width=250,
            height=6,
            mode="indeterminate",
            progress_color=ACCENT,
            fg_color=ACCENT_SOFT,
        )
        self.progress_bar.pack(padx=34, pady=(0, 24))
        self.progress_bar.start()

    def _build_input_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(
            card,
            text="Audio",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        ctk.CTkLabel(
            card,
            text="Mikrofon lub plik",
            font=ctk.CTkFont(size=10),
            text_color=MUTED,
            justify="left",
        ).grid(row=1, column=0, padx=12, pady=(0, 6), sticky="w")

        self.btn_record = ctk.CTkButton(
            card,
            text="Nagraj",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            corner_radius=16,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color="white",
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=2, column=0, padx=12, sticky="ew")

        self.btn_file = ctk.CTkButton(
            card,
            text="Wczytaj plik",
            font=ctk.CTkFont(size=10),
            height=30,
            corner_radius=14,
            fg_color=CARD,
            hover_color=CARD_H,
            text_color=TEXT2,
            border_width=1,
            border_color=BORDER,
            command=self._load_file,
        )
        self.btn_file.grid(row=3, column=0, padx=12, pady=(4, 8), sticky="ew")
        return card

    def _build_transcription_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(2, weight=1)

        head = ctk.CTkFrame(card, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            head,
            text="Transkrypcja",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")

        self.lbl_language = ctk.CTkLabel(
            head,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT2,
        )
        self.lbl_language.grid(row=0, column=1, sticky="e")

        ctk.CTkLabel(
            card,
            text="Rozpoznany tekst",
            font=ctk.CTkFont(size=10),
            text_color=MUTED,
        ).grid(row=1, column=0, padx=12, sticky="w")

        self.txt_transcription = ctk.CTkTextbox(
            card,
            height=76,
            font=ctk.CTkFont(size=11),
            fg_color=CARD,
            text_color=TEXT2,
            corner_radius=16,
            border_width=1,
            border_color=BORDER,
            wrap="word",
        )
        self.txt_transcription.grid(row=2, column=0, padx=12, pady=(2, 8), sticky="ew")
        return card

    def _build_translation_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text="Tlumaczenie (offline)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        controls = ctk.CTkFrame(card, fg_color="transparent")
        controls.grid(row=1, column=0, sticky="ew", padx=12)
        controls.grid_columnconfigure(0, weight=1)

        self.combo_translate = ctk.CTkOptionMenu(
            controls,
            values=[name for code, name in TRANSLATION_LABELS.items() if code != "pl"],
            font=ctk.CTkFont(size=11),
            fg_color=ACCENT,
            button_color=ACCENT_H,
            button_hover_color=ACCENT_H,
            dropdown_fg_color=CARD,
            dropdown_hover_color=CARD_H,
            text_color="white",
            dynamic_resizing=False,
        )
        self.combo_translate.grid(row=0, column=0, sticky="ew")

        self.btn_translate = ctk.CTkButton(
            controls,
            text="Tlumacz",
            font=ctk.CTkFont(size=10, weight="bold"),
            height=30,
            width=84,
            corner_radius=14,
            fg_color=TEXT,
            hover_color=TEXT2,
            text_color="white",
            command=self._translate,
        )
        self.btn_translate.grid(row=0, column=1, padx=(8, 6))

        self.btn_copy = ctk.CTkButton(
            controls,
            text="Kopiuj",
            font=ctk.CTkFont(size=9),
            height=30,
            width=64,
            corner_radius=14,
            fg_color=CARD,
            hover_color=CARD_H,
            text_color=TEXT2,
            border_width=1,
            border_color=BORDER,
            command=self._copy_translation,
        )
        self.btn_copy.grid(row=0, column=2)

        self.txt_translation = ctk.CTkTextbox(
            card,
            height=58,
            font=ctk.CTkFont(size=11),
            fg_color=CARD,
            text_color=MUTED,
            corner_radius=16,
            border_width=1,
            border_color=BORDER,
            wrap="word",
        )
        self.txt_translation.grid(row=2, column=0, padx=12, pady=(6, 4), sticky="ew")
        self.txt_translation.insert("1.0", "Wynik tlumaczenia pojawi sie tutaj.")
        self.txt_translation.configure(state="disabled")

        self.lbl_translation_state = ctk.CTkLabel(
            card,
            text="Sprawdzanie pakietow offline...",
            font=ctk.CTkFont(size=9),
            text_color=GREEN,
            anchor="w",
            justify="left",
        )
        self.lbl_translation_state.grid(row=3, column=0, padx=12, pady=(0, 8), sticky="ew")
        return card

    def _build_ingredients_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card,
            text="Skladniki (do wyszukiwania)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="w")

        self.ingredients_frame = ctk.CTkScrollableFrame(
            card,
            height=88,
            fg_color=CARD,
            corner_radius=16,
            border_width=1,
            border_color=BORDER,
            scrollbar_button_color=CARD_H,
            scrollbar_button_hover_color=ACCENT_SOFT,
        )
        self.ingredients_frame.grid(row=1, column=0, padx=12, sticky="nsew")

        self.lbl_filter_info = ctk.CTkLabel(
            card,
            text="",
            font=ctk.CTkFont(size=9),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=250,
        )
        self.lbl_filter_info.grid(row=2, column=0, padx=12, pady=(4, 8), sticky="ew")
        return card

    def _build_results_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=PANEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDER,
        )
        card.grid_rowconfigure(2, weight=1)
        card.grid_columnconfigure(0, weight=1)

        self.lbl_results_count = ctk.CTkLabel(
            card,
            text="Przepisy",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT,
            anchor="w",
        )
        self.lbl_results_count.grid(row=0, column=0, padx=18, pady=(14, 4), sticky="w")

        self.lbl_results_sub = ctk.CTkLabel(
            card,
            text="Wyniki pojawia sie po transkrypcji.",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            anchor="w",
        )
        self.lbl_results_sub.grid(row=1, column=0, padx=18, sticky="w")

        self.results_scroll = ctk.CTkScrollableFrame(
            card,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=CARD_H,
            scrollbar_button_hover_color=ACCENT_SOFT,
        )
        self.results_scroll.grid(row=2, column=0, sticky="nsew", padx=18, pady=(12, 16))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        self._empty_label = ctk.CTkLabel(
            self.results_scroll,
            text="Brak wynikow.\nDodaj nagranie, a tutaj pojawia sie dopasowane przepisy.",
            font=ctk.CTkFont(size=14),
            text_color=DIM,
            justify="center",
        )
        self._empty_label.grid(row=0, column=0, pady=110)
        return card

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
            self.workspace.grid_columnconfigure(0, weight=0, minsize=292)
            self.workspace.grid_columnconfigure(1, weight=1)
            self.workspace.grid_rowconfigure(0, weight=1)
            self.left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
            self.right_col.grid(row=0, column=1, sticky="nsew")
            self.lbl_filter_info.configure(wraplength=280)
            return

        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=0)
        self.right_col.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self.left_col.grid(row=1, column=0, sticky="ew")
        self.lbl_filter_info.configure(wraplength=560)

    def _build_sidebar(self):
        side = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=24,
            border_width=1,
            border_color=BORDER,
            width=368,
        )
        side.grid(row=0, column=0, sticky="nsew", padx=(20, 12), pady=20)
        side.grid_propagate(False)
        side.grid_columnconfigure(0, weight=1)
        side.grid_rowconfigure(2, weight=1)

        hero = ctk.CTkFrame(side, fg_color="transparent")
        hero.grid(row=0, column=0, padx=24, pady=(22, 14), sticky="ew")
        hero.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hero,
            text="Recipe Voice Filter",
            font=ctk.CTkFont(size=23, weight="bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            hero,
            text="Filtrowanie przepisow z nagrania lub pliku audio.",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(6, 12))

        self.status_card = ctk.CTkFrame(
            hero,
            fg_color=ACCENT_SOFT,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        self.status_card.grid(row=2, column=0, sticky="w")

        self.lbl_status = ctk.CTkLabel(
            self.status_card,
            text="Uruchamianie...",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT2,
        )
        self.lbl_status.pack(padx=12, pady=7)

        capture = ctk.CTkFrame(
            side,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER,
        )
        capture.grid(row=1, column=0, padx=24, pady=(0, 16), sticky="ew")
        capture.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            capture,
            text="Wyszukiwanie glosowe",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(16, 4), sticky="w")

        ctk.CTkLabel(
            capture,
            text="Podaj skladniki glosem albo wybierz gotowy plik audio.",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=1, column=0, padx=16, pady=(0, 14), sticky="w")

        self.btn_record = ctk.CTkButton(
            capture,
            text="Nagraj",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            corner_radius=14,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color=BG,
            command=self._toggle_recording,
        )
        self.btn_record.grid(row=2, column=0, padx=16, sticky="ew")

        self.btn_file = ctk.CTkButton(
            capture,
            text="Wczytaj plik audio",
            font=ctk.CTkFont(size=12),
            height=38,
            corner_radius=12,
            fg_color="transparent",
            hover_color=CARD_H,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT2,
            command=self._load_file,
        )
        self.btn_file.grid(row=3, column=0, padx=16, pady=(8, 16), sticky="ew")

        self.sidebar_tabs = ctk.CTkTabview(
            side,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER,
            segmented_button_fg_color=BG,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_H,
            segmented_button_unselected_color=TAG_BG,
            segmented_button_unselected_hover_color=CARD_H,
            text_color=TEXT,
            anchor="n",
        )
        self.sidebar_tabs.grid(row=2, column=0, padx=24, pady=(0, 24), sticky="nsew")

        trans_tab = self.sidebar_tabs.add("Transkrypcja")
        ing_tab = self.sidebar_tabs.add("Skladniki")
        tr_tab = self.sidebar_tabs.add("Tlumaczenie")
        for tab in (trans_tab, ing_tab, tr_tab):
            tab.grid_columnconfigure(0, weight=1)

        trans_head = ctk.CTkFrame(trans_tab, fg_color="transparent")
        trans_head.grid(row=0, column=0, sticky="ew", pady=(6, 8))
        trans_head.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            trans_head,
            text="Tekst rozpoznany przez model mowy",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.lbl_language = ctk.CTkLabel(
            trans_head,
            text="",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT2,
        )
        self.lbl_language.grid(row=0, column=1, sticky="e")

        self.txt_transcription = ctk.CTkTextbox(
            trans_tab,
            height=190,
            font=ctk.CTkFont(size=12),
            fg_color=BG,
            text_color=TEXT2,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            wrap="word",
        )
        self.txt_transcription.grid(row=1, column=0, sticky="nsew")

        ctk.CTkLabel(
            ing_tab,
            text="Wykryte skladniki z nagrania",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(6, 8))

        self.ingredients_frame = ctk.CTkScrollableFrame(
            ing_tab,
            height=260,
            fg_color=BG,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            scrollbar_button_color=CARD_H,
            scrollbar_button_hover_color=ACCENT_SOFT,
        )
        self.ingredients_frame.grid(row=1, column=0, sticky="nsew")

        self.lbl_filter_info = ctk.CTkLabel(
            ing_tab,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=MUTED,
            justify="left",
            anchor="w",
            wraplength=260,
        )
        self.lbl_filter_info.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        ctk.CTkLabel(
            tr_tab,
            text="Przetlumacz rozpoznany tekst na inny jezyk",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(6, 8))

        tr_controls = ctk.CTkFrame(tr_tab, fg_color="transparent")
        tr_controls.grid(row=1, column=0, sticky="ew")
        tr_controls.grid_columnconfigure(0, weight=1)

        self.combo_translate = ctk.CTkOptionMenu(
            tr_controls,
            values=["angielski", "niemiecki", "francuski", "hiszpanski"],
            font=ctk.CTkFont(size=11),
            fg_color=TAG_BG,
            button_color=CARD_H,
            button_hover_color=ACCENT_SOFT,
            dropdown_fg_color=PANEL,
            dropdown_hover_color=CARD_H,
            text_color=TEXT,
            dynamic_resizing=False,
        )
        self.combo_translate.grid(row=0, column=0, sticky="ew")

        self.btn_translate = ctk.CTkButton(
            tr_controls,
            text="Tlumacz",
            font=ctk.CTkFont(size=11, weight="bold"),
            height=34,
            width=84,
            corner_radius=12,
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            text_color=BG,
            command=self._translate,
        )
        self.btn_translate.grid(row=0, column=1, padx=(8, 6))

        self.btn_copy = ctk.CTkButton(
            tr_controls,
            text="Kopiuj",
            font=ctk.CTkFont(size=10),
            height=34,
            width=70,
            corner_radius=12,
            fg_color="transparent",
            hover_color=CARD_H,
            text_color=TEXT2,
            border_width=1,
            border_color=BORDER,
            command=self._copy_translation,
        )
        self.btn_copy.grid(row=0, column=2)

        self.txt_translation = ctk.CTkTextbox(
            tr_tab,
            height=150,
            font=ctk.CTkFont(size=11),
            fg_color=BG,
            text_color=MUTED,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
            wrap="word",
        )
        self.txt_translation.grid(row=2, column=0, pady=(10, 8), sticky="ew")
        self.txt_translation.insert("1.0", "Transkrypcja pojawi sie tutaj po tlumaczeniu...")
        self.txt_translation.configure(state="disabled")

        self.btn_install_pkg = ctk.CTkButton(
            tr_tab,
            text="Pobierz pakiety tlumaczen",
            font=ctk.CTkFont(size=10),
            height=30,
            corner_radius=12,
            fg_color="transparent",
            hover_color=CARD_H,
            text_color=AMBER,
            border_width=1,
            border_color=AMBER,
            command=self._install_translation_packages,
        )
        self.btn_install_pkg.grid(row=3, column=0, sticky="ew")
        self.btn_install_pkg.grid_remove()

    def _build_main(self):
        self._main_frame = ctk.CTkFrame(
            self,
            fg_color=PANEL,
            corner_radius=24,
            border_width=1,
            border_color=BORDER,
        )
        self._main_frame.grid(row=0, column=1, sticky="nsew", padx=(12, 20), pady=20)
        self._main_frame.grid_rowconfigure(1, weight=1)
        self._main_frame.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(
            self._main_frame,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER,
        )
        hdr.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 0))
        hdr.grid_columnconfigure(0, weight=1)

        self.lbl_results_count = ctk.CTkLabel(
            hdr,
            text="Przepisy",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT,
            anchor="w",
        )
        self.lbl_results_count.grid(row=0, column=0, sticky="w", padx=20, pady=(18, 2))

        self.lbl_results_sub = ctk.CTkLabel(
            hdr,
            text="Nagraj glos lub wczytaj plik, aby wyszukac przepisy",
            font=ctk.CTkFont(size=12),
            text_color=MUTED,
            anchor="w",
        )
        self.lbl_results_sub.grid(row=1, column=0, sticky="w", padx=20)

        ctk.CTkLabel(
            hdr,
            text="Skladniki do wyszukania",
            font=ctk.CTkFont(size=11),
            text_color=MUTED,
            anchor="w",
        ).grid(row=2, column=0, sticky="w", padx=20, pady=(16, 6))

        self.query_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        self.query_frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        ctk.CTkLabel(
            self.query_frame,
            text="Czekam na skladniki z nagrania...",
            font=ctk.CTkFont(size=11),
            text_color=DIM,
            anchor="w",
        ).pack(anchor="w", padx=4, pady=2)

        self.results_scroll = ctk.CTkScrollableFrame(
            self._main_frame,
            fg_color="transparent",
            corner_radius=0,
            scrollbar_button_color=CARD_H,
            scrollbar_button_hover_color=ACCENT_SOFT,
        )
        self.results_scroll.grid(row=1, column=0, sticky="nsew", padx=28, pady=(16, 24))
        self.results_scroll.grid_columnconfigure(0, weight=1)

        self._empty_label = ctk.CTkLabel(
            self.results_scroll,
            text="Brak wynikow.\nDodaj nagranie, a tutaj pojawia sie dopasowane przepisy.",
            font=ctk.CTkFont(size=14),
            text_color=DIM,
            justify="center",
        )
        self._empty_label.grid(row=0, column=0, pady=100)

        self._overlay = ctk.CTkFrame(self._main_frame, fg_color=BG, corner_radius=0)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        center = ctk.CTkFrame(
            self._overlay,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER,
        )
        center.place(relx=0.5, rely=0.42, anchor="center")

        ctk.CTkLabel(
            center,
            text="Recipe Voice Filter",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT,
        ).pack(padx=30, pady=(24, 6))

        self.lbl_loading_step = ctk.CTkLabel(
            center,
            text="Inicjalizacja",
            font=ctk.CTkFont(size=12), text_color=MUTED,
        )
        self.lbl_loading_step.pack(pady=(0, 16))

        self.progress_bar = ctk.CTkProgressBar(
            center,
            width=240,
            height=5,
            mode="indeterminate",
            progress_color=ACCENT, fg_color=CARD,
        )
        self.progress_bar.pack(padx=30, pady=(0, 24))
        self.progress_bar.start()

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
        self._set_status("Przetwarzanie mowy...", TEXT2)
        self._show_processing_overlay("Przetwarzanie nagrania")
        threading.Thread(
            target=lambda: self._process_audio(self.recorder.stop()), daemon=True
        ).start()

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
            self.current_language = (str(lang).lower().strip() if lang else "pl")
            self.after(0, lambda: self._update_transcription(text, lang_name))

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
            self.after(
                0,
                lambda: self._update_ingredients(
                    detected_ingredients, ignored_ingredients
                ),
            )

            results = self._search_recipes(ingredients)
            self.current_results = results
            self.after(0, lambda: self._update_results(results))
            self._set_status(f"Znaleziono {len(results)} przepisow", GREEN)

        except Exception as exc:
            self._set_status(f"Blad: {exc}", RED)
        finally:
            self._hide_processing_overlay()

    def _deduplicate_ingredients(self, ingredients: list) -> list:
        result = []
        for ing in ingredients:
            if not any(tokens_match(ing, existing) for existing in result):
                result.append(ing)
        return result

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
        results = self.matcher.search(ingredients)
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

    def _update_transcription(self, text, lang_name):
        self.txt_transcription.configure(state="normal")
        self.txt_transcription.delete("1.0", "end")
        self.txt_transcription.insert("1.0", text)
        self.lbl_language.configure(text=lang_name)
        self._refresh_translation_support()

    def _update_ingredients(self, ingredients, ignored_ingredients=None):
        for w in self.ingredients_frame.winfo_children():
            w.destroy()
        ignored_ingredients = ignored_ingredients or []
        if not ingredients:
            ctk.CTkLabel(
                self.ingredients_frame,
                text="Brak wykrytych skladnikow",
                font=ctk.CTkFont(size=11), text_color=MUTED,
            ).pack(anchor="w", padx=8)
            self.lbl_results_sub.configure(
                text="Nagraj skladniki — tutaj pojawi sie dopasowanie.",
            )
            self.lbl_filter_info.configure(text="")
            return
        _flow_layout_chips(
            self.ingredients_frame,
            ingredients,
            ACCENT_SOFT,
            TEXT,
            font_size=11,
            max_columns=3,
        )
        used = self.current_ingredients or ingredients
        summary = ", ".join(used[:6])
        if len(used) > 6:
            summary += f" (+{len(used) - 6})"
        self.lbl_results_sub.configure(text=f"Wyszukiwanie dla: {summary}")
        if ignored_ingredients:
            ignored_text = ", ".join(ignored_ingredients)
            self.lbl_filter_info.configure(
                text=f"Pominieto skladniki bazowe: {ignored_text}"
            )
        else:
            self.lbl_filter_info.configure(text="")

    def _update_results(self, results):
        for w in self.results_scroll.winfo_children():
            w.destroy()

        if not results:
            self.lbl_results_count.configure(text="Przepisy")
            self.lbl_results_sub.configure(text="Nie znaleziono pasujacych przepisow")
            ctk.CTkLabel(
                self.results_scroll,
                text="Brak wynikow dla podanych skladnikow.\nSprobuj krotsze lub bardziej konkretne nagranie.",
                font=ctk.CTkFont(size=13),
                text_color=DIM,
                justify="center",
            ).grid(row=0, column=0, pady=80)
            return

        total_asked = len(self.current_ingredients)
        self.lbl_results_count.configure(text=f"{len(results)} przepisow")
        if total_asked:
            self.lbl_results_sub.configure(text=f"Posortowane wedlug trafnosci dla {total_asked} skladnikow")
        else:
            self.lbl_results_sub.configure(text="Posortowane wedlug trafnosci")
        for display_idx, recipe in enumerate(results):
            recipe_idx = recipe.get("_idx", -1)
            matched_count = self.matcher.match_count(self.current_ingredients, recipe_idx)
            self._add_card(display_idx, recipe, matched_count, total_asked)

    def _add_card(self, display_idx, recipe, matched_count, total_asked):
        ings = self.matcher.preview_ingredients(recipe)
        ing_count = len(ings) if isinstance(ings, list) else 0

        card = ctk.CTkFrame(
            self.results_scroll,
            fg_color=CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER,
        )
        card.grid(row=display_idx, column=0, sticky="ew", pady=(0, 6))
        card.grid_columnconfigure(0, weight=1)

        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.grid(row=0, column=0, sticky="ew", padx=12, pady=8)
        inner.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(inner, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row,
            text=recipe.get("title", "Bez nazwy"),
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        if total_asked > 0:
            score_text = f"{matched_count}/{total_asked}"
            if matched_count == total_asked:
                score_color = GREEN
                score_bg = GREEN_SOFT
            elif matched_count > 0:
                score_color = AMBER
                score_bg = AMBER_SOFT
            else:
                score_color = DIM
                score_bg = TAG_BG
        else:
            score_text, score_color, score_bg = "", DIM, TAG_BG

        if score_text:
            score_badge = ctk.CTkFrame(
                title_row,
                fg_color=score_bg,
                corner_radius=10,
                border_width=1,
                border_color=BORDER,
            )
            score_badge.grid(row=0, column=1, sticky="e", padx=(8, 0))
            ctk.CTkLabel(
                score_badge,
                text=score_text,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=score_color,
            ).pack(padx=8, pady=3)

        cat = (recipe.get("category") or "").strip()
        meta = f"{cat} · {ing_count} skl." if cat else f"{ing_count} sklad."
        ctk.CTkLabel(
            inner,
            text=meta,
            font=ctk.CTkFont(size=10),
            text_color=MUTED,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 6))

        if isinstance(ings, list) and ings:
            chip_row = ctk.CTkFrame(inner, fg_color="transparent")
            chip_row.grid(row=2, column=0, sticky="w")
            for ing_text in ings[:3]:
                chip = ctk.CTkFrame(
                    chip_row,
                    fg_color=TAG_BG,
                    corner_radius=8,
                    border_width=1,
                    border_color=BORDER,
                )
                chip.pack(side="left", padx=(0, 4), pady=1)
                ctk.CTkLabel(
                    chip,
                    text=str(ing_text)[:28] + ("…" if len(str(ing_text)) > 28 else ""),
                    font=ctk.CTkFont(size=9),
                    text_color=TEXT2,
                ).pack(padx=6, pady=3)
            if ing_count > 3:
                ctk.CTkLabel(
                    chip_row,
                    text=f"+{ing_count-3}",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    text_color=DIM,
                ).pack(side="left", padx=2, pady=2)

        def on_enter(_e, c=card):
            c.configure(fg_color=CARD_H, border_color=ACCENT)

        def on_leave(_e, c=card):
            c.configure(fg_color=CARD, border_color=BORDER)

        def on_click(_e, r=recipe):
            self._show_detail(r)

        def bind_rec(w):
            w.bind("<Enter>", on_enter)
            w.bind("<Leave>", on_leave)
            w.bind("<Button-1>", on_click)
            for ch in w.winfo_children():
                bind_rec(ch)

        try:
            card.configure(cursor="hand2")
        except Exception:
            pass
        bind_rec(card)

    def _show_detail(self, recipe):
        recipe_idx = recipe.get("_idx", -1)
        now = time.monotonic()
        last = getattr(self, "_last_detail_open", None)
        if last and last[0] == recipe_idx and (now - last[1]) < 0.45:
            return
        self._last_detail_open = (recipe_idx, now)
        matched = self.matcher.matched_ingredients(self.current_ingredients, recipe_idx)
        total_asked = len(self.current_ingredients)
        DetailWindow(self, recipe, matched, total_asked)

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
        text = self.txt_transcription.get("1.0", "end").strip()
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
