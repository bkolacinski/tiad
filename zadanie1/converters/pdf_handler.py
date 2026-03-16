"""Konwersja danych z arkuszy Excel do formatu PDF."""

import os
from html import escape
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape as make_landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Mapowanie wyrownan
ALIGN_MAP = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}

# Mapowanie wyrownan z Excela
EXCEL_ALIGN_MAP = {
    "left": "left", "general": "left",
    "center": "center", "centre": "center",
    "right": "right", "justify": "left",
}

# Stale
GRID_COLOR = colors.Color(0.82, 0.82, 0.82)
MARGIN = 1.5 * cm
MIN_COL_WIDTH_PT = 5.0
EXCEL_CHAR_TO_PT = 7.0

# Domyslne czcionki (nadpisywane przez _register_fonts)
_FONT = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"


def _register_fonts() -> None:
    """Rejestruje czcionki z systemu obslugujace polskie znaki."""
    global _FONT, _FONT_BOLD
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "ArialPL", "ArialPL-Bold"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "DejaVuSans", "DejaVuSans-Bold"),
    ]
    for regular, bold, rname, bname in candidates:
        if not (os.path.exists(regular) and os.path.exists(bold)):
            continue
        try:
            pdfmetrics.registerFont(TTFont(rname, regular))
            pdfmetrics.registerFont(TTFont(bname, bold))
            _FONT, _FONT_BOLD = rname, bname
            return
        except Exception:
            continue


_register_fonts()


# ---------------------------------------------------------------------------
# Funkcje pomocnicze
# ---------------------------------------------------------------------------


def _parse_color(value: Any) -> colors.Color | None:
    """Konwertuje kolor hex na obiekt Color reportlab."""
    if isinstance(value, colors.Color):
        return value
    if not isinstance(value, str):
        return None
    raw = value.strip().lstrip("#")
    if len(raw) == 8:
        raw = raw[2:]
    if len(raw) != 6:
        return None
    try:
        return colors.Color(
            int(raw[0:2], 16) / 255,
            int(raw[2:4], 16) / 255,
            int(raw[4:6], 16) / 255,
        )
    except ValueError:
        return None


def _get_alignment(h_align: Any, fallback: str) -> str:
    """Mapuje wyrownanie z Excela na nazwe zrozumiala dla PDF."""
    if isinstance(h_align, str):
        return EXCEL_ALIGN_MAP.get(h_align.lower().strip(), fallback)
    return fallback


def _get_cell_meta(cells: list, row: int, col: int) -> dict:
    """Zwraca metadane komorki lub pusty slownik."""
    try:
        return cells[row][col]
    except (IndexError, TypeError):
        return {}


def _col_widths_pts(excel_widths: list, n_cols: int, available_pt: float) -> list[float]:
    """Przelicza szerokosci kolumn z Excela na punkty, skalujac do szerokosci strony."""
    widths = list(excel_widths[:n_cols])
    while len(widths) < n_cols:
        widths.append(8.43)

    natural = [w * EXCEL_CHAR_TO_PT for w in widths]
    total = sum(natural) or 1.0
    factor = available_pt / total
    scaled = [max(w * factor, MIN_COL_WIDTH_PT) for w in natural]

    # Korekta jesli minima spowodowaly przekroczenie
    total_scaled = sum(scaled)
    if total_scaled > available_pt:
        adjustable = [(i, w) for i, w in enumerate(scaled) if w > MIN_COL_WIDTH_PT]
        if adjustable:
            excess = total_scaled - available_pt
            per_col = excess / len(adjustable)
            for i, w in adjustable:
                scaled[i] = max(w - per_col, MIN_COL_WIDTH_PT)

    return scaled


def _get_font_scale(max_cols: int) -> float:
    """Dobiera skale czcionki w zaleznosci od liczby kolumn."""
    if max_cols <= 10:
        return 1.0
    if max_cols <= 15:
        return 0.85
    if max_cols <= 25:
        return 0.72
    if max_cols <= 50:
        return 0.6
    return 0.5


def _get_padding(max_cols: int) -> float:
    """Dobiera padding komorek w zaleznosci od liczby kolumn."""
    if max_cols <= 20:
        return 4.0
    if max_cols <= 30:
        return 3.0
    if max_cols <= 60:
        return 2.0
    return 1.0


def _get_page_size(max_cols: int) -> tuple:
    """Dobiera rozmiar strony w zaleznosci od liczby kolumn."""
    if max_cols > 20:
        page_w = max(841.89, max_cols * 30.0)
        page_h = max(595.27, page_w * 0.707)
        return (page_w, page_h)
    if max_cols > 6:
        return make_landscape(A4)
    return A4


def _border_weight(style_name: str) -> float:
    """Zwraca grubosc obramowania dla danego stylu z Excela."""
    return {
        "thin": 0.5, "medium": 1.0, "thick": 1.5,
        "hair": 0.25, "double": 1.0, "dotted": 0.5,
        "dashed": 0.5, "mediumDashed": 1.0,
    }.get(style_name, 0.5)


def _page_number_callback(canvas, doc) -> None:
    """Rysuje numer strony na srodku stopki."""
    canvas.saveState()
    canvas.setFont(_FONT, 8)
    canvas.setFillColor(colors.Color(0.5, 0.5, 0.5))
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.7 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Budowanie tabeli
# ---------------------------------------------------------------------------


def _build_table_data(df: pd.DataFrame, cells: list, merges: list,
                      font_scale: float, global_align: str,
                      padding: float) -> tuple[list, list]:
    """
    Buduje dane i style tabeli dla reportlab.

    Returns:
        (data, style_cmds) - dane wierszy i lista polecen stylu
    """
    n_rows, n_cols = df.shape

    # Bazowe style tabeli
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.25, GRID_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Scalenia komorek
    for merge in merges:
        if merge.get("is_anchor"):
            style_cmds.append((
                "SPAN",
                (merge["start_col"], merge["start_row"]),
                (merge["end_col"], merge["end_row"]),
            ))

    # Budowanie wierszy danych
    data = []
    for row_i in range(n_rows):
        row_data = []
        for col_j in range(n_cols):
            meta = _get_cell_meta(cells, row_i, col_j)

            if meta.get("hidden_by_merge"):
                row_data.append("")
                continue

            # Wartosc komorki
            value = meta.get("value") if meta.get("value") is not None else str(df.iat[row_i, col_j])
            text = escape(str(value)).replace("\n", "<br/>")

            # Styl czcionki
            style = meta.get("style", {})
            font = style.get("font") or {}
            fg = _parse_color(font.get("color")) or colors.black
            font_size = max(float(font.get("size") or 11.0) * font_scale, 5.0)
            bold = bool(font.get("bold"))

            h_align = (style.get("alignment") or {}).get("horizontal")
            align = _get_alignment(h_align, global_align)

            para_style = ParagraphStyle(
                name=f"c_{row_i}_{col_j}",
                fontName=_FONT_BOLD if bold else _FONT,
                fontSize=font_size,
                leading=font_size * 1.25,
                alignment=ALIGN_MAP.get(align, TA_LEFT),
                textColor=fg,
            )
            row_data.append(Paragraph(text, para_style))

            # Kolor tla
            fill_color = _parse_color((style.get("fill") or {}).get("color"))
            if fill_color:
                style_cmds.append(("BACKGROUND", (col_j, row_i), (col_j, row_i), fill_color))

            # Obramowania
            border_info = style.get("border") or {}
            border_map = {
                "top": "LINEABOVE", "bottom": "LINEBELOW",
                "left": "LINEBEFORE", "right": "LINEAFTER",
            }
            for side, cmd in border_map.items():
                bdata = border_info.get(side)
                if bdata:
                    bc = _parse_color(bdata.get("color")) or colors.black
                    weight = _border_weight(bdata.get("style", "thin"))
                    style_cmds.append((cmd, (col_j, row_i), (col_j, row_i), weight, bc))

        data.append(row_data)

    return data, style_cmds


# ---------------------------------------------------------------------------
# Funkcja glowna
# ---------------------------------------------------------------------------


def df_to_pdf(sheets: dict, settings: dict, output) -> None:
    """
    Konwertuje arkusze Excel na dokument PDF.

    Args:
        sheets: slownik {nazwa_arkusza: dane} z read_xlsx()
        settings: ustawienia formatowania (alignment, page_numbers, itp.)
        output: bufor wyjsciowy (BytesIO)
    """
    alignment = settings.get("alignment", "left")
    page_numbers = bool(settings.get("page_numbers", True))
    title = (settings.get("title") or "").strip()

    # Przygotowanie danych arkuszy
    prepared = []
    max_cols = 1
    for name, payload in sheets.items():
        df = payload.get("dataframe")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.fillna("")
        max_cols = max(max_cols, df.shape[1])
        prepared.append((
            name, df,
            payload.get("excel_col_widths", []),
            payload.get("cells", []),
            payload.get("merges", []),
        ))

    if not prepared:
        return

    # Konfiguracja strony
    page_size = _get_page_size(max_cols)
    font_scale = _get_font_scale(max_cols)
    padding = _get_padding(max_cols)
    available_pt = page_size[0] - 2 * MARGIN

    doc = SimpleDocTemplate(
        output, pagesize=page_size,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm if page_numbers else MARGIN,
    )

    # Budowanie zawartosci dokumentu
    story = []

    if title:
        story.append(Paragraph(
            escape(title),
            ParagraphStyle("title", fontName=_FONT_BOLD, fontSize=14,
                           leading=18, alignment=TA_CENTER, spaceAfter=10),
        ))
        story.append(Spacer(1, 0.4 * cm))

    for name, df, excel_widths, cells, merges in prepared:
        # Naglowek arkusza (jesli wiele arkuszy)
        if len(prepared) > 1:
            story.append(Paragraph(
                escape(name),
                ParagraphStyle("sheet", fontName=_FONT_BOLD, fontSize=11,
                               leading=14, textColor=colors.Color(0.22, 0.45, 0.70),
                               spaceBefore=12, spaceAfter=4),
            ))

        col_widths = _col_widths_pts(excel_widths, df.shape[1], available_pt)
        table_data, style_cmds = _build_table_data(
            df, cells, merges, font_scale, alignment, padding,
        )

        if not table_data:
            continue

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1, splitInRow=1)
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

    on_page = _page_number_callback if page_numbers else None
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
