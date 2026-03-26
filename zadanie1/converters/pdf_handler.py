"""Konwersja danych z arkuszy Excel do formatu PDF."""

import os
from html import escape
from typing import Any

import pandas as pd
from openpyxl.utils.cell import range_boundaries
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.pagesizes import landscape as make_landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Alignment mapping
ALIGN_MAP = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}

# Excel alignments mapping to match Reportlab's understanding
EXCEL_ALIGN_MAP = {
    "left": "left",
    "general": "left",
    "center": "center",
    "centre": "center",
    "right": "right",
    "justify": "left",
}

# Style
GRID_COLOR = colors.Color(0.82, 0.82, 0.82)
MARGIN = 1.5 * cm
DEFAULT_EXCEL_WIDTH = 8.43
EXCEL_CHAR_TO_PT = 7.0
LABEL_COLUMN_MIN_WIDTH_PT = 36.0

# Default fonts
_FONT = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONT_ITALIC = "Helvetica-Oblique"
_FONT_BOLD_ITALIC = "Helvetica-BoldOblique"


def _register_fonts() -> None:
    global _FONT, _FONT_BOLD, _FONT_ITALIC, _FONT_BOLD_ITALIC
    candidates = [
        {
            "regular": "C:/Windows/Fonts/arial.ttf",
            "bold": "C:/Windows/Fonts/arialbd.ttf",
            "italic": "C:/Windows/Fonts/ariali.ttf",
            "bold_italic": "C:/Windows/Fonts/arialbi.ttf",
            "family": "ArialPL",
        },
        {
            "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
            "bold_italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
            "family": "DejaVuSans",
        },
    ]
    for c in candidates:
        paths = [c["regular"], c["bold"], c["italic"], c["bold_italic"]]
        if not all(os.path.exists(p) for p in paths):
            continue
        try:
            fam = c["family"]
            names = {
                "regular": fam,
                "bold": f"{fam}-Bold",
                "italic": f"{fam}-Italic",
                "bold_italic": f"{fam}-BoldItalic",
            }
            pdfmetrics.registerFont(TTFont(names["regular"], c["regular"]))
            pdfmetrics.registerFont(TTFont(names["bold"], c["bold"]))
            pdfmetrics.registerFont(TTFont(names["italic"], c["italic"]))
            pdfmetrics.registerFont(TTFont(names["bold_italic"], c["bold_italic"]))
            pdfmetrics.registerFontFamily(
                fam,
                normal=names["regular"],
                bold=names["bold"],
                italic=names["italic"],
                boldItalic=names["bold_italic"],
            )
            _FONT = names["regular"]
            _FONT_BOLD = names["bold"]
            _FONT_ITALIC = names["italic"]
            _FONT_BOLD_ITALIC = names["bold_italic"]
            return
        except Exception:
            continue


_register_fonts()


def _parse_color(value: str | colors.Color) -> colors.Color | None:
    """
    Converts hex color to Reportlab Color object.
    :param value: Hex color in formats like "#RRGGBB", "#AARRGGBB",
    or a reportlab Color object.
    :return: Reportlab Color object or None if parsing fails.
    """
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
    """
    Maps excel alignment to PDF
    :param h_align: Excel horizontal alignment value
    :param fallback: Fallback alignment value
    :return: PDF alignment value
    """
    if isinstance(h_align, str):
        return EXCEL_ALIGN_MAP.get(h_align.lower().strip(), fallback)
    return fallback


def _get_cell_meta(cells: list, row: int, col: int) -> dict:
    """
    Gets cell metadata
    :param cells: List of cells
    :param row: Row of desired cell
    :param col: Column of desired cell
    :return: Metadata of the cell or empty dict if not found
    """
    try:
        return cells[row][col]
    except (IndexError, TypeError):
        return {}


def _normalize_excel_widths(excel_widths: list, n_cols: int) -> list[float]:
    """
    Fills empty values in Excel widths with default width
    :param excel_widths: List of Excel widths
    :param n_cols: Number of columns
    :return: List of all Excel column widths, with missing values filled
    with default width.
    """
    widths = list(excel_widths[:n_cols])
    while len(widths) < n_cols:
        widths.append(DEFAULT_EXCEL_WIDTH)
    return widths


def _get_label_columns(cells: list, merges: list, n_cols: int) -> set[int]:
    """
    Detects columns that contain single-column, multi-row labels
    :param cells: List of cells
    :param merges: List of merges
    :param n_cols: Number of columns
    :return: Set of columns that contain single-column
    """
    label_cols: set[int] = set()
    for merge in merges:
        start_col = merge.get("start_col")
        end_col = merge.get("end_col")
        start_row = merge.get("start_row")
        end_row = merge.get("end_row")
        if start_col != end_col or end_row <= start_row:
            continue
        if not isinstance(start_col, int) or not 0 <= start_col < n_cols:
            continue

        original_range = merge.get("range")
        if isinstance(original_range, str):
            try:
                min_col, _, max_col, _ = range_boundaries(original_range)
            except ValueError:
                pass
            else:
                if min_col != max_col:
                    continue

        value = _get_cell_meta(cells, start_row, start_col).get("value")
        if value is not None and str(value).strip():
            label_cols.add(start_col)

    return label_cols


def _col_widths_pts(
    excel_widths: list, n_cols: int, available_pt: float, cells: list, merges: list
) -> list[float]:
    """
    Converts excel widths into PDF widths
    :param excel_widths: List of Excel widths
    :param n_cols: Number of columns
    :param available_pt: Avil
    :param cells: List of cells
    :param merges: List of merges
    :return: List of PDF column widths
    """
    widths = _normalize_excel_widths(excel_widths, n_cols)

    natural = [w * EXCEL_CHAR_TO_PT for w in widths]
    total = sum(natural) or 1.0
    factor = available_pt / total
    scaled = [w * factor for w in natural]

    label_cols = _get_label_columns(cells, merges, n_cols)
    if not label_cols:
        return scaled

    reserved_widths = {
        idx: max(scaled[idx], LABEL_COLUMN_MIN_WIDTH_PT) for idx in sorted(label_cols)
    }
    reserved_total = sum(reserved_widths.values())
    if reserved_total >= available_pt:
        return scaled

    remaining_cols = [idx for idx in range(n_cols) if idx not in reserved_widths]
    remaining_natural = sum(natural[idx] for idx in remaining_cols)
    if remaining_natural <= 0:
        return scaled

    remaining_available = available_pt - reserved_total
    remaining_factor = remaining_available / remaining_natural
    constrained = []
    for idx in range(n_cols):
        if idx in reserved_widths:
            constrained.append(reserved_widths[idx])
        else:
            constrained.append(natural[idx] * remaining_factor)

    constrained[-1] += available_pt - sum(constrained)
    return constrained


def _get_font_scale(n_cols: int) -> float:
    """
    Chooses font scale depending on column count
    :param n_cols: Number of columns
    :return: Font scale, a number between 0.3 and 1.0
    """
    if n_cols <= 10:
        return 1.0
    if n_cols <= 15:
        return 0.85
    if n_cols <= 25:
        return 0.72
    if n_cols <= 50:
        return 0.55
    if n_cols <= 100:
        return 0.4
    return 0.3


def _get_padding(n_cols: int) -> float:
    """
    Chooses padding depending on column count
    :param n_cols: Number of columns
    :return: Padding value, a number between 0.5 and 4.0
    """
    if n_cols <= 20:
        return 4.0
    if n_cols <= 40:
        return 2.0
    if n_cols <= 80:
        return 1.0
    return 0.5


def _border_weight(style_name: str) -> float:
    """
    Chooses border weight depending on style
    :param style_name: Name of style
    :return: Border weight in points, a number between 0.25 and 1.5 (defaults to 0.5)
    """
    return {
        "thin": 0.5,
        "medium": 1.0,
        "thick": 1.5,
        "hair": 0.25,
        "double": 1.0,
        "dotted": 0.5,
        "dashed": 0.5,
        "mediumDashed": 1.0,
    }.get(style_name, 0.5)


def _page_number_callback(canvas, doc) -> None:
    """
    Prints page number at the bottom of the page
    :param canvas: Canvas
    :param doc: Document object
    :return: None
    """
    canvas.saveState()
    canvas.setFont(_FONT, 8)
    canvas.setFillColor(colors.Color(0.5, 0.5, 0.5))
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.7 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


def _build_table_data(
    df: pd.DataFrame,
    cells: list,
    merges: list,
    font_scale: float,
    global_align: str,
    padding: float,
) -> tuple[list, list]:
    """
    Creates table data for Reportlab based on DataFrame, cell metadata and merges.
    :param df: DataFrame with cell contents
    :param cells: List of cell
    :param merges: List of merges
    :param font_scale: Font scale
    :param global_align: Global alignment
    :param padding: Padding for cells
    :return: Data of rows and style commands
    """
    n_rows, n_cols = df.shape

    # Base styles
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.25, GRID_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Merged cells
    for merge in merges:
        if merge.get("is_anchor"):
            style_cmds.append(
                (
                    "SPAN",
                    (int(merge["start_col"]), int(merge["start_row"])),
                    (int(merge["end_col"]), int(merge["end_row"])),
                )
            )

    data = []
    for row_i in range(n_rows):
        row_data = []
        for col_j in range(n_cols):
            meta = _get_cell_meta(cells, row_i, col_j)

            if meta.get("hidden_by_merge"):
                row_data.append("")
                continue

            # Cell value
            value = (
                meta.get("value")
                if meta.get("value") is not None
                else str(df.iat[row_i, col_j])
            )
            text = escape(str(value)).replace("\n", "<br/>")

            # Font style
            style = meta.get("style", {})
            font = style.get("font") or {}
            fg = _parse_color(font.get("color")) or colors.black
            font_size = max(float(font.get("size") or 11.0) * font_scale, 5.0)
            bold = bool(font.get("bold"))
            italic = bool(font.get("italic"))
            underline = bool(font.get("underline"))

            if bold and italic:
                font_name = _FONT_BOLD_ITALIC
            elif bold:
                font_name = _FONT_BOLD
            elif italic:
                font_name = _FONT_ITALIC
            else:
                font_name = _FONT

            # HTML tags for underline
            if underline:
                text = f"<u>{text}</u>"

            h_align = (style.get("alignment") or {}).get("horizontal")
            align = _get_alignment(h_align, global_align)

            para_style = ParagraphStyle(
                name=f"c_{row_i}_{col_j}",
                fontName=font_name,
                fontSize=font_size,
                leading=font_size * 1.25,
                alignment=ALIGN_MAP.get(align, TA_LEFT),
                textColor=fg,
            )
            row_data.append(Paragraph(text, para_style))

            # Background
            fill_color = _parse_color((style.get("fill") or {}).get("color"))
            if fill_color:
                style_cmds.append(
                    ("BACKGROUND", (col_j, row_i), (col_j, row_i), fill_color)
                )

            # Frame
            border_info = style.get("border") or {}
            border_map = {
                "top": "LINEABOVE",
                "bottom": "LINEBELOW",
                "left": "LINEBEFORE",
                "right": "LINEAFTER",
            }
            for side, cmd in border_map.items():
                bdata = border_info.get(side)
                if bdata:
                    bc = _parse_color(bdata.get("color")) or colors.black
                    weight = _border_weight(bdata.get("style", "thin"))
                    style_cmds.append((cmd, (col_j, row_i), (col_j, row_i), weight, bc))

        data.append(row_data)

    return data, style_cmds


def df_to_pdf(sheets: dict, settings: dict, output) -> None:
    """
    Converts a pandas dataframe into an Excel file.
    :param sheets: Dictionary of sheet names to dataframes and metadata,
    as returned by read_xlsx()
    :param settings: Dictionary of settings (alignment, page_numbers, etc.)
    :param output: Output buffer (BytesIO) to write the PDF content to.
    :return: None
    """
    alignment = settings.get("alignment", "left")
    page_numbers = bool(settings.get("page_numbers", True))
    title = (settings.get("title") or "").strip()

    prepared = []
    for name, payload in sheets.items():
        df = payload.get("dataframe")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.fillna("")
        prepared.append(
            (
                name,
                df,
                payload.get("excel_col_widths", []),
                payload.get("cells", []),
                payload.get("merges", []),
            )
        )

    if not prepared:
        return

    max_cols = max(df.shape[1] for _, df, *_ in prepared)
    page_size = A4 if max_cols <= 6 else make_landscape(A4)
    available_pt = page_size[0] - 2 * MARGIN

    doc = SimpleDocTemplate(
        output,
        pagesize=page_size,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.0 * cm if page_numbers else MARGIN,
    )

    story = []

    if title:
        story.append(
            Paragraph(
                escape(title),
                ParagraphStyle(
                    "title",
                    fontName=_FONT_BOLD,
                    fontSize=14,
                    leading=18,
                    alignment=TA_CENTER,
                    spaceAfter=10,
                ),
            )
        )
        story.append(Spacer(1, 0.4 * cm))

    for i, (name, df, excel_widths, cells, merges) in enumerate(prepared):
        if i > 0:
            story.append(PageBreak())

        # Sheet specific font size and padding
        sheet_cols = df.shape[1]
        sheet_font_scale = _get_font_scale(sheet_cols)
        sheet_padding = _get_padding(sheet_cols)

        # Sheet name if there are more than one sheet
        if len(prepared) > 1:
            story.append(
                Paragraph(
                    escape(name),
                    ParagraphStyle(
                        "sheet",
                        fontName=_FONT_BOLD,
                        fontSize=11,
                        leading=14,
                        textColor=colors.Color(0.22, 0.45, 0.70),
                        spaceBefore=12,
                        spaceAfter=4,
                    ),
                )
            )

        col_widths = _col_widths_pts(
            excel_widths,
            sheet_cols,
            available_pt,
            cells,
            merges,
        )
        table_data, style_cmds = _build_table_data(
            df,
            cells,
            merges,
            sheet_font_scale,
            alignment,
            sheet_padding,
        )

        if not table_data:
            continue

        tbl = Table(table_data, colWidths=col_widths, repeatRows=1, splitInRow=1)
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)

    on_page = _page_number_callback if page_numbers else None
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
