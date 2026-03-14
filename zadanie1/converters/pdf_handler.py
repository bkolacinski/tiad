import os
from html import escape
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape as make_landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

_ALIGN_PARA = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}

_GRID_COLOR = colors.Color(0.82, 0.82, 0.82)

_FONT = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_MARGIN = 1.5 * cm
_SAFE_MIN_COL_WIDTH = 5.0

# Approximate conversion: 1 Excel character unit ~ 7 points
_EXCEL_CHAR_TO_PT = 7.0


def _register_fonts() -> None:
    global _FONT, _FONT_BOLD
    candidates = [
        (
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "ArialPL",
            "ArialPL-Bold",
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans",
            "DejaVuSans-Bold",
        ),
    ]
    for regular, bold, rname, bname in candidates:
        if not (os.path.exists(regular) and os.path.exists(bold)):
            continue
        try:
            pdfmetrics.registerFont(TTFont(rname, regular))
            pdfmetrics.registerFont(TTFont(bname, bold))
            _FONT = rname
            _FONT_BOLD = bname
            return
        except Exception:
            continue


_register_fonts()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_sheet(payload: Any):
    """Return (DataFrame, excel_col_widths, cells_meta, merges)."""
    if isinstance(payload, dict):
        df = payload.get("dataframe")
        if df is None:
            df = payload.get("df")
        if df is None:
            df = payload.get("sheet")
        excel_widths = payload.get("excel_col_widths")
        if excel_widths is None:
            excel_widths = payload.get("column_widths")
        if excel_widths is None:
            excel_widths = payload.get("widths")
        if excel_widths is None:
            excel_widths = []
        cells = payload.get("cells") or []
        merges = payload.get("merges") or []
        return df, list(excel_widths), cells, merges

    if isinstance(payload, (tuple, list)) and len(payload) >= 1:
        df = payload[0]
        excel_widths = list(payload[1]) if len(payload) > 1 else []
        meta = payload[2] if len(payload) > 2 and isinstance(payload[2], dict) else {}
        cells = meta.get("cells") or []
        merges = meta.get("merges") or []
        return df, excel_widths, cells, merges

    return payload, [], [], []


def _parse_color(value: Any) -> colors.Color | None:
    if isinstance(value, colors.Color):
        return value
    if isinstance(value, str):
        raw = value.strip().lstrip("#")
        if len(raw) == 8:
            raw = raw[2:]
        if len(raw) == 6:
            try:
                r = int(raw[0:2], 16) / 255
                g = int(raw[2:4], 16) / 255
                b = int(raw[4:6], 16) / 255
                return colors.Color(r, g, b)
            except ValueError:
                return None
    return None


def _col_widths_pts(excel_widths: list, n_cols: int, available_pt: float) -> list[float]:
    """Convert Excel char-unit widths to points, scaling proportionally to fill page."""
    widths = list(excel_widths[:n_cols])
    while len(widths) < n_cols:
        widths.append(8.43)

    # Convert to natural point widths
    natural = [w * _EXCEL_CHAR_TO_PT for w in widths]
    total_natural = sum(natural) or 1.0

    # Scale to fill available width proportionally
    factor = available_pt / total_natural
    scaled = [max(w * factor, _SAFE_MIN_COL_WIDTH) for w in natural]

    # Re-adjust if minimums pushed us over
    total_scaled = sum(scaled)
    if total_scaled > available_pt:
        adjustable = [(i, w) for i, w in enumerate(scaled) if w > _SAFE_MIN_COL_WIDTH]
        if adjustable:
            excess = total_scaled - available_pt
            per_col = excess / len(adjustable)
            for i, w in adjustable:
                scaled[i] = max(w - per_col, _SAFE_MIN_COL_WIDTH)

    return scaled


def _cell_meta(cells: list, row: int, col: int) -> tuple[str | None, dict]:
    """Return (value, style_dict)."""
    try:
        m = cells[row][col]
        return m.get("value"), m.get("style") or {}
    except (IndexError, TypeError, AttributeError):
        return None, {}


def _is_hidden_by_merge(cells: list, row: int, col: int) -> bool:
    try:
        m = cells[row][col]
        return bool(m.get("hidden_by_merge"))
    except (IndexError, TypeError, AttributeError):
        return False


def _to_align(h_align: Any, fallback: str) -> str:
    if isinstance(h_align, str):
        return {
            "left": "left",
            "general": "left",
            "center": "center",
            "centre": "center",
            "right": "right",
            "justify": "left",
        }.get(h_align.lower().strip(), fallback)
    return fallback


def _make_para_style(
    alignment: str, font_size: float, bold: bool, fg: colors.Color
) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"c_{alignment}_{font_size}_{int(bold)}",
        fontName=_FONT_BOLD if bold else _FONT,
        fontSize=font_size,
        leading=font_size * 1.25,
        alignment=_ALIGN_PARA.get(alignment, TA_LEFT),
        textColor=fg,
        wordWrap="LTR",
    )


def _page_number_cb(canvas, doc):
    canvas.saveState()
    canvas.setFont(_FONT, 8)
    canvas.setFillColor(colors.Color(0.5, 0.5, 0.5))
    canvas.drawCentredString(doc.pagesize[0] / 2, 0.7 * cm, str(canvas.getPageNumber()))
    canvas.restoreState()


def _border_weight(style_name: str) -> float:
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


# ---------------------------------------------------------------------------
# Table builder — preserves original Excel styling
# ---------------------------------------------------------------------------


def _build_table(
    df,
    cells: list,
    merges: list,
    col_widths_pt: list,
    font_scale: float,
    global_align: str,
    padding_h: float = 4.0,
):
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, []

    n_rows, n_cols = df.shape
    data = []

    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.25, _GRID_COLOR),
        ("LEFTPADDING", (0, 0), (-1, -1), padding_h),
        ("RIGHTPADDING", (0, 0), (-1, -1), padding_h),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]

    # Merged cell spans
    for merge in merges:
        if merge.get("is_anchor"):
            style_cmds.append(
                (
                    "SPAN",
                    (merge["start_col"], merge["start_row"]),
                    (merge["end_col"], merge["end_row"]),
                )
            )

    for row_i in range(n_rows):
        row_data = []
        for col_j in range(n_cols):
            # Hidden merge cells — add empty placeholder
            if _is_hidden_by_merge(cells, row_i, col_j):
                row_data.append("")
                continue

            meta_value, style = _cell_meta(cells, row_i, col_j)
            df_val = str(df.iat[row_i, col_j])
            text = str(meta_value) if meta_value is not None else df_val

            font = style.get("font") or {}
            is_bold = bool(font.get("bold"))

            # Font color from metadata
            fg = _parse_color(font.get("color")) or colors.black

            # Alignment from metadata, fallback to global
            h_align = (style.get("alignment") or {}).get("horizontal")
            align = _to_align(h_align, global_align)

            # Font size: preserve original, scaled by factor for page fit
            original_size = float(font.get("size") or 11.0)
            cell_font_size = max(original_size * font_scale, 5.0)

            para_style = _make_para_style(align, cell_font_size, is_bold, fg)
            safe = escape(text).replace("\n", "<br/>")
            row_data.append(Paragraph(safe, para_style))

            # Cell background from metadata
            fill_color = (style.get("fill") or {}).get("color")
            if fill_color:
                bg = _parse_color(fill_color)
                if bg:
                    style_cmds.append(
                        ("BACKGROUND", (col_j, row_i), (col_j, row_i), bg)
                    )

            # Cell borders from metadata
            border_info = style.get("border") or {}
            border_side_map = {
                "top": "LINEABOVE",
                "bottom": "LINEBELOW",
                "left": "LINEBEFORE",
                "right": "LINEAFTER",
            }
            for side, cmd_name in border_side_map.items():
                bdata = border_info.get(side)
                if not bdata:
                    continue
                bc = _parse_color(bdata.get("color")) or colors.black
                weight = _border_weight(bdata.get("style", "thin"))
                style_cmds.append(
                    (cmd_name, (col_j, row_i), (col_j, row_i), weight, bc)
                )

        data.append(row_data)

    return data, style_cmds


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def df_to_pdf(sheets: dict, settings: dict, output) -> None:
    import pandas as pd

    alignment = settings.get("alignment", "left")
    page_numbers = bool(settings.get("page_numbers", True))
    title = (settings.get("title") or "").strip()

    # Normalise sheets and find max column count
    normalized: dict[str, tuple] = {}
    max_cols = 1
    for name, payload in sheets.items():
        df, excel_widths, cells, merges = _normalize_sheet(payload)
        if not isinstance(df, pd.DataFrame):
            continue
        df = df.fillna("")
        if df.empty:
            continue
        if df.shape[1] > max_cols:
            max_cols = df.shape[1]
        normalized[name] = (df, excel_widths, cells, merges)

    if not normalized:
        return

    # Page orientation and size
    use_landscape = max_cols > 6
    if max_cols > 20:
        page_w = max(841.89, max_cols * 30.0)
        page_h = max(595.27, page_w * 0.707)
        page_size = (page_w, page_h)
    else:
        page_size = make_landscape(A4) if use_landscape else A4
    page_w, _page_h = page_size

    available_pt = page_w - 2 * _MARGIN

    # Font scale factor — preserve relative sizes but scale down for wide tables
    if max_cols <= 10:
        font_scale = 1.0
    elif max_cols <= 15:
        font_scale = 0.85
    elif max_cols <= 25:
        font_scale = 0.72
    elif max_cols <= 50:
        font_scale = 0.6
    else:
        font_scale = 0.5

    # Padding
    padding_h = 4.0
    if max_cols > 20:
        padding_h = 3.0
    if max_cols > 30:
        padding_h = 2.0
    if max_cols > 60:
        padding_h = 1.0

    bottom_margin = 2.0 * cm if page_numbers else _MARGIN

    doc = SimpleDocTemplate(
        output,
        pagesize=page_size,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=bottom_margin,
    )

    title_style = ParagraphStyle(
        name="title",
        fontName=_FONT_BOLD,
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=10,
    )
    sheet_style = ParagraphStyle(
        name="sheet",
        fontName=_FONT_BOLD,
        fontSize=11,
        leading=14,
        textColor=colors.Color(0.22, 0.45, 0.70),
        spaceBefore=12,
        spaceAfter=4,
    )

    story = []

    if title:
        story.append(Paragraph(escape(title), title_style))
        story.append(Spacer(1, 0.4 * cm))

    for sheet_name, (df, excel_widths, cells, merges) in normalized.items():
        if len(normalized) > 1:
            story.append(Paragraph(escape(sheet_name), sheet_style))

        col_widths_pt = _col_widths_pts(excel_widths, df.shape[1], available_pt)
        table_data, style_cmds = _build_table(
            df, cells, merges, col_widths_pt, font_scale, alignment, padding_h
        )

        if not table_data:
            continue

        tbl = Table(
            table_data, colWidths=col_widths_pt, repeatRows=1, splitInRow=1
        )
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 0.5 * cm))

    on_page = _page_number_cb if page_numbers else None
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
