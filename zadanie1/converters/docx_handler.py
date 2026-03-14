import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

_ALIGN = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
}

_MIN_COL_TWIPS = 567  # ~1 cm


def _add_page_number_field(footer) -> None:
    para = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = para.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, end])


def _shade_cell(cell, fill_hex: str) -> None:
    """Set solid background colour on a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex.lstrip("#").upper())
    existing = tcPr.find(qn("w:shd"))
    if existing is not None:
        tcPr.remove(existing)
    tcPr.append(shd)


def _set_cell_width(cell, width_twips: int) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcW = OxmlElement("w:tcW")
    tcW.set(qn("w:w"), str(width_twips))
    tcW.set(qn("w:type"), "dxa")
    existing = tcPr.find(qn("w:tcW"))
    if existing is not None:
        tcPr.remove(existing)
    tcPr.append(tcW)


def _normalize_sheet_payload(payload):
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


def _safe_dataframe(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.fillna("")
    return pd.DataFrame()


def _normalize_widths(excel_widths: list, n_cols: int) -> list[float]:
    widths = [
        float(w) if w not in (None, "") else 8.43
        for w in (excel_widths or [])[:n_cols]
    ]
    if len(widths) < n_cols:
        widths.extend([8.43] * (n_cols - len(widths)))
    return widths


def _excel_to_twips_scaled(excel_widths: list[float], available_twips: int) -> list[int]:
    """Convert Excel char-unit widths to twips, scaled to fill available_twips."""
    total = sum(excel_widths) or len(excel_widths) or 1
    scaled = [available_twips * w / total for w in excel_widths]
    twips = [max(int(w), _MIN_COL_TWIPS) for w in scaled]
    total_tw = sum(twips)
    if total_tw > available_twips:
        factor = available_twips / total_tw
        twips = [max(int(w * factor), _MIN_COL_TWIPS) for w in twips]
    return twips


def _cell_meta(cells: list, row: int, col: int):
    """Return (value, style_dict) or (None, {})."""
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


def _parse_rgb(value) -> RGBColor | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().lstrip("#")
    if len(raw) == 8:
        raw = raw[2:]
    if len(raw) == 6:
        try:
            return RGBColor(int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
        except ValueError:
            return None
    return None


def _to_para_align(h_align: str | None, fallback) -> WD_ALIGN_PARAGRAPH:
    if isinstance(h_align, str):
        mapping = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "general": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "centre": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
        }
        return mapping.get(h_align.lower().strip(), fallback)
    return fallback


def df_to_docx(sheets, settings: dict, output) -> None:
    alignment = _ALIGN.get(
        settings.get("alignment", "left"), WD_ALIGN_PARAGRAPH.LEFT
    )
    line_spacing = float(settings.get("line_spacing", 1.15))
    space_after = int(settings.get("space_after", 6))
    page_numbers = settings.get("page_numbers", True)
    title = settings.get("title", "").strip()

    # Normalise all sheets
    normalized: dict[str, tuple] = {}
    for sheet_name, payload in sheets.items():
        df, excel_widths, cells, merges = _normalize_sheet_payload(payload)
        df = _safe_dataframe(df)
        excel_widths = _normalize_widths(excel_widths, df.shape[1])
        normalized[sheet_name] = (df, excel_widths, cells, merges)

    # Decide orientation based on max column count
    max_cols = max(
        (df.shape[1] for df, _, _, _ in normalized.values() if not df.empty),
        default=1,
    )
    use_landscape = max_cols > 6

    # Font scale factor for preserving relative sizes from Excel
    if max_cols <= 10:
        font_scale = 1.0
    elif max_cols <= 15:
        font_scale = 0.85
    elif max_cols <= 25:
        font_scale = 0.72
    else:
        font_scale = 0.6

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    if use_landscape:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Cm(29.7), Cm(21.0)
    else:
        section.orientation = WD_ORIENT.PORTRAIT
        section.page_width, section.page_height = Cm(21.0), Cm(29.7)

    margins_twips = int((section.left_margin.pt + section.right_margin.pt) * 20)
    page_w_twips = int(section.page_width.pt * 20)
    available_twips = page_w_twips - margins_twips

    if title:
        heading = doc.add_heading(title, level=0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for sheet_name, (df, excel_widths, cells, merges) in normalized.items():
        if len(normalized) > 1:
            sh = doc.add_heading(sheet_name, level=1)
            sh.alignment = alignment

        if df.empty:
            continue

        twip_widths = _excel_to_twips_scaled(excel_widths, available_twips)
        n_rows, n_cols = df.shape

        table = doc.add_table(rows=n_rows, cols=n_cols)
        table.style = "Table Grid"

        # Handle merged cells
        for merge in merges:
            if merge.get("is_anchor"):
                try:
                    start_cell = table.cell(merge["start_row"], merge["start_col"])
                    end_cell = table.cell(merge["end_row"], merge["end_col"])
                    start_cell.merge(end_cell)
                except (IndexError, ValueError):
                    pass

        for row_i in range(n_rows):
            for col_j in range(n_cols):
                # Skip cells hidden by merge
                if _is_hidden_by_merge(cells, row_i, col_j):
                    continue

                cell = table.cell(row_i, col_j)
                meta_value, style = _cell_meta(cells, row_i, col_j)
                df_val = str(df.iat[row_i, col_j])
                value = str(meta_value) if meta_value is not None else df_val
                cell.text = value

                _set_cell_width(cell, twip_widths[col_j])

                font_info = style.get("font") or {}
                fill_info = style.get("fill") or {}
                align_info = style.get("alignment") or {}

                # Cell background from metadata
                if fill_info.get("color"):
                    _shade_cell(cell, fill_info["color"].lstrip("#"))

                # Font size: preserve original, scaled by factor
                original_size = float(font_info.get("size") or 11.0)
                scaled_size = max(original_size * font_scale, 6.0)

                for para in cell.paragraphs:
                    # Alignment from metadata, fallback to global
                    para.alignment = _to_para_align(
                        align_info.get("horizontal"), alignment
                    )

                    fmt = para.paragraph_format
                    fmt.line_spacing = line_spacing
                    fmt.space_after = Pt(space_after)

                    for run in para.runs:
                        run.font.size = Pt(scaled_size)
                        if font_info.get("bold"):
                            run.bold = True
                        if font_info.get("italic"):
                            run.italic = True
                        color = _parse_rgb(font_info.get("color"))
                        if color:
                            run.font.color.rgb = color

        doc.add_paragraph()

    if page_numbers:
        _add_page_number_field(section.footer)

    doc.save(output)
