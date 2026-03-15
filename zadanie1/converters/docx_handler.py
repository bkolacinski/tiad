import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm
from lxml import etree

_ALIGN = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
}

_MIN_COL_TWIPS = 567
_MAX_WORD_COLS = 63

# ---------------------------------------------------------------------------
# Pre-computed namespace-qualified tag/attribute names (lxml {uri}local form).
# Using these with etree.SubElement avoids repeated qn() lookups.
# ---------------------------------------------------------------------------
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_XML = "http://www.w3.org/XML/1998/namespace"

_T_P = f"{{{_W}}}p"
_T_PPR = f"{{{_W}}}pPr"
_T_SPACING = f"{{{_W}}}spacing"
_T_JC = f"{{{_W}}}jc"
_T_R = f"{{{_W}}}r"
_T_RPR = f"{{{_W}}}rPr"
_T_B = f"{{{_W}}}b"
_T_I = f"{{{_W}}}i"
_T_COLOR = f"{{{_W}}}color"
_T_SZ = f"{{{_W}}}sz"
_T_SZCS = f"{{{_W}}}szCs"
_T_U = f"{{{_W}}}u"
_T_STRIKE = f"{{{_W}}}strike"
_T_T = f"{{{_W}}}t"
_T_SHD = f"{{{_W}}}shd"
_T_TCW = f"{{{_W}}}tcW"
_T_TCPR = f"{{{_W}}}tcPr"
_T_TR = f"{{{_W}}}tr"
_T_TC = f"{{{_W}}}tc"
_T_GRIDSPAN = f"{{{_W}}}gridSpan"
_T_VMERGE = f"{{{_W}}}vMerge"
_T_TCBORDERS = f"{{{_W}}}tcBorders"
_T_VALIGN = f"{{{_W}}}vAlign"
_T_RFONTS = f"{{{_W}}}rFonts"

_A_VAL = f"{{{_W}}}val"
_A_W = f"{{{_W}}}w"
_A_TYPE = f"{{{_W}}}type"
_A_LINE = f"{{{_W}}}line"
_A_LINERULE = f"{{{_W}}}lineRule"
_A_AFTER = f"{{{_W}}}after"
_A_SZ = f"{{{_W}}}sz"
_A_SPACE = f"{{{_W}}}space"
_A_COLOR = f"{{{_W}}}color"
_A_FILL = f"{{{_W}}}fill"
_A_ASCII = f"{{{_W}}}ascii"
_A_HANSI = f"{{{_W}}}hAnsi"
_A_CS = f"{{{_W}}}cs"
_A_XML_SPACE = f"{{{_XML}}}space"

# Pre-cache qn() for python-docx API calls that still need it
_QN_W_FLDCHARTYPE = qn("w:fldCharType")

_ALIGN_MAP = {
    "left": "left", "general": "left",
    "center": "center", "centre": "center",
    "right": "right", "justify": "left",
}

# Color cache
_RGB_CACHE: dict[str, str | None] = {}


def _parse_rgb_hex(value) -> str | None:
    if not isinstance(value, str):
        return None
    if value in _RGB_CACHE:
        return _RGB_CACHE[value]
    raw = value.strip().lstrip("#")
    if len(raw) == 8:
        raw = raw[2:]
    result = raw.upper() if len(raw) == 6 else None
    _RGB_CACHE[value] = result
    return result


def _pt_half(size: float) -> str:
    return str(round(size * 2))


def _add_page_number_field(footer) -> None:
    para = footer.paragraphs[0]
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # OOXML requires field chars in separate runs
    run1 = para.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(_QN_W_FLDCHARTYPE, "begin")
    run1._r.append(begin)

    run2 = para.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    run2._r.append(instr)

    run3 = para.add_run()
    end = OxmlElement("w:fldChar")
    end.set(_QN_W_FLDCHARTYPE, "end")
    run3._r.append(end)


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
    total = sum(excel_widths) or len(excel_widths) or 1
    scaled = [available_twips * w / total for w in excel_widths]
    twips = [max(int(w), _MIN_COL_TWIPS) for w in scaled]
    total_tw = sum(twips)
    if total_tw > available_twips:
        factor = available_twips / total_tw
        twips = [max(int(w * factor), _MIN_COL_TWIPS) for w in twips]
    return twips


def _border_val(style: str) -> str:
    # Map openpyxl/excel styles to Word border types
    return {
        "thin": "single",
        "medium": "single",
        "thick": "single",
        "double": "double",
        "hair": "single",
        "dashed": "dashed",
        "dotted": "dotted",
        "dashDot": "dashDot",
        "dashDotDot": "dashDotDot",
        "slantDashDot": "dashDotStr",
        "mediumDashed": "mediumDashed",
        "mediumDashDot": "mediumDashDot",
        "mediumDashDotDot": "mediumDashDotDot",
    }.get(style, "single")


def _border_sz(style: str) -> str:
    # Border size in 1/8 pt
    return {
        "thin": "4",      # 0.5 pt
        "medium": "12",   # 1.5 pt
        "thick": "24",    # 3.0 pt
        "hair": "2",      # 0.25 pt
        "double": "4",
    }.get(style, "4")


def _set_cell_content(tc, value: str, width_twips: int,
                      font_info: dict, fill_info: dict, align_info: dict,
                      border_info: dict,
                      font_scale: float, line_val: str, after_val: str,
                      fallback_jc: str, v_merge: str | None = None) -> None:
    """Modify an existing cell's content using fast lxml SubElement calls.

    The cell (tc) was created by doc.add_table() so the structure is valid.
    We modify tcPr attributes and replace the paragraph content.
    """
    SE = etree.SubElement  # local ref for speed

    # --- tcPr: update width and vertical merge ---
    tcPr = tc.find(_T_TCPR)
    if tcPr is None:
        tcPr = etree.Element(_T_TCPR)
        tc.insert(0, tcPr)

    # Schema order for tcPr:
    # cnfStyle, tcW, gridSpan, hMerge, vMerge, tcBorders, shd, ..., vAlign
    
    # 1. tcW (Width)
    tcW = tcPr.find(_T_TCW)
    if tcW is None:
        tcW = etree.Element(_T_TCW)
        tcPr.insert(0, tcW)
    tcW.set(_A_W, str(width_twips))
    tcW.set(_A_TYPE, "dxa")

    # 3. vMerge
    if v_merge:
        vm = tcPr.find(_T_VMERGE)
        if vm is None:
            vm = etree.Element(_T_VMERGE)
            # Position after tcW and gridSpan
            pos = 0
            for i, child in enumerate(tcPr):
                if child.tag in (_T_TCW, _T_GRIDSPAN):
                    pos = i + 1
            tcPr.insert(pos, vm)
        vm.set(_A_VAL, v_merge)

    # 4. tcBorders
    if border_info:
        # Clear existing tcBorders to ensure correct child order (top, left, bottom, right)
        old_borders = tcPr.find(_T_TCBORDERS)
        if old_borders is not None:
            tcPr.remove(old_borders)
        
        borders = etree.Element(_T_TCBORDERS)
        # Position after vMerge/gridSpan/tcW
        pos = 0
        for i, child in enumerate(tcPr):
            if child.tag in (_T_TCW, _T_GRIDSPAN, f"{{{_W}}}hMerge", _T_VMERGE):
                pos = i + 1
        tcPr.insert(pos, borders)
        
        # OOXML order: top, left, bottom, right
        for side in ("top", "left", "bottom", "right"):
            bdata = border_info.get(side)
            if not bdata:
                continue
            
            tag = f"{{{_W}}}{side}"
            b_el = SE(borders, tag)
            b_el.set(_A_VAL, _border_val(bdata.get("style")))
            b_el.set(_A_SZ, _border_sz(bdata.get("style")))
            b_el.set(_A_SPACE, "0")
            bc = _parse_rgb_hex(bdata.get("color")) or "000000"
            b_el.set(_A_COLOR, bc)

    # 5. shd (Shading)
    fill_color = fill_info.get("color") if fill_info else None
    if fill_color:
        fill_hex = _parse_rgb_hex(fill_color)
        if fill_hex:
            # Clear existing shd to ensure correct order
            old_shd = tcPr.find(_T_SHD)
            if old_shd is not None:
                tcPr.remove(old_shd)
                
            shd = etree.Element(_T_SHD)
            # Position after tcBorders
            pos = 0
            for i, child in enumerate(tcPr):
                if child.tag in (_T_TCW, _T_GRIDSPAN, _T_VMERGE, _T_TCBORDERS):
                    pos = i + 1
            tcPr.insert(pos, shd)
            shd.set(_A_VAL, "clear")
            shd.set(_A_COLOR, "auto")
            shd.set(_A_FILL, fill_hex)

    # 6. vAlign
    v_align = align_info.get("vertical") if align_info else None
    if v_align:
        va_map = {"center": "center", "centre": "center", "bottom": "bottom", "top": "top"}
        va_val = va_map.get(v_align.lower().strip())
        if va_val:
            va = tcPr.find(_T_VALIGN)
            if va is None:
                va = SE(tcPr, _T_VALIGN)
            va.set(_A_VAL, va_val)

    # --- Replace paragraph content ---
    # Remove all existing paragraphs
    for old_p in tc.findall(_T_P):
        tc.remove(old_p)

    # Build new paragraph
    p = SE(tc, _T_P)

    # pPr (paragraph properties)
    pPr = SE(p, _T_PPR)

    # spacing BEFORE jc (OOXML schema order)
    sp = SE(pPr, _T_SPACING)
    sp.set(_A_LINE, line_val)
    sp.set(_A_LINERULE, "auto")
    sp.set(_A_AFTER, after_val)

    h_align = align_info.get("horizontal") if align_info else None
    jc_val = fallback_jc
    if isinstance(h_align, str):
        jc_val = _ALIGN_MAP.get(h_align.lower().strip(), fallback_jc)
    jc = SE(pPr, _T_JC)
    jc.set(_A_VAL, jc_val)

    # Run
    r = SE(p, _T_R)

    # rPr (run properties)
    rPr = SE(r, _T_RPR)

    # rFonts must be first in rPr
    font_name = font_info.get("name") if font_info else None
    if font_name:
        rf = SE(rPr, _T_RFONTS)
        rf.set(_A_ASCII, font_name)
        rf.set(_A_HANSI, font_name)
        rf.set(_A_CS, font_name)

    if font_info.get("bold"):
        SE(rPr, _T_B).set(_A_VAL, "1")
    if font_info.get("italic"):
        SE(rPr, _T_I).set(_A_VAL, "1")
    if font_info.get("strike"):
        SE(rPr, _T_STRIKE).set(_A_VAL, "1")

    font_color_hex = _parse_rgb_hex(font_info.get("color")) if font_info else None
    if font_color_hex:
        col = SE(rPr, _T_COLOR)
        col.set(_A_VAL, font_color_hex)

    original_size = float(font_info.get("size") or 11.0) if font_info else 11.0
    sz_val = _pt_half(max(original_size * font_scale, 6.0))

    sz = SE(rPr, _T_SZ)
    sz.set(_A_VAL, sz_val)
    szCs = SE(rPr, _T_SZCS)
    szCs.set(_A_VAL, sz_val)

    u_style = font_info.get("underline") if font_info else None
    if u_style:
        # Excel's 'single', 'double', etc. usually map directly or to 'single'
        u_val = "single" if u_style in (True, "single") else (u_style if isinstance(u_style, str) else "single")
        SE(rPr, _T_U).set(_A_VAL, u_val)

    # Text
    t = SE(r, _T_T)
    t.set(_A_XML_SPACE, "preserve")
    t.text = value


def _find_safe_splits(merges, n_cols, max_cols=_MAX_WORD_COLS):
    """Return list of (start, end) column ranges that avoid breaking merges."""
    if n_cols <= max_cols:
        return [(0, n_cols)]

    blocked = set()
    for merge in merges:
        for c in range(merge["start_col"] + 1, merge["end_col"] + 1):
            blocked.add(c)

    ranges = []
    start = 0
    while start < n_cols:
        if start + max_cols >= n_cols:
            ranges.append((start, n_cols))
            break
        for sp in range(start + max_cols, start, -1):
            if sp not in blocked:
                ranges.append((start, sp))
                start = sp
                break
        else:
            ranges.append((start, start + max_cols))
            start += max_cols
    return ranges


def _split_sheet(df, excel_widths, cells, merges, max_cols=_MAX_WORD_COLS):
    """Split wide sheets into column chunks that fit within Word's limit."""
    n_cols = df.shape[1]
    if n_cols <= max_cols:
        return [(df, excel_widths, cells, merges)]

    col_ranges = _find_safe_splits(merges, n_cols, max_cols)
    chunks = []

    for col_start, col_end in col_ranges:
        chunk_width = col_end - col_start

        df_chunk = df.iloc[:, col_start:col_end].copy()
        df_chunk.columns = range(chunk_width)
        widths_chunk = excel_widths[col_start:col_end]
        cells_chunk = [row[col_start:col_end] for row in cells]

        merges_chunk = []
        valid_ranges = set()
        for merge in merges:
            sc, ec = merge["start_col"], merge["end_col"]
            if sc >= col_start and ec < col_end:
                m = dict(merge)
                m["start_col"] = sc - col_start
                m["end_col"] = ec - col_start
                merges_chunk.append(m)
                valid_ranges.add(merge["range"])

        # Fix cells whose merge was dropped (cross-boundary, shouldn't happen
        # with safe splits but handle defensively)
        for row in cells_chunk:
            for ci, cell in enumerate(row):
                merge_info = cell.get("merge")
                if merge_info and merge_info.get("range") not in valid_ranges:
                    row[ci] = {
                        "value": cell.get("value", ""),
                        "style": cell.get("style", {}),
                        "merge": None,
                        "hidden_by_merge": False,
                    }

        chunks.append((df_chunk, widths_chunk, cells_chunk, merges_chunk))

    return chunks


def df_to_docx(sheets, settings: dict, output) -> None:
    alignment_str = settings.get("alignment", "left")
    alignment = _ALIGN.get(alignment_str, WD_ALIGN_PARAGRAPH.LEFT)
    line_spacing = float(settings.get("line_spacing", 1.15))
    space_after = int(settings.get("space_after", 6))
    page_numbers = settings.get("page_numbers", True)
    title = settings.get("title", "").strip()

    fallback_jc = _ALIGN_MAP.get(alignment_str, "left")
    line_val = str(int(line_spacing * 240))
    after_val = str(int(space_after * 20))

    # Normalise all sheets and split wide ones
    split_sheets: dict[str, list[tuple]] = {}
    for sheet_name, payload in sheets.items():
        df, excel_widths, cells, merges = _normalize_sheet_payload(payload)
        df = _safe_dataframe(df)
        excel_widths = _normalize_widths(excel_widths, df.shape[1])
        chunks = _split_sheet(df, excel_widths, cells, merges)
        split_sheets[sheet_name] = chunks

    max_cols = max(
        (chunk_df.shape[1]
         for chunks in split_sheets.values()
         for chunk_df, _, _, _ in chunks
         if not chunk_df.empty),
        default=1,
    )
    use_landscape = max_cols > 6

    if max_cols <= 10:
        font_scale = 1.0
    elif max_cols <= 15:
        font_scale = 0.85
    elif max_cols <= 25:
        font_scale = 0.72
    elif max_cols <= 40:
        font_scale = 0.65
    else:
        font_scale = 0.55

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    if max_cols > 20:
        # Dynamic width: approx 1.2 cm per column, but not less than A4 landscape
        # Word max page width is 22 inches (55.88 cm).
        use_landscape = True
        page_w_cm = min(55.8, max(29.7, max_cols * 1.2))
        page_h_cm = 21.0
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = Cm(page_w_cm), Cm(page_h_cm)
    elif use_landscape:
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

    for sheet_name, chunks in split_sheets.items():
        if len(split_sheets) > 1:
            sh = doc.add_heading(sheet_name, level=1)
            sh.alignment = alignment

        for chunk_idx, (df, excel_widths, cells, merges) in enumerate(chunks):
            if df.empty:
                continue

            twip_widths = _excel_to_twips_scaled(excel_widths, available_twips)
            n_rows, n_cols = df.shape

            table = doc.add_table(rows=n_rows, cols=n_cols)
            table.style = "Table Grid"

            for merge in merges:
                if merge.get("is_anchor"):
                    try:
                        start_cell = table.cell(merge["start_row"], merge["start_col"])
                        end_cell = table.cell(merge["end_row"], merge["end_col"])
                        start_cell.merge(end_cell)
                    except (IndexError, ValueError):
                        pass

            tbl_el = table._tbl

            tbl_grid = tbl_el.find(qn("w:tblGrid"))
            if tbl_grid is not None:
                tbl_el.remove(tbl_grid)
            tbl_grid = OxmlElement("w:tblGrid")
            for tw in twip_widths:
                gc = OxmlElement("w:gridCol")
                gc.set(qn("w:w"), str(tw))
                tbl_grid.append(gc)

            tbl_pr = tbl_el.find(qn("w:tblPr"))
            if tbl_pr is not None:
                idx = tbl_el.index(tbl_pr)
                tbl_el.insert(idx + 1, tbl_grid)
            else:
                tbl_el.insert(0, tbl_grid)

            tr_list = tbl_el.findall(_T_TR)
            df_values = df.values.astype(str)

            for row_i, tr in enumerate(tr_list):
                tc_list = tr.findall(_T_TC)
                cells_row = cells[row_i] if row_i < len(cells) else None

                true_col_j = 0
                for tc in tc_list:
                    span = 1
                    tcPr_el = tc.find(_T_TCPR)
                    if tcPr_el is not None:
                        gs_el = tcPr_el.find(_T_GRIDSPAN)
                        if gs_el is not None:
                            try:
                                span = int(gs_el.get(_A_VAL, "1"))
                            except ValueError:
                                span = 1

                    cd = None
                    if cells_row and true_col_j < len(cells_row):
                        cd = cells_row[true_col_j]

                    if cd and cd.get("hidden_by_merge"):
                        cell_w = sum(twip_widths[true_col_j : true_col_j + span])
                        tcPr = tc.find(_T_TCPR)
                        if tcPr is None:
                            tcPr = etree.Element(_T_TCPR)
                            tc.insert(0, tcPr)

                        tcW = tcPr.find(_T_TCW)
                        if tcW is None:
                            tcW = etree.Element(_T_TCW)
                            tcPr.insert(0, tcW)
                        tcW.set(_A_W, str(cell_w))
                        tcW.set(_A_TYPE, "dxa")

                        v_merge_data = cd.get("merge")
                        v_merge = v_merge_data.get("vMerge") if v_merge_data else None
                        if v_merge:
                            vm = tcPr.find(_T_VMERGE)
                            if vm is None:
                                vm = etree.Element(_T_VMERGE)
                                pos = 0
                                for i, child in enumerate(tcPr):
                                    if child.tag in (_T_TCW, _T_GRIDSPAN):
                                        pos = i + 1
                                tcPr.insert(pos, vm)
                            vm.set(_A_VAL, v_merge)

                        true_col_j += span
                        continue

                    if cd:
                        meta_value = cd.get("value")
                        style = cd.get("style") or {}
                    else:
                        meta_value = None
                        style = {}

                    value = str(meta_value) if meta_value is not None else (
                        df_values[row_i][true_col_j] if true_col_j < n_cols else ""
                    )

                    cell_w = sum(twip_widths[true_col_j : true_col_j + span])

                    _set_cell_content(
                        tc, value, cell_w,
                        style.get("font") or {},
                        style.get("fill") or {},
                        style.get("alignment") or {},
                        style.get("border") or {},
                        font_scale, line_val, after_val, fallback_jc,
                        v_merge=(cd.get("merge") or {}).get("vMerge") if cd else None
                    )
                    true_col_j += span

            doc.add_paragraph()

    if page_numbers:
        _add_page_number_field(section.footer)

    doc.save(output)
