import io
from copy import copy

import openpyxl
import pandas as pd
from openpyxl.utils import get_column_letter

DEFAULT_COLUMN_WIDTH = 8.43
DEFAULT_ROW_HEIGHT = 15.0
DEFAULT_FONT_SIZE = 11.0


def _normalize_color(value) -> str | None:
    if not value:
        return None

    rgb = getattr(value, "rgb", None)
    if rgb and isinstance(rgb, str):
        rgb = rgb.upper()
        if len(rgb) == 8:
            rgb = rgb[2:]
        if len(rgb) == 6:
            return f"#{rgb}"

    indexed = getattr(value, "indexed", None)
    if indexed is not None:
        indexed_map = {
            0: "#000000",
            1: "#FFFFFF",
            2: "#FF0000",
            3: "#00FF00",
            4: "#0000FF",
            5: "#FFFF00",
            6: "#FF00FF",
            7: "#00FFFF",
            8: "#000000",
            9: "#FFFFFF",
            10: "#FF0000",
            11: "#00FF00",
            12: "#0000FF",
            13: "#FFFF00",
            14: "#FF00FF",
            15: "#00FFFF",
            16: "#800000",
            17: "#008000",
            18: "#000080",
            19: "#808000",
            20: "#800080",
            21: "#008080",
            22: "#C0C0C0",
            23: "#808080",
        }
        return indexed_map.get(indexed)

    return None


def _extract_fill_color(fill) -> str | None:
    if not fill:
        return None

    pattern = getattr(fill, "patternType", None)
    if pattern not in {"solid", "gray125"}:
        return None

    color = _normalize_color(getattr(fill, "fgColor", None))
    if color:
        return color

    return _normalize_color(getattr(fill, "start_color", None))


def _extract_border(border) -> dict:
    if not border:
        return {}

    result = {}
    for side_name in ("left", "right", "top", "bottom"):
        side = getattr(border, side_name, None)
        if not side or not getattr(side, "style", None):
            continue
        result[side_name] = {
            "style": side.style,
            "color": _normalize_color(getattr(side, "color", None)),
        }
    return result


def _extract_alignment(alignment) -> dict:
    if not alignment:
        return {}

    return {
        "horizontal": alignment.horizontal,
        "vertical": alignment.vertical,
        "wrap_text": bool(alignment.wrap_text),
        "shrink_to_fit": bool(alignment.shrink_to_fit),
        "text_rotation": alignment.text_rotation or 0,
    }


def _extract_font(font) -> dict:
    if not font:
        return {}

    return {
        "name": font.name,
        "size": float(font.sz) if font.sz is not None else DEFAULT_FONT_SIZE,
        "bold": bool(font.b),
        "italic": bool(font.i),
        "underline": font.u if font.u else None,
        "strike": bool(font.strike),
        "color": _normalize_color(getattr(font, "color", None)),
    }


def _extract_number_format(cell) -> str:
    return cell.number_format or "General"


def _style_id(cell) -> str:
    return f"style-{getattr(cell, 'style_id', 0)}"


def _extract_cell_style(cell) -> dict:
    fill_color = _extract_fill_color(cell.fill)
    return {
        "style_id": _style_id(cell),
        "font": _extract_font(cell.font),
        "fill": {
            "color": fill_color,
        },
        "border": _extract_border(cell.border),
        "alignment": _extract_alignment(cell.alignment),
        "number_format": _extract_number_format(cell),
        "is_date": bool(cell.is_date),
    }


def _collect_merged_map(ws) -> dict[tuple[int, int], dict]:
    merged_map: dict[tuple[int, int], dict] = {}

    for merged_range in ws.merged_cells.ranges:
        min_col = merged_range.min_col
        min_row = merged_range.min_row
        max_col = merged_range.max_col
        max_row = merged_range.max_row

        meta = {
            "range": str(merged_range),
            "start_row": min_row - 1,
            "start_col": min_col - 1,
            "end_row": max_row - 1,
            "end_col": max_col - 1,
            "rowspan": max_row - min_row + 1,
            "colspan": max_col - min_col + 1,
            "is_anchor": True,
        }

        merged_map[(min_row, min_col)] = meta

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if row == min_row and col == min_col:
                    continue
                merged_map[(row, col)] = {
                    "range": str(merged_range),
                    "start_row": min_row - 1,
                    "start_col": min_col - 1,
                    "end_row": max_row - 1,
                    "end_col": max_col - 1,
                    "rowspan": max_row - min_row + 1,
                    "colspan": max_col - min_col + 1,
                    "is_anchor": False,
                }

    return merged_map


def _sheet_dimensions(ws, df: pd.DataFrame) -> tuple[int, int]:
    max_row = max(ws.max_row or 0, df.shape[0])
    max_col = max(ws.max_column or 0, df.shape[1])

    if max_row == 0:
        max_row = df.shape[0]
    if max_col == 0:
        max_col = df.shape[1]

    return max_row, max_col


def _collect_column_widths(ws, n_cols: int) -> list[float]:
    widths: list[float] = []
    for i in range(n_cols):
        letter = get_column_letter(i + 1)
        dim = ws.column_dimensions.get(letter)
        width = float(dim.width or DEFAULT_COLUMN_WIDTH) if dim else DEFAULT_COLUMN_WIDTH
        widths.append(width)
    return widths


def _collect_row_heights(ws, n_rows: int) -> list[float]:
    heights: list[float] = []
    for i in range(n_rows):
        dim = ws.row_dimensions.get(i + 1)
        height = float(dim.height or DEFAULT_ROW_HEIGHT) if dim else DEFAULT_ROW_HEIGHT
        heights.append(height)
    return heights


def _cell_display_value(cell) -> str:
    if cell.value is None:
        return ""
    return str(cell.value)


def _build_dataframe(ws, n_rows: int, n_cols: int) -> pd.DataFrame:
    rows = []
    for row_idx in range(1, n_rows + 1):
        row = []
        for col_idx in range(1, n_cols + 1):
            row.append(_cell_display_value(ws.cell(row=row_idx, column=col_idx)))
        rows.append(row)
    return pd.DataFrame(rows, dtype=str).fillna("")


def _collect_cells_metadata(ws, n_rows: int, n_cols: int, merged_map: dict[tuple[int, int], dict]) -> list[list[dict]]:
    cells_meta: list[list[dict]] = []

    for row_idx in range(1, n_rows + 1):
        row_meta: list[dict] = []
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            merge_meta = merged_map.get((row_idx, col_idx))

            if merge_meta and not merge_meta["is_anchor"]:
                row_meta.append(
                    {
                        "value": "",
                        "style": {},
                        "merge": merge_meta,
                        "hidden_by_merge": True,
                    }
                )
                continue

            row_meta.append(
                {
                    "value": _cell_display_value(cell),
                    "style": _extract_cell_style(cell),
                    "merge": merge_meta,
                    "hidden_by_merge": False,
                }
            )
        cells_meta.append(row_meta)

    return cells_meta


def read_xlsx(file) -> dict[str, dict]:
    """
    Returns dict[sheet_name -> metadata] where metadata contains:
    - dataframe: DataFrame with displayed cell values
    - excel_col_widths: list of Excel column widths in character units
    - row_heights: list of row heights in points
    - cells: 2D list with cell values, styles and merge metadata
    - merges: list of merged range descriptors
    """
    data = file.read()

    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)

    result: dict[str, dict] = {}

    for name in wb.sheetnames:
        ws = wb[name]
        merged_map = _collect_merged_map(ws)
        max_row, max_col = _sheet_dimensions(ws, pd.DataFrame())

        df = _build_dataframe(ws, max_row, max_col)
        col_widths = _collect_column_widths(ws, max_col)
        row_heights = _collect_row_heights(ws, max_row)
        cells_meta = _collect_cells_metadata(ws, max_row, max_col, merged_map)

        merges = []
        seen_ranges = set()
        for meta in merged_map.values():
            range_name = meta["range"]
            if range_name in seen_ranges or not meta["is_anchor"]:
                continue
            seen_ranges.add(range_name)
            merges.append(copy(meta))

        result[name] = {
            "dataframe": df,
            "excel_col_widths": col_widths,
            "row_heights": row_heights,
            "cells": cells_meta,
            "merges": merges,
        }

    return result
