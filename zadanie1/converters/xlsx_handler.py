"""Odczyt pliku Excel (XLSX) z ekstrakcja danych i metadanych formatowania."""

import io
from copy import copy

import openpyxl
import pandas as pd
from openpyxl.utils import get_column_letter

DEFAULT_COLUMN_WIDTH = 8.43
DEFAULT_ROW_HEIGHT = 15.0
DEFAULT_FONT_SIZE = 11.0

# Mapa kolorow indeksowanych z Excela (standardowa paleta)
_INDEXED_COLORS = {
    0: "#000000", 1: "#FFFFFF", 2: "#FF0000", 3: "#00FF00",
    4: "#0000FF", 5: "#FFFF00", 6: "#FF00FF", 7: "#00FFFF",
    8: "#000000", 9: "#FFFFFF", 10: "#FF0000", 11: "#00FF00",
    12: "#0000FF", 13: "#FFFF00", 14: "#FF00FF", 15: "#00FFFF",
    16: "#800000", 17: "#008000", 18: "#000080", 19: "#808000",
    20: "#800080", 21: "#008080", 22: "#C0C0C0", 23: "#808080",
}


def _normalize_color(value) -> str | None:
    """Konwertuje kolor z openpyxl (RGB lub indexed) na format hex #RRGGBB."""
    if not value:
        return None

    rgb = getattr(value, "rgb", None)
    if rgb and isinstance(rgb, str):
        rgb = rgb.upper()
        if len(rgb) == 8:
            rgb = rgb[2:]  # Pominiecie kanalu alfa
        if len(rgb) == 6:
            return f"#{rgb}"

    indexed = getattr(value, "indexed", None)
    if indexed is not None:
        return _INDEXED_COLORS.get(indexed)

    return None


def _extract_fill_color(fill) -> str | None:
    """Pobiera kolor wypelnienia komorki."""
    if not fill or getattr(fill, "patternType", None) not in {"solid", "gray125"}:
        return None
    return _normalize_color(getattr(fill, "fgColor", None)) or \
           _normalize_color(getattr(fill, "start_color", None))


def _extract_border(border) -> dict:
    """Pobiera informacje o obramowaniu komorki."""
    if not border:
        return {}
    result = {}
    for side_name in ("left", "right", "top", "bottom"):
        side = getattr(border, side_name, None)
        if side and getattr(side, "style", None):
            result[side_name] = {
                "style": side.style,
                "color": _normalize_color(getattr(side, "color", None)),
            }
    return result


def _extract_alignment(alignment) -> dict:
    """Pobiera informacje o wyrownaniu tekstu."""
    if not alignment:
        return {}
    return {
        "horizontal": alignment.horizontal,
        "vertical": alignment.vertical,
        "wrap_text": bool(alignment.wrap_text),
    }


def _extract_font(font) -> dict:
    """Pobiera informacje o czcionce."""
    if not font:
        return {}
    return {
        "name": font.name,
        "size": float(font.sz) if font.sz is not None else DEFAULT_FONT_SIZE,
        "bold": bool(font.b),
        "italic": bool(font.i),
        "color": _normalize_color(getattr(font, "color", None)),
    }


def _extract_cell_style(cell) -> dict:
    """Zbiera wszystkie style komorki w jeden slownik."""
    return {
        "font": _extract_font(cell.font),
        "fill": {"color": _extract_fill_color(cell.fill)},
        "border": _extract_border(cell.border),
        "alignment": _extract_alignment(cell.alignment),
        "number_format": cell.number_format or "General",
    }


def _collect_merged_map(ws) -> dict[tuple[int, int], dict]:
    """Tworzy mape scalonych komorek dla arkusza."""
    merged_map: dict[tuple[int, int], dict] = {}

    for merged_range in ws.merged_cells.ranges:
        min_col, min_row = merged_range.min_col, merged_range.min_row
        max_col, max_row = merged_range.max_col, merged_range.max_row

        meta = {
            "range": str(merged_range),
            "start_row": min_row - 1, "start_col": min_col - 1,
            "end_row": max_row - 1, "end_col": max_col - 1,
            "is_anchor": True,
        }

        merged_map[(min_row, min_col)] = meta

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                if row == min_row and col == min_col:
                    continue
                hidden_meta = meta.copy()
                hidden_meta["is_anchor"] = False
                merged_map[(row, col)] = hidden_meta

    return merged_map


def _collect_column_widths(ws, n_cols: int) -> list[float]:
    """Pobiera szerokosci kolumn z arkusza."""
    widths = []
    for i in range(1, n_cols + 1):
        dim = ws.column_dimensions.get(get_column_letter(i))
        widths.append(float(dim.width or DEFAULT_COLUMN_WIDTH) if dim else DEFAULT_COLUMN_WIDTH)
    return widths


def _build_dataframe(ws, n_rows: int, n_cols: int) -> pd.DataFrame:
    """Buduje DataFrame z wartosciami komorek arkusza."""
    rows = []
    for row_idx in range(1, n_rows + 1):
        row = []
        for col_idx in range(1, n_cols + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            row.append(str(val) if val is not None else "")
        rows.append(row)
    return pd.DataFrame(rows, dtype=str).fillna("")


def _collect_cells_metadata(ws, n_rows: int, n_cols: int,
                            merged_map: dict[tuple[int, int], dict]) -> list[list[dict]]:
    """Zbiera metadane (styl, scalenie, wartosc) dla kazdej komorki arkusza."""
    cells_meta = []

    for row_idx in range(1, n_rows + 1):
        row_meta = []
        for col_idx in range(1, n_cols + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            merge_meta = merged_map.get((row_idx, col_idx))

            if merge_meta and not merge_meta["is_anchor"]:
                row_meta.append({
                    "value": "",
                    "style": {},
                    "merge": merge_meta,
                    "hidden_by_merge": True,
                })
                continue

            val = str(cell.value) if cell.value is not None else ""
            row_meta.append({
                "value": val,
                "style": _extract_cell_style(cell),
                "merge": merge_meta,
                "hidden_by_merge": False,
            })
        cells_meta.append(row_meta)

    return cells_meta


def read_xlsx(file) -> dict[str, dict]:
    """
    Wczytuje plik Excel i wyciaga dane oraz formatowanie dla kazdego arkusza.

    Zwraca slownik: nazwa_arkusza -> {
        dataframe: DataFrame z wartosciami komorek,
        excel_col_widths: szerokosci kolumn w jednostkach Excela,
        cells: metadane komorek (style, scalenia),
        merges: lista opisow scalonych zakresow,
    }
    """
    wb = openpyxl.load_workbook(io.BytesIO(file.read()), data_only=True)
    result = {}

    for name in wb.sheetnames:
        ws = wb[name]
        n_rows = ws.max_row or 0
        n_cols = ws.max_column or 0

        if n_rows == 0 or n_cols == 0:
            continue

        merged_map = _collect_merged_map(ws)
        df = _build_dataframe(ws, n_rows, n_cols)
        cells_meta = _collect_cells_metadata(ws, n_rows, n_cols, merged_map)
        col_widths = _collect_column_widths(ws, n_cols)

        # Wyciagniecie unikalnych scalen (tylko anchory)
        seen = set()
        merges = []
        for meta in merged_map.values():
            if meta["is_anchor"] and meta["range"] not in seen:
                seen.add(meta["range"])
                merges.append(copy(meta))

        result[name] = {
            "dataframe": df,
            "excel_col_widths": col_widths,
            "cells": cells_meta,
            "merges": merges,
        }

    return result
