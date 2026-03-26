import io
import json
import logging
import os
from pathlib import Path

from converters.docx_handler import df_to_docx
from converters.pdf_handler import df_to_pdf
from converters.xlsx_handler import read_xlsx

LOGGER = logging.getLogger(__name__)
APP_NAME = "KonwerterXLSX"
MAX_DOCX_COLUMNS = 63

DEFAULT_SETTINGS = {
    "alignment": "left",
    "line_spacing": 1.15,
    "space_after": 6,
    "page_numbers": True,
}


class ConversionError(Exception):
    """
    Error intended to be shown directly to the user.
    """


def get_app_data_dir() -> Path:
    """
    Returns a writable directory for application data and settings.
    :return: Path to the application data directory.
    """
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / APP_NAME

    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / APP_NAME

    return Path.home() / f".{APP_NAME.lower()}"


def get_settings_file() -> Path:
    """
    Returns the path to the settings file.
    :return: Path to settings.json.
    """
    return get_app_data_dir() / "settings.json"


def load_settings() -> dict:
    """
    Loads settings from disk and merges them with defaults.
    :return: Settings dictionary.
    """
    settings_file = get_settings_file()
    if settings_file.exists():
        try:
            with settings_file.open(encoding="utf-8") as handle:
                saved = json.load(handle)
            return {**DEFAULT_SETTINGS, **saved}
        except (OSError, json.JSONDecodeError):
            LOGGER.exception("Nie udało się odczytać ustawień z %s", settings_file)
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """
    Saves persistent settings to disk.
    :param settings: Settings dictionary.
    :return: None
    """
    sanitized = sanitize_settings(settings)
    persistent = {key: value for key, value in sanitized.items() if key != "title"}
    settings_file = get_settings_file()
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    with settings_file.open("w", encoding="utf-8") as handle:
        json.dump(persistent, handle, indent=2, ensure_ascii=False)


def sanitize_settings(raw_settings: dict) -> dict:
    """
    Normalizes and validates form settings.
    :param raw_settings: Raw settings from the GUI.
    :return: Sanitized settings dictionary.
    """
    alignment = raw_settings.get("alignment") or DEFAULT_SETTINGS["alignment"]
    if alignment not in {"left", "center", "right"}:
        alignment = DEFAULT_SETTINGS["alignment"]

    return {
        "title": (raw_settings.get("title") or "").strip(),
        "alignment": alignment,
        "line_spacing": _parse_float(
            raw_settings.get("line_spacing"),
            DEFAULT_SETTINGS["line_spacing"],
            1.0,
            3.0,
        ),
        "space_after": _parse_int(
            raw_settings.get("space_after"),
            DEFAULT_SETTINGS["space_after"],
            0,
            40,
        ),
        "page_numbers": _parse_bool(
            raw_settings.get("page_numbers"),
            DEFAULT_SETTINGS["page_numbers"],
        ),
    }


def get_default_output_path(input_path: str | Path, fmt: str) -> Path:
    """
    Returns the default output path next to the input file.
    :param input_path: Path to the input XLSX file.
    :param fmt: Output format.
    :return: Suggested output path.
    """
    source = Path(input_path)
    return source.with_suffix(f".{fmt}")


def convert_xlsx_file(
    input_path: str | Path,
    output_path: str | Path,
    settings: dict,
    fmt: str | None = None,
) -> Path:
    """
    Converts an XLSX file to DOCX or PDF and writes it to disk.
    :param input_path: Path to the source XLSX file.
    :param output_path: Target file path.
    :param settings: Conversion settings.
    :param fmt: Explicit output format. If omitted, it is inferred from output_path.
    :return: Final output path.
    """
    source = Path(input_path)
    target = Path(output_path)
    output_format = (fmt or target.suffix.lstrip(".")).lower()

    if not source.exists():
        raise ConversionError("Wybrany plik nie istnieje.")
    if source.suffix.lower() != ".xlsx":
        raise ConversionError("Wybierz plik w formacie .xlsx.")
    if output_format not in {"docx", "pdf"}:
        raise ConversionError("Nieznany format wyjściowy.")

    normalized_settings = sanitize_settings(settings)

    try:
        with source.open("rb") as handle:
            sheets = read_xlsx(handle)
    except Exception as exc:
        LOGGER.exception("Błąd odczytu pliku: %s", source)
        raise ConversionError("Nie udało się odczytać pliku Excel.") from exc

    if not sheets:
        raise ConversionError("Plik nie zawiera danych do konwersji.")

    max_cols = _max_columns(sheets)
    if output_format == "docx" and max_cols > MAX_DOCX_COLUMNS:
        raise ConversionError(
            f"Arkusz ma {max_cols} kolumn, a Word obsługuje maksymalnie "
            f"{MAX_DOCX_COLUMNS}. Wybierz format PDF."
        )

    buffer = io.BytesIO()

    try:
        if output_format == "docx":
            df_to_docx(sheets, normalized_settings, buffer)
        else:
            df_to_pdf(sheets, normalized_settings, buffer)
    except Exception as exc:
        LOGGER.exception("Błąd konwersji: %s -> %s", source, output_format)
        raise ConversionError(f"Błąd konwersji: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    buffer.seek(0)
    with target.open("wb") as handle:
        handle.write(buffer.getvalue())

    return target


def _parse_float(value, default: float, lo: float, hi: float) -> float:
    """
    Parses and clamps float values.
    :param value: Raw value.
    :param default: Default value.
    :param lo: Minimum allowed value.
    :param hi: Maximum allowed value.
    :return: Parsed float.
    """
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _parse_int(value, default: int, lo: int, hi: int) -> int:
    """
    Parses and clamps integer values.
    :param value: Raw value.
    :param default: Default value.
    :param lo: Minimum allowed value.
    :param hi: Maximum allowed value.
    :return: Parsed integer.
    """
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _parse_bool(value, default: bool) -> bool:
    """
    Converts raw values to boolean.
    :param value: Raw value.
    :param default: Default value.
    :return: Parsed boolean.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _max_columns(sheets: dict) -> int:
    """
    Returns the maximum number of columns across all sheets.
    :param sheets: Workbook payload returned by read_xlsx().
    :return: Maximum number of columns.
    """
    cols = 0
    for payload in sheets.values():
        dataframe = payload.get("dataframe")
        if dataframe is not None and not dataframe.empty:
            cols = max(cols, dataframe.shape[1])
    return cols
