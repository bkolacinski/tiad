"""Aplikacja Flask do konwersji plików XLSX na DOCX/PDF."""

import io
import json
import logging
import os

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)

from converters.docx_handler import df_to_docx
from converters.pdf_handler import df_to_pdf
from converters.xlsx_handler import read_xlsx

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")
app.logger.setLevel(logging.INFO)

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
MAX_DOCX_COLUMNS = 63

DEFAULT_SETTINGS = {
    "alignment": "left",
    "line_spacing": 1.15,
    "space_after": 6,
    "page_numbers": True,
}


def load_settings() -> dict:
    """Wczytuje ustawienia z pliku JSON lub zwraca domyslne."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """Zapisuje ustawienia do pliku JSON."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def _parse_float(value, default: float, lo: float, hi: float) -> float:
    """Parsuje wartosc float z ograniczeniem zakresu."""
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _parse_int(value, default: int, lo: int, hi: int) -> int:
    """Parsuje wartosc int z ograniczeniem zakresu."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _max_columns(sheets: dict) -> int:
    """Zwraca maksymalna liczbe kolumn sposrod wszystkich arkuszy."""
    cols = 0
    for payload in sheets.values():
        df = payload.get("dataframe")
        if df is not None and not df.empty:
            cols = max(cols, df.shape[1])
    return cols


@app.route("/")
def index():
    return render_template("index.html", settings=load_settings())


@app.route("/convert", methods=["POST"])
def convert():
    """Obsluguje konwersje pliku XLSX na wybrany format."""
    file = request.files.get("file")
    filename = file.filename if file and file.filename else ""
    if not file or not filename.lower().endswith(".xlsx"):
        flash("Wybierz plik w formacie .xlsx.")
        return redirect(url_for("index"))

    fmt = request.form.get("format") or "docx"
    alignment = request.form.get("alignment") or DEFAULT_SETTINGS["alignment"]
    if alignment not in {"left", "center", "right"}:
        alignment = DEFAULT_SETTINGS["alignment"]

    settings = {
        "title": (request.form.get("title") or "").strip(),
        "alignment": alignment,
        "line_spacing": _parse_float(
            request.form.get("line_spacing"),
            DEFAULT_SETTINGS["line_spacing"], 1.0, 3.0,
        ),
        "space_after": _parse_int(
            request.form.get("space_after"),
            DEFAULT_SETTINGS["space_after"], 0, 40,
        ),
        "page_numbers": "page_numbers" in request.form,
    }

    if "save_settings" in request.form:
        save_settings({k: v for k, v in settings.items() if k != "title"})
        flash("Ustawienia zapisane.")

    # Wczytanie pliku Excel
    try:
        sheets = read_xlsx(file)
    except Exception:
        app.logger.exception("Blad odczytu pliku: %s", filename)
        flash("Nie udalo sie odczytac pliku Excel.")
        return redirect(url_for("index"))

    if not sheets:
        flash("Plik nie zawiera danych do konwersji.")
        return redirect(url_for("index"))

    # Walidacja limitu kolumn dla formatu DOCX
    max_cols = _max_columns(sheets)
    if fmt == "docx" and max_cols > MAX_DOCX_COLUMNS:
        flash(
            f"Arkusz ma {max_cols} kolumn, a Word obsluguje maksymalnie "
            f"{MAX_DOCX_COLUMNS}. Wybierz format PDF."
        )
        return redirect(url_for("index"))

    # Konwersja
    buf = io.BytesIO()
    base = os.path.splitext(filename)[0]

    try:
        if fmt == "docx":
            df_to_docx(sheets, settings, buf)
            mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ext = "docx"
        elif fmt == "pdf":
            df_to_pdf(sheets, settings, buf)
            mimetype = "application/pdf"
            ext = "pdf"
        else:
            flash("Nieznany format wyjsciowy.")
            return redirect(url_for("index"))
    except Exception as exc:
        app.logger.exception("Blad konwersji: %s -> %s", filename, fmt)
        flash(f"Blad konwersji: {exc}")
        return redirect(url_for("index"))

    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"{base}.{ext}", mimetype=mimetype)


if __name__ == "__main__":
    app.run(debug=True)
