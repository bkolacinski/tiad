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

DEFAULT_SETTINGS = {
    "alignment": "left",
    "line_spacing": 1.15,
    "space_after": 6,
    "page_numbers": True,
}


def load_settings() -> dict:
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except (OSError, json.JSONDecodeError):
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def parse_float(value: str, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def parse_int(value: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


@app.route("/")
def index():
    return render_template("index.html", settings=load_settings())


@app.route("/convert", methods=["POST"])
def convert():
    file = request.files.get("file")
    filename = file.filename if file and file.filename else ""
    if not file or not filename.lower().endswith(".xlsx"):
        flash("Wybierz plik w formacie .xlsx.")
        return redirect(url_for("index"))

    fmt = request.form.get("format") or "docx"
    alignment = request.form.get("alignment") or DEFAULT_SETTINGS["alignment"]
    if alignment not in {"left", "center", "right"}:
        alignment = DEFAULT_SETTINGS["alignment"]

    title = (request.form.get("title") or "").strip()
    line_spacing_value = request.form.get("line_spacing") or str(
        DEFAULT_SETTINGS["line_spacing"]
    )
    space_after_value = request.form.get("space_after") or str(
        DEFAULT_SETTINGS["space_after"]
    )

    settings = {
        "title": title,
        "alignment": alignment,
        "line_spacing": parse_float(
            line_spacing_value,
            DEFAULT_SETTINGS["line_spacing"],
            1.0,
            3.0,
        ),
        "space_after": parse_int(
            space_after_value,
            DEFAULT_SETTINGS["space_after"],
            0,
            40,
        ),
        "page_numbers": "page_numbers" in request.form,
    }

    if "save_settings" in request.form:
        persistable = {k: v for k, v in settings.items() if k != "title"}
        save_settings(persistable)
        flash("Ustawienia zostały zapisane.")

    try:
        sheets = read_xlsx(file)
    except Exception:
        app.logger.exception("Failed to read uploaded Excel file: %s", filename)
        flash("Nie udało się odczytać pliku Excel. Sprawdź, czy plik nie jest uszkodzony.")
        return redirect(url_for("index"))

    if not sheets:
        flash("Plik nie zawiera danych do konwersji.")
        return redirect(url_for("index"))

    buf = io.BytesIO()
    base = os.path.splitext(filename)[0]

    try:
        if fmt == "docx":
            df_to_docx(sheets, settings, buf)
            buf.seek(0)
            return send_file(
                buf,
                as_attachment=True,
                download_name=f"{base}.docx",
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        if fmt == "pdf":
            df_to_pdf(sheets, settings, buf)
            buf.seek(0)
            return send_file(
                buf,
                as_attachment=True,
                download_name=f"{base}.pdf",
                mimetype="application/pdf",
            )
    except Exception as exc:
        import traceback as _tb
        tb_str = _tb.format_exc()
        app.logger.exception(
            "Conversion failed for file %s with format %s and settings %s",
            filename,
            fmt,
            settings,
        )
        flash(f"BŁĄD: {exc}\n\n{tb_str}")
        return redirect(url_for("index"))

    flash("Nieznany format wyjściowy.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
