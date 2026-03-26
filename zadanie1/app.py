import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from converter_service import (
    DEFAULT_SETTINGS,
    ConversionError,
    convert_xlsx_file,
    get_app_data_dir,
    get_default_output_path,
    load_settings,
    save_settings,
)

APP_TITLE = "Konwerter XLSX"
WINDOW_SIZE = "760x540"

ALIGNMENT_OPTIONS = {
    "left": "Do lewej",
    "center": "Wycentrowane",
    "right": "Do prawej",
}

FORMAT_LABELS = {
    "docx": "DOCX",
    "pdf": "PDF",
}


def configure_logging() -> None:
    """
    Configures file logging for the desktop application.
    :return: None
    """
    log_dir = get_app_data_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "app.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        encoding="utf-8",
    )


class DesktopConverterApp(tk.Tk):
    """
    Main tkinter window for the XLSX desktop converter.
    """

    def __init__(self) -> None:
        super().__init__()
        self.saved_settings = load_settings()
        self._configure_window()
        self._create_variables()
        self._build_ui()
        self._apply_saved_settings()

    def _configure_window(self) -> None:
        """
        Configures window geometry and base styles.
        :return: None
        """
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(700, 500)
        self.configure(bg="#eef1f4")

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Root.TFrame", background="#eef1f4")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Section.TLabelframe", background="#ffffff")
        style.configure("Section.TLabelframe.Label", background="#ffffff")
        style.configure("Status.TLabel", background="#eef1f4", foreground="#1f2933")
        style.configure("Hint.TLabel", background="#ffffff", foreground="#5b6570")
        style.configure("Title.TLabel", background="#eef1f4", foreground="#16202a")
        style.configure("Subtitle.TLabel", background="#eef1f4", foreground="#52606d")

    def _create_variables(self) -> None:
        """
        Creates tkinter variables used by the form.
        :return: None
        """
        self.input_path_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.format_var = tk.StringVar(value="docx")
        self.alignment_var = tk.StringVar(value=DEFAULT_SETTINGS["alignment"])
        self.line_spacing_var = tk.StringVar()
        self.space_after_var = tk.StringVar()
        self.page_numbers_var = tk.BooleanVar(value=DEFAULT_SETTINGS["page_numbers"])
        self.status_var = tk.StringVar(
            value="Wybierz plik XLSX, ustaw parametry i zapisz wynik jako DOCX lub PDF."
        )

    def _build_ui(self) -> None:
        """
        Builds the layout of the application window.
        :return: None
        """
        root = ttk.Frame(self, padding=24, style="Root.TFrame")
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text=APP_TITLE,
            style="Title.TLabel",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Desktopowa aplikacja do konwersji arkuszy Excel z zachowaniem formatowania.",
            style="Subtitle.TLabel",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        card = ttk.Frame(root, padding=20, style="Card.TFrame")
        card.grid(row=1, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)

        self._build_file_section(card)
        self._build_output_section(card)
        self._build_settings_section(card)
        self._build_actions(card)

        status = ttk.Label(
            root,
            textvariable=self.status_var,
            style="Status.TLabel",
            wraplength=680,
            font=("Segoe UI", 10),
        )
        status.grid(row=2, column=0, sticky="ew", pady=(14, 0))

    def _build_file_section(self, parent) -> None:
        """
        Builds the file selection section.
        :param parent: Parent container.
        :return: None
        """
        section = ttk.LabelFrame(
            parent,
            text="Plik wejściowy",
            padding=16,
            style="Section.TLabelframe",
        )
        section.grid(row=0, column=0, sticky="ew")
        section.columnconfigure(0, weight=1)

        ttk.Entry(
            section,
            textvariable=self.input_path_var,
            state="readonly",
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 12))

        self.browse_button = ttk.Button(
            section,
            text="Wybierz plik",
            command=self.choose_input_file,
        )
        self.browse_button.grid(row=0, column=1, sticky="e")

        ttk.Label(
            section,
            text="Obsługiwany format wejściowy: .xlsx",
            style="Hint.TLabel",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_output_section(self, parent) -> None:
        """
        Builds the output settings section.
        :param parent: Parent container.
        :return: None
        """
        section = ttk.LabelFrame(
            parent,
            text="Format wyjściowy",
            padding=16,
            style="Section.TLabelframe",
        )
        section.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        section.columnconfigure(1, weight=1)

        ttk.Label(section, text="Typ pliku:", font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w"
        )
        radio_frame = ttk.Frame(section, style="Card.TFrame")
        radio_frame.grid(row=0, column=1, sticky="w")
        for col, (fmt, label) in enumerate(FORMAT_LABELS.items()):
            ttk.Radiobutton(
                radio_frame,
                text=label,
                value=fmt,
                variable=self.format_var,
            ).grid(row=0, column=col, sticky="w", padx=(0, 18))

        ttk.Label(section, text="Tytuł dokumentu:", font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", pady=(14, 0)
        )
        ttk.Entry(
            section,
            textvariable=self.title_var,
            font=("Segoe UI", 10),
        ).grid(row=1, column=1, sticky="ew", pady=(14, 0))

    def _build_settings_section(self, parent) -> None:
        """
        Builds the document settings section.
        :param parent: Parent container.
        :return: None
        """
        section = ttk.LabelFrame(
            parent,
            text="Ustawienia strony",
            padding=16,
            style="Section.TLabelframe",
        )
        section.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        section.columnconfigure(1, weight=1)
        section.columnconfigure(3, weight=1)

        ttk.Label(section, text="Wyrównanie:", font=("Segoe UI", 10)).grid(
            row=0, column=0, sticky="w"
        )
        alignment_frame = ttk.Frame(section, style="Card.TFrame")
        alignment_frame.grid(row=0, column=1, sticky="w", padx=(10, 0))
        for col, (value, label) in enumerate(ALIGNMENT_OPTIONS.items()):
            ttk.Radiobutton(
                alignment_frame,
                text=label,
                value=value,
                variable=self.alignment_var,
            ).grid(row=0, column=col, sticky="w", padx=(0, 14))

        ttk.Label(section, text="Interlinia:", font=("Segoe UI", 10)).grid(
            row=1, column=0, sticky="w", pady=(14, 0)
        )
        ttk.Spinbox(
            section,
            from_=1.0,
            to=3.0,
            increment=0.05,
            textvariable=self.line_spacing_var,
            width=10,
            font=("Segoe UI", 10),
        ).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(14, 0))

        ttk.Label(section, text="Odstęp po akapicie (pt):", font=("Segoe UI", 10)).grid(
            row=1, column=2, sticky="w", pady=(14, 0)
        )
        ttk.Spinbox(
            section,
            from_=0,
            to=40,
            increment=1,
            textvariable=self.space_after_var,
            width=10,
            font=("Segoe UI", 10),
        ).grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(14, 0))

        ttk.Checkbutton(
            section,
            text="Dodaj numerację stron",
            variable=self.page_numbers_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(16, 0))

        ttk.Label(
            section,
            text="DOCX obsługuje maksymalnie 63 kolumny w pojedynczym arkuszu. Szersze arkusze zapisz jako PDF.",
            style="Hint.TLabel",
            wraplength=620,
            font=("Segoe UI", 9),
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(10, 0))

    def _build_actions(self, parent) -> None:
        """
        Builds action buttons.
        :param parent: Parent container.
        :return: None
        """
        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(20, 0))
        actions.columnconfigure(0, weight=1)

        left = ttk.Frame(actions, style="Card.TFrame")
        left.grid(row=0, column=0, sticky="w")
        right = ttk.Frame(actions, style="Card.TFrame")
        right.grid(row=0, column=1, sticky="e")

        self.save_button = ttk.Button(
            left,
            text="Zapisz ustawienia",
            command=self.persist_settings,
        )
        self.save_button.grid(row=0, column=0, sticky="w")

        self.reset_button = ttk.Button(
            left,
            text="Przywróć domyślne",
            command=self.reset_defaults,
        )
        self.reset_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.convert_button = ttk.Button(
            right,
            text="Konwertuj i zapisz",
            command=self.convert_document,
        )
        self.convert_button.grid(row=0, column=0, sticky="e")

    def _apply_saved_settings(self) -> None:
        """
        Loads saved settings into the form.
        :return: None
        """
        self.alignment_var.set(self.saved_settings.get("alignment", "left"))
        self.line_spacing_var.set(str(self.saved_settings.get("line_spacing", 1.15)))
        self.space_after_var.set(str(self.saved_settings.get("space_after", 6)))
        self.page_numbers_var.set(bool(self.saved_settings.get("page_numbers", True)))

    def choose_input_file(self) -> None:
        """
        Opens a file picker for the XLSX source file.
        :return: None
        """
        initial_dir = self._input_parent_or_cwd()
        path = filedialog.askopenfilename(
            title="Wybierz plik XLSX",
            initialdir=initial_dir,
            filetypes=[("Arkusz Excel", "*.xlsx")],
        )
        if not path:
            return
        self.input_path_var.set(path)
        self.status_var.set(f"Wybrano plik: {path}")

    def persist_settings(self) -> None:
        """
        Saves form settings to the user configuration file.
        :return: None
        """
        try:
            save_settings(self._collect_settings())
        except OSError:
            logging.exception("Nie udało się zapisać ustawień.")
            message = "Nie udało się zapisać ustawień."
            self.status_var.set(message)
            messagebox.showerror("Błąd zapisu", message)
            return

        self.saved_settings = load_settings()
        self.status_var.set(
            "Ustawienia zapisano. Przy następnym uruchomieniu zostaną wczytane automatycznie."
        )
        messagebox.showinfo("Ustawienia zapisane", "Ustawienia zostały zapisane.")

    def reset_defaults(self) -> None:
        """
        Resets the form to default settings.
        :return: None
        """
        self.alignment_var.set(DEFAULT_SETTINGS["alignment"])
        self.line_spacing_var.set(str(DEFAULT_SETTINGS["line_spacing"]))
        self.space_after_var.set(str(DEFAULT_SETTINGS["space_after"]))
        self.page_numbers_var.set(bool(DEFAULT_SETTINGS["page_numbers"]))
        self.status_var.set("Przywrócono domyślne ustawienia formularza.")

    def convert_document(self) -> None:
        """
        Validates input, asks for an output path and performs the conversion.
        :return: None
        """
        input_value = self.input_path_var.get().strip()
        if not input_value:
            messagebox.showwarning("Brak pliku", "Najpierw wybierz plik XLSX do konwersji.")
            return

        input_path = Path(input_value)
        fmt = self.format_var.get()
        default_output = get_default_output_path(input_path, fmt)
        output_value = filedialog.asksaveasfilename(
            title="Zapisz plik wynikowy",
            initialdir=str(default_output.parent),
            initialfile=default_output.name,
            defaultextension=f".{fmt}",
            filetypes=[(FORMAT_LABELS[fmt], f"*.{fmt}")],
        )
        if not output_value:
            self.status_var.set("Zapis pliku wynikowego został anulowany.")
            return

        output_path = Path(output_value)
        if output_path.suffix.lower() != f".{fmt}":
            output_path = output_path.with_suffix(f".{fmt}")

        self._set_busy(True)
        self.status_var.set("Trwa konwersja pliku. To może potrwać chwilę...")
        self.update_idletasks()

        try:
            final_path = convert_xlsx_file(
                input_path=input_path,
                output_path=output_path,
                settings=self._collect_settings(),
                fmt=fmt,
            )
        except ConversionError as exc:
            logging.warning("Konwersja anulowana: %s", exc)
            self.status_var.set(str(exc))
            messagebox.showerror("Konwersja nieudana", str(exc))
        except Exception:
            logging.exception("Nieoczekiwany błąd aplikacji.")
            message = (
                "Wystąpił nieoczekiwany błąd. Szczegóły zapisano w logu aplikacji."
            )
            self.status_var.set(message)
            messagebox.showerror("Błąd", message)
        else:
            message = f"Plik został zapisany: {final_path}"
            self.status_var.set(message)
            messagebox.showinfo("Konwersja zakończona", message)
        finally:
            self._set_busy(False)

    def _collect_settings(self) -> dict:
        """
        Collects current form values into a settings dictionary.
        :return: Settings dictionary.
        """
        return {
            "title": self.title_var.get(),
            "alignment": self.alignment_var.get(),
            "line_spacing": self.line_spacing_var.get(),
            "space_after": self.space_after_var.get(),
            "page_numbers": self.page_numbers_var.get(),
        }

    def _input_parent_or_cwd(self) -> str:
        """
        Returns the directory of the selected file or current working directory.
        :return: Directory path as string.
        """
        current = self.input_path_var.get().strip()
        if current:
            return str(Path(current).expanduser().resolve().parent)
        return str(Path.cwd())

    def _set_busy(self, busy: bool) -> None:
        """
        Enables or disables controls during conversion.
        :param busy: True to disable actions, False to enable them.
        :return: None
        """
        state = "disabled" if busy else "normal"
        for widget in (
            self.browse_button,
            self.save_button,
            self.reset_button,
            self.convert_button,
        ):
            widget.configure(state=state)
        self.configure(cursor="wait" if busy else "")


def main() -> None:
    """
    Starts the desktop application.
    :return: None
    """
    configure_logging()
    app = DesktopConverterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
