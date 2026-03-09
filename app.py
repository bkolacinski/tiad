import os
from flask import Flask, render_template, request, send_file, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
import tempfile
import uuid
from docx_converter import generate_docx
from pdf_converter import generate_pdf

app = Flask(__name__)
app.secret_key = 'super-secret-key-123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload size

ALLOWED_EXTENSIONS = {'xlsx'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_excel(file_path):
    # Wczytanie wszystkich arkuszy i połączenie lub po prostu wczytanie pierwszego arkusza.
    # Wczytujemy pierwszy arkusz z zachowaniem pustych wartości jako pustych ciągów
    df = pd.read_excel(file_path, sheet_name=0, dtype=str)
    df.fillna("", inplace=True)
    return df

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Brak pliku')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Nie wybrano pliku')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Odbiór ustawień z formularza
        doc_title = request.form.get('title', 'Dokument')
        page_orientation = request.form.get('orientation', 'portrait')
        font_size = request.form.get('fontSize', '12')
        page_numbers = request.form.get('pageNumbers', 'false') == 'true'
        spacing = request.form.get('spacing', 'normal')

        target_format = request.form.get('format', 'pdf')

        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        safe_filename = f"{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(filepath)

        try:
            df = read_excel(filepath)

            settings = {
                'title': doc_title,
                'orientation': page_orientation,
                'font_size': font_size,
                'page_numbers': page_numbers,
                'spacing': spacing
            }

            if target_format == 'docx':
                output_filename = f"{unique_id}.docx"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                generate_docx(df, settings, output_path)
                return send_file(output_path, as_attachment=True, download_name=f"{filename.rsplit('.', 1)[0]}.docx")

            elif target_format == 'pdf':
                output_filename = f"{unique_id}.pdf"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                generate_pdf(df, settings, output_path)
                return send_file(output_path, as_attachment=True, download_name=f"{filename.rsplit('.', 1)[0]}.pdf")
            else:
                flash('Nieznany format docelowy')
                return redirect(url_for('index'))

        except Exception as e:
            flash(f'Wystąpił błąd podczas przetwarzania pliku: {str(e)}')
            return redirect(url_for('index'))
        finally:
            # Nie usuwamy jeszcze output_path, bo send_file wysyła go z dysku.
            # Ale możemy usunąć wejściowy XLSX
            if os.path.exists(filepath):
                os.remove(filepath)

    else:
        flash('Niedozwolony format pliku. Proszę wgrać plik .xlsx')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
