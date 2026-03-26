@echo off
setlocal

if not exist .venv\Scripts\python.exe (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-build.txt

python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name KonwerterXLSX ^
    --collect-data docx ^
    app.py

echo.
echo Gotowe. Plik wykonywalny znajdziesz w katalogu dist\KonwerterXLSX.exe
