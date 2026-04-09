@echo off
setlocal
set KMP_DUPLICATE_LIB_OK=TRUE
set VENV_DIR=%TEMP%\RecipeVoiceFilter_build_venv
set PYTHON_EXE=%VENV_DIR%\Scripts\python.exe
set RELEASE_DIR=%~dp0release
set WORK_DIR=%~dp0build_release

echo ============================================
echo  Recipe Voice Filter - Build EXE
echo ============================================

:: Find Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo.
echo [1/4] Preparing local virtual environment...
for %%D in ("%RELEASE_DIR%\RecipeVoiceFilter" "%WORK_DIR%" "%~dp0build" "%~dp0build_fast" "%~dp0build_cached" "%~dp0release_fast" "%~dp0release_cached" "%~dp0dist") do (
    if exist %%~fD powershell -NoProfile -Command "if (Test-Path '%%~fD') { Remove-Item -LiteralPath '%%~fD' -Recurse -Force -ErrorAction SilentlyContinue }"
)
:: Zawsze czysc stare venv (rd jest pewniejsze niz PowerShell przy polamanym folderze)
if exist "%VENV_DIR%" (
    echo   Usuwanie poprzedniego venv...
    rd /s /q "%VENV_DIR%" 2>nul
)
:: Polamany venv: jest Scripts\python.exe ale brak pyvenv.cfg — wtedy rd wyzej tez czysci
if exist "%VENV_DIR%" (
    echo ERROR: Nie mozna usunac folderu venv: %VENV_DIR%
    echo Zamknij procesy uzywajace tego folderu i sprobuj ponownie.
    exit /b 1
)
echo   Tworzenie venv: %VENV_DIR%
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)
if not exist "%VENV_DIR%\pyvenv.cfg" (
    echo ERROR: Brak pyvenv.cfg po utworzeniu venv — sprawdz instalacje Pythona.
    exit /b 1
)

echo.
echo [2/4] Installing project dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip "setuptools<82" wheel --quiet
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip/setuptools
    exit /b 1
)
"%PYTHON_EXE%" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install project requirements
    exit /b 1
)
"%PYTHON_EXE%" setup.py
if errorlevel 1 (
    echo ERROR: setup.py failed
    exit /b 1
)
"%PYTHON_EXE%" smoke_test.py
if errorlevel 1 (
    echo ERROR: smoke_test.py failed
    exit /b 1
)

echo.
echo [3/4] Installing PyInstaller...
"%PYTHON_EXE%" -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    exit /b 1
)

echo.
echo [3.5/4] Zamykam ewentualnie dzialajacy RecipeVoiceFilter.exe (blokada plikow w release\)...
taskkill /IM RecipeVoiceFilter.exe /F >nul 2>&1
ping -n 3 127.0.0.1 >nul

echo.
echo [4/4] Building EXE (this may take a few minutes)...

"%PYTHON_EXE%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onedir ^
    --windowed ^
    --distpath "%RELEASE_DIR%" ^
    --workpath "%WORK_DIR%" ^
    --name "RecipeVoiceFilter" ^
    --add-data "data;data" ^
    --add-data "models;models" ^
    --add-data "translation_packages;translation_packages" ^
    --hidden-import "whisper" ^
    --hidden-import "whisper.audio" ^
    --hidden-import "sklearn.feature_extraction.text" ^
    --hidden-import "sklearn.metrics.pairwise" ^
    --hidden-import "spacy" ^
    --hidden-import "pl_core_news_sm" ^
    --hidden-import "customtkinter" ^
    --hidden-import "sounddevice" ^
    --hidden-import "soundfile" ^
    --hidden-import "argostranslate" ^
    --hidden-import "ctranslate2" ^
    --hidden-import "rapidfuzz" ^
    --hidden-import "certifi" ^
    --hidden-import "vosk" ^
    --hidden-import "transformers" ^
    --hidden-import "transformers.models.wav2vec2" ^
    --collect-data "certifi" ^
    --collect-submodules "sklearn" ^
    --collect-submodules "scipy" ^
    --collect-submodules "argostranslate" ^
    --collect-all "whisper" ^
    --collect-all "customtkinter" ^
    --collect-all "pl_core_news_sm" ^
    --collect-all "argostranslate" ^
    --collect-all "ctranslate2" ^
    --collect-all "tiktoken_ext" ^
    --collect-all "vosk" ^
    --collect-all "transformers" ^
    --collect-data "spacy" ^
    app.py
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

echo.
echo Cleaning temporary build directories...
for %%D in ("%WORK_DIR%" "%~dp0build" "%~dp0build_fast" "%~dp0build_cached" "%~dp0release_fast" "%~dp0release_cached" "%~dp0dist" "%~dp0__pycache__" "%~dp0.venv") do (
    if exist %%~fD powershell -NoProfile -Command "if (Test-Path '%%~fD') { Remove-Item -LiteralPath '%%~fD' -Recurse -Force -ErrorAction SilentlyContinue }"
)
if exist "%VENV_DIR%" powershell -NoProfile -Command "if (Test-Path '%VENV_DIR%') { Remove-Item -LiteralPath '%VENV_DIR%' -Recurse -Force -ErrorAction SilentlyContinue }"

echo.
echo ============================================
echo  DONE!
echo  Run: release\RecipeVoiceFilter\RecipeVoiceFilter.exe
echo ============================================
