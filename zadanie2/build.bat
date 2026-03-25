@echo off
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
echo [1/3] Installing PyInstaller...
python -m pip install pyinstaller==6.6.0 --quiet

echo.
echo [2/3] Building EXE (this may take a few minutes)...

python -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "RecipeVoiceFilter" ^
    --add-data "data;data" ^
    --add-data "models;models" ^
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
    --hidden-import "rapidfuzz" ^
    --collect-all "whisper" ^
    --collect-all "customtkinter" ^
    --collect-all "pl_core_news_sm" ^
    --collect-all "argostranslate" ^
    --collect-all "tiktoken_ext" ^
    --collect-data "spacy" ^
    app.py

echo.
echo [3/3] Copying data and models to dist...
xcopy /E /I /Y "data" "dist\RecipeVoiceFilter\data"
xcopy /E /I /Y "models" "dist\RecipeVoiceFilter\models"

echo.
echo ============================================
echo  DONE!
echo  Run: dist\RecipeVoiceFilter\RecipeVoiceFilter.exe
echo ============================================
pause
