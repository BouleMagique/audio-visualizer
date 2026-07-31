#!/usr/bin/env bash
# =============================================================================
# Audio Visualizer — Build macOS (.app)
# =============================================================================
# Produit dist/AudioVisualizer.app + dist/AudioVisualizer-macos-<arch>.zip
# Python + Qt + ModernGL + shaders bundles. ffmpeg reste une dep systeme (PATH).
#
# Usage :
#   bash build-macos.sh
#
# Python 3.10+ requis (PySide6). Si le python3 du systeme est trop vieux
# (macOS livre encore 3.9), le script recupere un interpreteur via uv, installe
# dans ~/.local/bin — rien n'est touche hors du home, pas de sudo.
# =============================================================================
set -euo pipefail

APP="AudioVisualizer"
ENTRY="main.py"
ARCH="$(uname -m)"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HOME/.cache/audio-visualizer-build-macos"
OUT="$SRC/dist"
PYREQ="3.12"

echo ""
echo "=== Audio Visualizer — Build macOS (.app) ==="
echo "Source : $SRC"
echo "Build  : $BUILD"
echo "Arch   : $ARCH"
echo ""

# --- Python 3.10+ -----------------------------------------------------------
PY=""
if command -v python3 >/dev/null &&
   python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    PY="$(command -v python3)"
else
    echo "[..] python3 systeme trop ancien ou absent — passage par uv..."
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    if [ ! -x "$UV" ]; then
        echo "[..] Installation de uv (~/.local/bin, sans sudo)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
        UV="$HOME/.local/bin/uv"
    fi
    "$UV" python install "$PYREQ" >/dev/null
    PY="$("$UV" python find "$PYREQ")"
fi
echo "[OK] Python $("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])') — $PY"

# --- Copie source ------------------------------------------------------------
echo "[..] Copie des sources..."
mkdir -p "$BUILD/src"
rsync -a --delete \
    --exclude venv --exclude .git --exclude __pycache__ --exclude '.pytest_cache' \
    --exclude dist --exclude build \
    "$SRC"/ "$BUILD/src"/
cd "$BUILD/src"

# --- Venv + deps + PyInstaller ----------------------------------------------
[ -d venv ] || "$PY" -m venv venv
echo "[..] Installation des dependances + PyInstaller (Qt : ca peut etre long)..."
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
venv/bin/pip install pyinstaller -q
echo "[OK] Environnement de build pret"

# --- PyInstaller : bundle .app ----------------------------------------------
echo "[..] PyInstaller..."
rm -rf dist build
venv/bin/pyinstaller --windowed --name "$APP" \
    --osx-bundle-identifier "com.boulemagique.audiovisualizer" \
    --add-data "$BUILD/src/render/shaders:render/shaders" \
    --collect-all moderngl --collect-all glcontext \
    "$ENTRY"
[ -d "dist/$APP.app" ] || { echo "[ERREUR] dist/$APP.app absent." >&2; exit 1; }
echo "[OK] Bundle genere : dist/$APP.app"

# --- Livrable ----------------------------------------------------------------
mkdir -p "$OUT"
rm -rf "$OUT/$APP.app" "$OUT/$APP-macos-$ARCH.zip"
cp -R "dist/$APP.app" "$OUT/$APP.app"
# ditto (et pas zip) : preserve les liens symboliques et les metadonnees du bundle,
# sinon le .app decompresse ne se lance pas.
ditto -c -k --keepParent "$OUT/$APP.app" "$OUT/$APP-macos-$ARCH.zip"

echo ""
echo "=== TERMINE ==="
echo "Livrable : $OUT/$APP-macos-$ARCH.zip"
echo "Test     : open '$OUT/$APP.app'"
echo ""
echo "Note : bundle NON signe. Au premier lancement, Gatekeeper le bloque —"
echo "       xattr -dr com.apple.quarantine '$APP.app' puis clic droit > Ouvrir."
echo "       ffmpeg doit etre dans le PATH pour l'export video."
