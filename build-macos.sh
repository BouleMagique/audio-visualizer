#!/usr/bin/env bash
# =============================================================================
# Audio Visualizer — Build macOS (.app)
# =============================================================================
# Produit dist/AudioVisualizer.app + dist/AudioVisualizer-macos-<arch>.zip
# Python + Qt + ModernGL + shaders bundles. ffmpeg reste une dep systeme (PATH).
#
# Usage :
#   bash build-macos.sh                     # arch native (arm64 sur Apple Silicon)
#   TARGET_ARCH=x86_64 bash build-macos.sh  # Mac Intel (ex. Mac Pro 2013, Monterey)
#                                           # depuis un Mac Apple Silicon, via Rosetta
#
# Build Intel : cible macOS 12, donc PySide6 6.9.3 (dernier a supporter Monterey,
# requirements-macos-intel.txt) et un Python x86_64 recupere par uv.
#
# Python 3.10+ requis (PySide6). Si le python3 du systeme est trop vieux
# (macOS livre encore 3.9), le script recupere un interpreteur via uv, installe
# dans ~/.local/bin — rien n'est touche hors du home, pas de sudo.
# =============================================================================
set -euo pipefail

APP="AudioVisualizer"
ENTRY="main.py"
ARCH="${TARGET_ARCH:-$(uname -m)}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HOME/.cache/audio-visualizer-build-macos-$ARCH"
REQ="requirements.txt"
RUN=""
if [ "$ARCH" = "x86_64" ]; then
    REQ="requirements-macos-intel.txt"
    export MACOSX_DEPLOYMENT_TARGET=12.0
    # Sur Apple Silicon, tout le build tourne sous Rosetta
    [ "$(uname -m)" = "arm64" ] && RUN="arch -x86_64"
fi
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
# cross-arch : toujours un Python uv de la bonne arch
if [ "$ARCH" = "$(uname -m)" ] && command -v python3 >/dev/null &&
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
    PYKEY="$PYREQ"
    [ "$ARCH" != "$(uname -m)" ] && PYKEY="cpython-$PYREQ-macos-$ARCH-none"
    "$UV" python install "$PYKEY" >/dev/null
    PY="$("$UV" python find "$PYKEY")"
fi
[ -z "$PY" ] && { echo "[ERREUR] Python introuvable." >&2; exit 1; }
echo "[OK] Python $("$PY" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])') — $PY"

# --- Copie source ------------------------------------------------------------
echo "[..] Copie des sources..."
mkdir -p "$BUILD/src"
rsync -a --delete \
    --exclude venv --exclude .git --exclude __pycache__ --exclude '.pytest_cache' \
    --exclude dist --exclude build --exclude wheels-mac12 \
    "$SRC"/ "$BUILD/src"/
cd "$BUILD/src"

# --- Venv + deps + PyInstaller ----------------------------------------------
[ -d venv ] || $RUN "$PY" -m venv venv
echo "[..] Installation des dependances + PyInstaller (Qt : ca peut etre long)..."
$RUN venv/bin/pip install --upgrade pip -q
if [ "$ARCH" = "x86_64" ]; then
    # pip choisit les wheels selon le macOS de l'HOTE (ex. scipy/numpy en
    # macosx_14_0 + Accelerate) : on force des wheels marquees macOS 12, sinon
    # le .app plante au chargement sur Monterey.
    rm -rf wheels-mac12
    $RUN venv/bin/pip download -r "$REQ" pyinstaller -q -d wheels-mac12 \
        --only-binary=:all: --platform macosx_12_0_x86_64 --python-version "$PYREQ"
    $RUN venv/bin/pip install --no-index --find-links wheels-mac12 --force-reinstall -r "$REQ" pyinstaller -q
else
    $RUN venv/bin/pip install -r "$REQ" -q
    $RUN venv/bin/pip install pyinstaller -q
fi
echo "[OK] Environnement de build pret"

# --- PyInstaller : bundle .app ----------------------------------------------
echo "[..] PyInstaller..."
rm -rf dist build
$RUN venv/bin/pyinstaller --windowed --target-arch "$ARCH" --name "$APP" \
    --osx-bundle-identifier "com.boulemagique.audiovisualizer" \
    --add-data "$BUILD/src/render/shaders:render/shaders" \
    --collect-all moderngl --collect-all glcontext \
    "$ENTRY"
[ -d "dist/$APP.app" ] || { echo "[ERREUR] dist/$APP.app absent." >&2; exit 1; }
# Mode live : sans NSMicrophoneUsageDescription, macOS refuse l'acces aux entrees
# audio (BlackHole compris) sans meme afficher de demande. App Nap coupe : la
# sortie projo ne doit pas ralentir quand la fenetre de controle passe derriere.
PLIST="dist/$APP.app/Contents/Info.plist"
plutil -replace NSMicrophoneUsageDescription -string \
    "Analyse en direct du son (BlackHole, entree ligne) pour le mode live." "$PLIST"
plutil -replace LSAppNapIsDisabled -bool YES "$PLIST"
# Info.plist modifie => signature ad hoc a refaire, sinon le bundle est "endommage"
codesign --force --deep --sign - "dist/$APP.app"
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
