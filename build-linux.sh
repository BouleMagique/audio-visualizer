#!/usr/bin/env bash
# =============================================================================
# Audio Visualizer — Build Linux (AppImage)
# =============================================================================
# Produit un VRAI livrable Linux : dist/AudioVisualizer-x86_64.AppImage
# Un seul fichier, chmod +x -> ça tourne. Python + Qt + ModernGL + shaders bundlés.
#
# Usage :
#   bash build-linux.sh
#
# Le build se fait dans un dossier LOCAL (hors CIFS/Samba, ou PyInstaller casse).
# =============================================================================
set -euo pipefail

APP="AudioVisualizer"
ENTRY="main.py"
ARCH="x86_64"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${XDG_CACHE_HOME:-$HOME/.cache}/audio-visualizer-build"
TOOLS="$BUILD/tools"
APPDIR="$BUILD/AppDir"
OUT="$SRC/dist"

echo ""
echo "=== Audio Visualizer — Build Linux (AppImage) ==="
echo "Source : $SRC"
echo "Build  : $BUILD  (local, hors CIFS)"
echo ""

# --- Python -----------------------------------------------------------------
command -v python3 >/dev/null || { echo "[ERREUR] python3 introuvable." >&2; exit 1; }
PYVER=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python3 -c 'import sys; exit(0 if sys.version_info >= (3,10) else 1)' \
    || { echo "[ERREUR] Python 3.10+ requis (detecte : $PYVER)." >&2; exit 1; }
echo "[OK] Python $PYVER"

# --- Copie source hors CIFS -------------------------------------------------
echo "[..] Copie des sources..."
mkdir -p "$BUILD/src"
rsync -a --delete \
    --exclude venv --exclude .git --exclude __pycache__ --exclude '.pytest_cache' \
    --exclude dist --exclude build --exclude AppDir --exclude tools \
    "$SRC"/ "$BUILD/src"/
cd "$BUILD/src"

# --- Venv + deps + PyInstaller ----------------------------------------------
[ -d venv ] || python3 -m venv venv
echo "[..] Installation des dependances + PyInstaller (Qt : ca peut etre long)..."
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
venv/bin/pip install pyinstaller -q
echo "[OK] Environnement de build pret"

# --- PyInstaller : bundle onedir --------------------------------------------
echo "[..] PyInstaller..."
rm -rf dist build
venv/bin/pyinstaller --onedir --windowed --name "$APP" \
    --add-data "$BUILD/src/render/shaders:render/shaders" \
    --collect-all moderngl --collect-all glcontext \
    "$ENTRY"
echo "[OK] Bundle genere : dist/$APP/"

# --- appimagetool -----------------------------------------------------------
mkdir -p "$TOOLS"
if [ ! -x "$TOOLS/appimagetool" ]; then
    echo "[..] Telechargement d'appimagetool..."
    curl -sSL -o "$TOOLS/appimagetool" \
        "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${ARCH}.AppImage"
    chmod +x "$TOOLS/appimagetool"
fi

# --- AppDir -----------------------------------------------------------------
echo "[..] Assemblage de l'AppDir..."
rm -rf "$APPDIR"; mkdir -p "$APPDIR/usr/bin"
cp -r "dist/$APP" "$APPDIR/usr/bin/$APP"

# Icone 256x256 (barres d'egaliseur) — pur Python, sans dep
venv/bin/python - "$APPDIR/audio-visualizer.png" <<'PY'
import sys, zlib, struct, math
W=H=256
bars=[0.35,0.6,0.85,0.55,0.95,0.7,0.45,0.75,0.5]
bw=W//len(bars)
def px(x,y):
    i=min(x//bw, len(bars)-1); h=bars[i]*H
    if y < H-h or x % bw < 2: return (14,14,22,255)
    t=(H-y)/H
    r=int(120+120*t); g=int(40+80*(1-t)); b=int(200+40*math.sin(t*3))
    return (min(r,255),min(g,255),min(b,255),255)
raw=bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W): raw+=bytes(px(x,y))
def ch(t,d): return struct.pack(">I",len(d))+t+d+struct.pack(">I",zlib.crc32(t+d)&0xffffffff)
png=b"\x89PNG\r\n\x1a\n"+ch(b"IHDR",struct.pack(">IIBBBBB",W,H,8,6,0,0,0))+ch(b"IDAT",zlib.compress(bytes(raw),9))+ch(b"IEND",b"")
open(sys.argv[1],"wb").write(png)
PY

cat > "$APPDIR/$APP.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Audio Visualizer
Comment=Visualiseur audio GPU -> export video (shaders GLSL)
Exec=AudioVisualizer
Icon=audio-visualizer
Categories=AudioVideo;Audio;Graphics;
Terminal=false
EOF

cat > "$APPDIR/AppRun" <<EOF
#!/bin/bash
HERE="\$(dirname "\$(readlink -f "\$0")")"
exec "\$HERE/usr/bin/$APP/$APP" "\$@"
EOF
chmod +x "$APPDIR/AppRun"

# --- Build AppImage ----------------------------------------------------------
echo "[..] Build de l'AppImage..."
mkdir -p "$OUT"
ARCH="$ARCH" APPIMAGE_EXTRACT_AND_RUN=1 \
    "$TOOLS/appimagetool" "$APPDIR" "$OUT/$APP-$ARCH.AppImage"

echo ""
echo "=== TERMINE ==="
echo "Livrable : $OUT/$APP-$ARCH.AppImage"
echo "Test     : chmod +x '$OUT/$APP-$ARCH.AppImage' && '$OUT/$APP-$ARCH.AppImage'"
echo ""
echo "Note : ffmpeg doit etre dans le PATH pour l'export video."
echo "       Le rendu GPU utilise les pilotes OpenGL du systeme (non bundles, normal)."
