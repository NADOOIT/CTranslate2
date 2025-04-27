#!/bin/bash
set -e

# === Konfiguration ===
PYTHON_VERSION=3.12
WHEEL_VERSION=4.5.1
PYTHON_DIR="python"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Submodules sicherstellen
cd "$ROOT_DIR"
echo "[1/5] Initialisiere Submodule (Googletest etc.)..."
git submodule update --init --recursive

# 2. Native Bibliothek bauen (MPS, ARM64)
echo "[2/5] Baue native Bibliothek mit MPS-Unterstützung..."
rm -rf build/
mkdir build && cd build
cmake -DWITH_MPS=ON -DWITH_CUDA=OFF -DWITH_PYTHON=ON -DCMAKE_OSX_ARCHITECTURES=arm64 ..
make -j$(sysctl -n hw.ncpu)

# 3. Python-Wheel bauen
cd "$ROOT_DIR/$PYTHON_DIR"
echo "[3/5] Baue Python-Wheel..."
rm -rf build/ dist/ ctranslate2.egg-info/
$PYTHON_VERSION -m venv venv312
source venv312/bin/activate
pip install --upgrade pip build
export ARCHFLAGS='-arch arm64'
export CMAKE_OSX_ARCHITECTURES='arm64'
$PYTHON_VERSION -m build --wheel

echo "[4/5] Prüfe Wheel-Inhalt..."
if ! unzip -l dist/ctranslate2-$WHEEL_VERSION-*.whl | grep -q 'libctranslate2.*.dylib'; then
    echo "Fehler: libctranslate2.dylib nicht im Wheel enthalten!"
    exit 1
fi

echo "[5/5] Teste Wheel in frischem venv..."
cd "$ROOT_DIR/$PYTHON_DIR"
$PYTHON_VERSION -m venv testvenv
source testvenv/bin/activate
pip install dist/ctranslate2-$WHEEL_VERSION-*.whl
python -c "import ctranslate2; print(ctranslate2.list_supported_devices())"

echo "\nFERTIG! Das Wheel ist unter python/dist/ bereit und unterstützt nativ MPS."
