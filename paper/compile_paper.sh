#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "[PDF BUILD] Compiling LaTeX manuscript via tectonic..."
if command -v tectonic &> /dev/null; then
    tectonic main.tex
elif [ -x /opt/homebrew/bin/tectonic ]; then
    /opt/homebrew/bin/tectonic main.tex
else
    pdflatex -interaction=nonstopmode main.tex
fi
echo "[PDF BUILD SUCCESS] Generated $(pwd)/main.pdf"
