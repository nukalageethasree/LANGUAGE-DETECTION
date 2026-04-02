#!/bin/bash
# ──────────────────────────────────────────────────────────────
#  LinguaDetect — Startup Script
#  Usage: bash run.sh
# ──────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       LinguaDetect — AI Language ID      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ─────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "❌  Python 3 not found. Please install Python 3.9+."
  exit 1
fi
echo "✅  Python: $(python3 --version)"

# ── Check model file ─────────────────────────────────────────
MODEL_PATH="$SCRIPT_DIR/model/language_detector.keras"
if [ ! -f "$MODEL_PATH" ]; then
  echo ""
  echo "❌  Model file not found at: $MODEL_PATH"
  echo "    Please copy language_detector.keras into the model/ folder."
  exit 1
fi
echo "✅  Model file found."

# ── Virtual environment ───────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
  echo ""
  echo "📦  Creating virtual environment…"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── Install dependencies ──────────────────────────────────────
echo ""
echo "📦  Installing/verifying dependencies…"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q
echo "✅  Dependencies ready."

# ── Launch ────────────────────────────────────────────────────
echo ""
echo "🚀  Starting LinguaDetect API…"
echo ""
echo "   ┌─────────────────────────────────────────┐"
echo "   │  Frontend:  http://localhost:8000        │"
echo "   │  API Docs:  http://localhost:8000/docs   │"
echo "   │  Health:    http://localhost:8000/health │"
echo "   └─────────────────────────────────────────┘"
echo ""
echo "   Press  Ctrl+C  to stop."
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
