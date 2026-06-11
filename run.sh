#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create venv if missing
if [ ! -f "$PYTHON" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Check and install dependencies
check_deps() {
    local missing=()
    "$PYTHON" -c "import PyQt6" 2>/dev/null || missing+=("PyQt6")
    "$PYTHON" -c "import cv2" 2>/dev/null || missing+=("opencv-python-headless")
    "$PYTHON" -c "import onnxruntime" 2>/dev/null || missing+=("onnxruntime")
    "$PYTHON" -c "import mss" 2>/dev/null || missing+=("mss")
    "$PYTHON" -c "import ultralytics" 2>/dev/null || missing+=("ultralytics")
    echo "${missing[@]}"
}

MISSING_DEPS=$(check_deps)
if [ -n "$MISSING_DEPS" ]; then
    print_info "Installing dependencies: $MISSING_DEPS"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --quiet $MISSING_DEPS
fi

# Show help
show_help() {
    cat << EOF
${GREEN}Brawlhalla AI Launcher${NC}

${YELLOW}Usage:${NC}
    ./run.sh [MODE] [OPTIONS]

${YELLOW}Modes:${NC}
    gui              Launch GUI interface (default)
    cli              Launch command-line interface
    dry              Test mode (no key inputs sent)

${YELLOW}CLI Options:${NC}
    --mode MODE      Game mode: full, assist, combat, poke
    --aggressive N   Aggression level 0.0-1.0 (default: 0.5)
    --character CHR  Character profile: balanced, aggressive, defensive, bodvar, orion, ember

${YELLOW}Examples:${NC}
    ./run.sh gui                    # Launch GUI
    ./run.sh cli --mode combat      # Combat mode via CLI
    ./run.sh dry --aggressive 0.8   # Test with high aggression
    ./run.sh cli --character bodvar # Use Bodvar profile

${YELLOW}Environment Variables:${NC}
    BH_CHARACTER     Character profile
    BH_AGGRESSION    Default aggression
    BH_DEVICE        Device: cuda, cpu
    BH_CONF_THRESH   Confidence threshold (default: 0.30)
    BH_AUTO_DETECT   Auto-detect game window: true, false

EOF
}

# Parse mode
MODE="${1:-gui}"
case "$MODE" in
    gui|cli|dry)
        shift
        ;;
    -h|--help)
        show_help
        exit 0
        ;;
    *)
        MODE="gui"
        ;;
esac

# Build command
CMD="$PYTHON"

case "$MODE" in
    gui)
        print_info "Launching GUI..."
        CMD="$CMD gui.py"
        ;;
    cli)
        print_info "Launching CLI..."
        CMD="$CMD main.py"
        ;;
    dry)
        print_info "Launching dry-run mode..."
        CMD="$CMD main.py --dry-run"
        ;;
esac

# Append remaining arguments
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

# Check for model files
MODEL_ONNX="./model/brawlhalla_vision.onnx"
MODEL_PT="./model/brawlhalla_vision.pt"

if [ ! -f "$MODEL_ONNX" ] && [ ! -f "$MODEL_PT" ]; then
    print_warn "Model files not found!"
    print_warn "Run: python download_model.py"
    print_warn "(Set ROBOFLOW_API_KEY environment variable first)"
fi

# Run
print_info "Starting Brawlhalla AI..."
exec $CMD