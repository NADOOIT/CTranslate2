#!/bin/zsh

source "$(dirname "$0")/build_utils.sh"

validate_python() {
    local python_bin="$1"

    if ! command -v "$python_bin" &> /dev/null; then
        log_error "Python not found at: $python_bin"
        log_info "Available Python versions:"
        ls /opt/homebrew/opt/python* 2>/dev/null
        die "Please install Python 3.9 or set PYTHON_BIN"
    fi

    if ! "$python_bin" --version 2>&1 | grep -q "Python 3.9"; then
        die "Python 3.9 required, found: $("$python_bin" --version 2>&1)"
    fi
}

setup_virtual_env() {
    "$PYTHON_BIN" -m venv "$VENV_DIR" || die "Failed to create venv"
    source "$VENV_DIR/bin/activate" || die "Failed to activate venv"
}
