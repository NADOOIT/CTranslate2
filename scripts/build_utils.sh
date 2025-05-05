#!/bin/zsh

# Utility Functions
die() {
    echo "❌ FATAL ERROR: $1" >&2
    exit 1
}

print_system_info() {
    echo "=== Starting CTranslate2 Build ==="
    echo "System: $(sw_vers -productName) $(sw_vers -productVersion)"
    echo "Build Type: Metal/MPS"
    echo "--------------------------------"
}

log_info() {
    echo "ℹ️ $1"
}

log_error() {
    echo "❌ $1" >&2
}
