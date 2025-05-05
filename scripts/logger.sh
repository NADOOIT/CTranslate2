#!/bin/zsh

# Logging levels
LOG_LEVEL_DEBUG=0
LOG_LEVEL_INFO=1
LOG_LEVEL_WARN=2
LOG_LEVEL_ERROR=3

# Default log level
CURRENT_LOG_LEVEL=$LOG_LEVEL_INFO

# Log file
LOG_FILE="/tmp/ctranslate2_build.log"

# Color codes
COLOR_GREEN="\033[0;32m"
COLOR_YELLOW="\033[0;33m"
COLOR_RED="\033[0;31m"
COLOR_RESET="\033[0m"

# Logging functions
log_debug() {
    [[ $CURRENT_LOG_LEVEL -le $LOG_LEVEL_DEBUG ]] && {
        echo -e "${COLOR_GREEN}[DEBUG]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
    }
}

log_info() {
    [[ $CURRENT_LOG_LEVEL -le $LOG_LEVEL_INFO ]] && {
        echo -e "${COLOR_GREEN}[INFO]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
    }
}

log_warn() {
    [[ $CURRENT_LOG_LEVEL -le $LOG_LEVEL_WARN ]] && {
        echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $*" | tee -a "$LOG_FILE" >&2
    }
}

log_error() {
    [[ $CURRENT_LOG_LEVEL -le $LOG_LEVEL_ERROR ]] && {
        echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" | tee -a "$LOG_FILE" >&2
    }
}

# Set log level based on argument
set_log_level() {
    case "$1" in
        debug) CURRENT_LOG_LEVEL=$LOG_LEVEL_DEBUG ;;
        info)  CURRENT_LOG_LEVEL=$LOG_LEVEL_INFO ;;
        warn)  CURRENT_LOG_LEVEL=$LOG_LEVEL_WARN ;;
        error) CURRENT_LOG_LEVEL=$LOG_LEVEL_ERROR ;;
        *)     log_warn "Invalid log level. Using default." ;;
    esac
}

# Trap errors and log them
set -o errexit
trap 'log_error "Command failed: $BASH_COMMAND"' ERR
