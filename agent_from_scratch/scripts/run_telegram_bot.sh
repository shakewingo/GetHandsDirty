#!/bin/sh
# Loads local secrets, then runs the Telegram bot.
# Copy deploy/telegram-bot.env.example to deploy/telegram-bot.env (untracked) and fill in
# real values before using this script.
set -eu
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
ENV_FILE="$SCRIPT_DIR/deploy/telegram-bot.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi
cd "$SCRIPT_DIR/.."
# launchd starts jobs with a bare PATH, where python3 is the system Python without this
# project's dependencies. Prefer $PYTHON, then the repo's .venv, then python3 on PATH.
if [ -z "${PYTHON:-}" ]; then
    if [ -x .venv/bin/python ]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi
# Timestamp each (re)start so crash tracebacks that follow can be dated in the log.
echo "$(date '+%Y-%m-%d %H:%M:%S') starting telegram bot with $PYTHON"
exec "$PYTHON" -m agent_from_scratch.bots.telegram_bot
