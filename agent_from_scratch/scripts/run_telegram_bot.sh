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
exec python3 -m agent_from_scratch.bots.telegram_bot
