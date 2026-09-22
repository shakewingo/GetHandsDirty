# Telegram bot setup (Mac mini)

One-time setup to run the agent as a Telegram bot on the always-on Mac mini. Design:
[superpowers/specs/2026-09-22-telegram-bot-design.md](superpowers/specs/2026-09-22-telegram-bot-design.md).

## 1. Get the repo and weights onto the Mac mini

Clone/copy this repository, then copy the GGUF weights to the path `config.py` expects
(`MODEL_PATH`) — about 4.7 GB. Install the same Python environment used for local development
(`llama-cpp-python`, `httpx`, `loguru`, and the rest of `requirements-tools.txt`).

## 2. Create a Telegram bot and find your user ID

1. Message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the
   prompts. Save the token it gives you.
2. Message [@userinfobot](https://t.me/userinfobot) to get your own numeric Telegram user ID.

## 3. Configure secrets

```sh
cp agent_from_scratch/deploy/telegram-bot.env.example agent_from_scratch/deploy/telegram-bot.env
```

Edit `agent_from_scratch/deploy/telegram-bot.env` and fill in `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_ALLOWED_USER_ID`. This file is untracked (matches the existing `agent_from_scratch/docs/`
gitignore pattern's spirit — never commit it).

## 4. Try it once in the foreground

```sh
cd /path/to/repo
sh agent_from_scratch/scripts/run_telegram_bot.sh
```

Message your bot from Telegram. Confirm you get a reply, `/new` starts a fresh session, and
(if you can test from a second account) a non-allowlisted sender gets no reply at all.
Stop with Ctrl-C.

## 5. Install the launchd job

```sh
cp agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist ~/Library/LaunchAgents/
# Edit ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist:
# replace /REPLACE/WITH/REPO/PATH with the absolute path to this repo on this machine.
launchctl load ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist
```

Check it's running and tail the log:

```sh
launchctl list | grep agent-telegram-bot
tail -f agent_from_scratch/outputs/telegram_bot.log
```

## 6. Everyday use

- `/new` — start a fresh session in this chat.
- `/reset` — clear the current session's history.
- `/session <id>` — switch this chat to a named session.
- `/compact` — summarize history before the next message.
- To stop the service: `launchctl unload ~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist`.
- To restart after a code change: unload, then `launchctl load` again.

## Energy Saver

The Mac mini must not sleep for the bot to stay reachable. System Settings → Energy Saver
(or Battery, on a laptop) → set "Prevent automatic sleeping when the display is off" while
plugged in. Screen sleep is fine; system sleep is not.
