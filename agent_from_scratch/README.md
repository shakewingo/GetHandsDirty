# agent_from_scratch — Telegram setup

Chat with this agent from your phone. The bot runs on your Mac, uses a local
Qwen2.5-7B model, and replies only to your own Telegram account.

**You need:** a Mac (Apple Silicon recommended), Python 3.11+, about 5 GB of free disk space,
and a Telegram account.

## 1. Get the code and install dependencies

```sh
git clone -b feat/telegram-bot https://github.com/shakewingo/GetHandsDirty.git
cd GetHandsDirty

python3 -m venv .venv
source .venv/bin/activate
pip install llama-cpp-python loguru huggingface_hub -r agent_from_scratch/requirements-tools.txt
```

Once the bot is merged into `main`, you can leave out `-b feat/telegram-bot`.
Run every command below from the `GetHandsDirty` folder with the venv active.

## 2. Download the model (~4.7 GB)

```sh
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m-*.gguf" \
  --revision bb5d59e06d9551d752d08b292a50eb208b07ab1f \
  --cache-dir gz-data/hub
```

This saves the files to the path in `agent_from_scratch/config.py` (`MODEL_PATH`).

## 3. Create your bot in Telegram

1. Open [@BotFather](https://t.me/BotFather), send `/newbot`, and follow the prompts.
   Copy the **bot token** it gives you.
2. Open [@userinfobot](https://t.me/userinfobot) and copy your numeric **user ID**.

## 4. Add your token and user ID

```sh
cp agent_from_scratch/deploy/telegram-bot.env.example agent_from_scratch/deploy/telegram-bot.env
```

Open `agent_from_scratch/deploy/telegram-bot.env` and fill in:

```sh
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_ALLOWED_USER_ID=123456789
```

If you installed the dependencies somewhere other than `.venv`, such as a conda env, also
set `PYTHON` to that interpreter's full path, for example `PYTHON=/path/to/env/bin/python`.

> [!WARNING]
> Keep this file private. Git already ignores it; don't force-add it. Anyone with the token
> can control the bot.
> The bot can read and write files and run shell commands, so it replies only to
> `TELEGRAM_ALLOWED_USER_ID` and ignores everyone else.

## 5. Start the bot and test it

```sh
sh agent_from_scratch/scripts/run_telegram_bot.sh
```

Send your bot a message in Telegram. The first reply can take a while because the model
has to load. Press `Ctrl-C` to stop the bot.

That's all you need if you only run the bot while your terminal is open.

## 6. Optional: keep the bot running 24/7

Use this step to start the bot automatically at login and restart it if it crashes.

```sh
mkdir -p agent_from_scratch/outputs
PLIST=~/Library/LaunchAgents/com.shakewingo.agent-telegram-bot.plist
cp agent_from_scratch/deploy/com.shakewingo.agent-telegram-bot.plist "$PLIST"
sed -i '' "s|/REPLACE/WITH/REPO/PATH|$PWD|g" "$PLIST"
launchctl load "$PLIST"
tail -f agent_from_scratch/outputs/telegram_bot.log
```

- **To stop the bot:** run `launchctl unload "$PLIST"`.
- **To apply code changes:** unload the job, then load it again.
- **To keep it reachable:** stop your Mac from sleeping. In System Settings, go to
  Energy Saver (or Battery, on a laptop) and turn on "Prevent automatic sleeping when the
  display is off". The display can still sleep.

## Chat commands

| Command         | What it does                                   |
|-----------------|------------------------------------------------|
| `/new`          | Start a fresh session                          |
| `/reset`        | Clear the current session's history            |
| `/session <id>` | Switch to a named session                      |
| `/compact`      | Summarize the history before your next message |

## Troubleshooting

- **`TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_ID must be set`:** the `.env` file is
  missing or has an empty value. Repeat step 4.
- **`No module named 'llama_cpp'` or `'loguru'`:** the bot started with a Python that
  doesn't have the dependencies. Install them into `.venv` (step 1), or set `PYTHON` in the
  `.env` file (step 4).
- **The bot never replies:** check that `TELEGRAM_ALLOWED_USER_ID` is your own ID, and look
  for errors in the terminal or in `agent_from_scratch/outputs/telegram_bot.log`.
