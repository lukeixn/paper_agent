# config/settings.py

import os

PROVIDER = "openai"   # or "deepseek"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

OPENAI_MODEL = "gpt-4o"
DEEPSEEK_MODEL = "deepseek-chat"