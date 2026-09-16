import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # picks up OPENROUTER_API_KEY from a local .env file, if present

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

_client: OpenAI | None = None


class MissingAPIKeyError(RuntimeError):
    """OPENROUTER_API_KEY isn't set. Not retryable -- fix your env, don't retry."""


def _get_client() -> OpenAI:
    """Build the OpenRouter client lazily, on first real use.

    Building it at import time is what the original version did, but the
    OpenAI SDK's constructor raises immediately if no key is available --
    so a missing OPENROUTER_API_KEY crashed on `import util`, before this
    module's own check ever ran, and with a confusing "set OPENAI_API_KEY"
    message. Lazy construction means we control exactly when and how a
    missing key is reported, and we never fall back to an unrelated
    OPENAI_API_KEY the environment happens to have set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise MissingAPIKeyError("OPENROUTER_API_KEY environment variable not set")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    return _client


def llm_call(prompt: str, system_prompt: str = "", model: str = DEFAULT_MODEL, max_tokens: int = 4096) -> str:
    """Send a prompt to a model via OpenRouter and return the text response."""
    client = _get_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content or ""


def extract_xml(text: str, tag: str) -> str:
    """Extract the content of the given XML tag from the text."""
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""
