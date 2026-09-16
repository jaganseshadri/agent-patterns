import os
import re

from openai import OpenAI

# OpenRouter exposes an OpenAI-compatible API — same client, different base_url.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


def llm_call(prompt: str, system_prompt: str = "", model: str = DEFAULT_MODEL, max_tokens: int = 4096) -> str:
    """Send a prompt to a model via OpenRouter and return the text response."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

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
