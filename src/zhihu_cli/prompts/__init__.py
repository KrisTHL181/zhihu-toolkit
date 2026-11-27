"""LLM prompts and user-message templates loaded from text files.

Each prompt is a module-level string loaded from a ``.txt`` file in this
directory. Import them directly::

    from zhihu_cli.prompts import AGORA_VOTE_SYSTEM_PROMPT

The ``*_USER_PROMPT`` templates are rendered with :meth:`str.format`, so their
``{name}`` placeholders must be filled by the caller; write a literal brace as
``{{`` or ``}}``.

To modify a prompt, edit the corresponding ``.txt`` file — no code changes
needed.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def _load_prompt(filename: str) -> str:
    """Read a prompt from a text file, stripping trailing whitespace.

    :param filename: Name of the ``.txt`` file in this directory.
    :returns: Prompt string.
    """
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").rstrip()


AGORA_VOTE_SYSTEM_PROMPT: str = _load_prompt("agora_vote.txt")
"""System prompt for the Zhihu Agora (community moderation) AI voting feature."""

AGORA_VOTE_USER_PROMPT: str = _load_prompt("agora_vote_user.txt")
"""User-message template for Agora AI voting; rendered by the caller."""


__all__ = [
    "AGORA_VOTE_SYSTEM_PROMPT",
    "AGORA_VOTE_USER_PROMPT",
]
