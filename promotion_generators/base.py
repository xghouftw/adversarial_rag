"""Shared interfaces and result types for promotion document generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from promotion_selectors.base import contains_token_sequence


class TextProcessor(Protocol):
    def segments(self, passage: str) -> list[str]: ...

    def lemma_tokens(self, text: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class GeneratedDocument:
    strategy: str
    status: str
    text: str
    kept_segments: tuple[str, ...]
    removed_segments: tuple[str, ...]
    target_present: bool


DocumentGenerator = Callable[
    [str, str, Sequence[str], TextProcessor], GeneratedDocument
]


def contains_lemma_sequence(
    text: str,
    phrase: str,
    text_processor: TextProcessor,
) -> bool:
    return contains_token_sequence(
        text_processor.lemma_tokens(text), text_processor.lemma_tokens(phrase)
    )


__all__ = [
    "DocumentGenerator",
    "GeneratedDocument",
    "TextProcessor",
    "contains_lemma_sequence",
]
