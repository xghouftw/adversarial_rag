"""Extractive document generation by deleting gold-bearing segments."""

from __future__ import annotations

from typing import Sequence

from promotion_selectors.base import normalized_text

from .base import GeneratedDocument, TextProcessor, contains_lemma_sequence


def delete_gold_segments(
    source_passage: str,
    target: str,
    gold_aliases: Sequence[str],
    text_processor: TextProcessor,
) -> GeneratedDocument:
    kept = []
    removed = []
    for segment in text_processor.segments(source_passage):
        contains_gold = any(
            contains_lemma_sequence(segment, alias, text_processor)
            for alias in gold_aliases
            if normalized_text(alias)
        )
        if contains_gold:
            removed.append(segment)
        else:
            kept.append(segment)

    text = "\n".join(kept).strip()
    target_present = bool(text) and contains_lemma_sequence(
        text, target, text_processor
    )
    if not text:
        status = "empty_after_gold_deletion"
    elif not target_present:
        status = "target_absent_after_gold_deletion"
    else:
        status = "ok"
    return GeneratedDocument(
        strategy="delete_gold_segments",
        status=status,
        text=text,
        kept_segments=tuple(kept),
        removed_segments=tuple(removed),
        target_present=target_present,
    )


__all__ = ["delete_gold_segments"]
