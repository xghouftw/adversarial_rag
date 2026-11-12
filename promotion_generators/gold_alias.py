"""Constrained rewriting by substituting targets for complete gold aliases."""

from __future__ import annotations

import re
from typing import Sequence

from promotion_selectors.base import normalized_text, normalized_tokens

from .base import GeneratedDocument, TextProcessor, contains_lemma_sequence


def _gold_alias_spans(
    text: str,
    gold_aliases: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Return longest-first, non-overlapping full-alias spans in source order."""
    text_tokens = tuple(
        (match.group(0).casefold(), match.start(), match.end())
        for match in re.finditer(r"[a-z0-9]+", text, re.IGNORECASE)
    )
    candidate_spans: dict[tuple[int, int], int] = {}
    for alias in gold_aliases:
        alias_text = str(alias)
        alias_tokens = normalized_tokens(alias_text)
        if not alias_tokens or len(alias_tokens) > len(text_tokens):
            continue

        first_alphanumeric = re.search(r"[a-z0-9]", alias_text, re.IGNORECASE)
        last_alphanumeric = tuple(
            re.finditer(r"[a-z0-9]", alias_text, re.IGNORECASE)
        )
        alias_prefix = (
            alias_text[: first_alphanumeric.start()]
            if first_alphanumeric is not None
            else ""
        )
        alias_suffix = (
            alias_text[last_alphanumeric[-1].end() :]
            if last_alphanumeric
            else ""
        )

        width = len(alias_tokens)
        for start_index in range(len(text_tokens) - width + 1):
            window = tuple(
                token
                for token, _, _ in text_tokens[start_index : start_index + width]
            )
            if window != alias_tokens:
                continue

            start = text_tokens[start_index][1]
            end = text_tokens[start_index + width - 1][2]
            if (
                alias_prefix
                and text[max(0, start - len(alias_prefix)) : start] == alias_prefix
            ):
                start -= len(alias_prefix)
            if alias_suffix and text[end : end + len(alias_suffix)] == alias_suffix:
                end += len(alias_suffix)
            candidate_spans[(start, end)] = max(
                width, candidate_spans.get((start, end), 0)
            )

    selected: list[tuple[int, int]] = []
    for (start, end), width in sorted(
        candidate_spans.items(),
        key=lambda item: (
            -item[1],
            -(item[0][1] - item[0][0]),
            item[0][0],
            item[0][1],
        ),
    ):
        if any(
            start < selected_end and selected_start < end
            for selected_start, selected_end in selected
        ):
            continue
        selected.append((start, end))
    return tuple(sorted(selected))


def gold_alias_substitution(
    source_passage: str,
    target: str,
    gold_aliases: Sequence[str],
    text_processor: TextProcessor,
) -> GeneratedDocument:
    """Replace every full gold-alias occurrence with an existing target."""
    source_has_target = bool(source_passage.strip()) and contains_lemma_sequence(
        source_passage, target, text_processor
    )
    spans = _gold_alias_spans(source_passage, gold_aliases)

    if not source_passage.strip():
        status = "empty_source_passage"
        text = ""
    elif not source_has_target:
        status = "target_absent_from_source"
        text = source_passage
    elif not spans:
        status = "no_gold_alias_in_source"
        text = source_passage
    else:
        pieces = []
        cursor = 0
        for start, end in spans:
            pieces.extend((source_passage[cursor:start], target))
            cursor = end
        pieces.append(source_passage[cursor:])
        text = "".join(pieces)
        remaining_gold = any(
            contains_lemma_sequence(text, alias, text_processor)
            for alias in gold_aliases
            if normalized_text(alias)
        )
        status = (
            "gold_alias_remaining_after_substitution" if remaining_gold else "ok"
        )

    target_present = bool(text.strip()) and contains_lemma_sequence(
        text, target, text_processor
    )
    return GeneratedDocument(
        strategy="gold_alias_substitution",
        status=status,
        text=text,
        kept_segments=tuple(text_processor.segments(text)) if text.strip() else (),
        removed_segments=tuple(source_passage[start:end] for start, end in spans),
        target_present=target_present,
    )


__all__ = ["gold_alias_substitution"]
