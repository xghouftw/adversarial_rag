"""Raw structural promotion selector."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from promotion_attack import Candidate, QuestionRecord

from .base import SelectorContext, SelectorResult


def select_raw_max_vote(
    record: QuestionRecord,
    candidates: Sequence[Candidate],
    context: SelectorContext,
) -> SelectorResult:
    """Keep the complete structural pool in its precomputed ranking."""
    del record, context
    return SelectorResult(candidates=tuple(candidates))
