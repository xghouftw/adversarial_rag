"""Document-aware reranking after deterministic NLP candidate filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from promotion_attack import Candidate, QuestionRecord

from .base import (
    CandidateAssessment,
    SelectorContext,
    SelectorResult,
    contains_token_sequence,
    normalized_tokens,
)
from .nlp import select_nlp_answer_shape


def _target_occurrence_count(text: str, target: str) -> int:
    """Count full normalized target-token occurrences, including overlaps."""
    text_tokens = normalized_tokens(text)
    target_tokens = normalized_tokens(target)
    if not target_tokens or len(target_tokens) > len(text_tokens):
        return 0
    width = len(target_tokens)
    return sum(
        tuple(text_tokens[start : start + width]) == target_tokens
        for start in range(len(text_tokens) - width + 1)
    )


def document_strength_features(candidate: Candidate) -> dict[str, int | bool]:
    """Return the fixed, generator-aware features used by the reranker."""
    first_segment = (
        candidate.document.kept_segments[0]
        if candidate.document.kept_segments
        else ""
    )
    target_tokens = normalized_tokens(candidate.target)
    target_in_first_segment = contains_token_sequence(
        normalized_tokens(first_segment),
        target_tokens,
    )
    replaced_gold_spans = (
        len(candidate.document.removed_segments)
        if candidate.document.strategy == "gold_alias_substitution"
        else 0
    )
    return {
        "target_in_first_segment": target_in_first_segment,
        "replaced_gold_spans": replaced_gold_spans,
        "target_occurrences": _target_occurrence_count(
            candidate.document.text,
            candidate.target,
        ),
    }


def select_document_strength(
    record: QuestionRecord,
    candidates: Sequence[Candidate],
    context: SelectorContext,
) -> SelectorResult:
    """Apply selector 1+2, then prefer stronger generated poison documents.

    Ranking is lexicographic and deterministic: target in the first generated
    segment, number of substituted gold spans, target occurrences, then the
    complete existing 1+2 order. The substitution count is zero for other
    document generators, for which ``removed_segments`` has a different
    meaning.
    """
    baseline = select_nlp_answer_shape(record, candidates, context)
    baseline_rank = {
        (candidate.target, candidate.source_index): rank
        for rank, candidate in enumerate(baseline.candidates)
    }
    features_by_key = {
        (candidate.target, candidate.source_index): document_strength_features(
            candidate
        )
        for candidate in candidates
    }
    candidate_by_key = {
        (candidate.target, candidate.source_rank): candidate
        for candidate in candidates
    }

    def ranking_key(candidate: Candidate) -> tuple[int, int, int, int]:
        key = (candidate.target, candidate.source_index)
        features = features_by_key[key]
        return (
            -int(features["target_in_first_segment"]),
            -int(features["replaced_gold_spans"]),
            -int(features["target_occurrences"]),
            baseline_rank[key],
        )

    ranked = tuple(
        sorted(
            baseline.candidates,
            key=ranking_key,
        )
    )

    assessments = []
    for assessment in baseline.assessments:
        candidate = candidate_by_key[(assessment.target, assessment.source_rank)]
        details = dict(assessment.details)
        details["nlp_answer_shape_reason"] = assessment.reason
        details.update(
            features_by_key[(candidate.target, candidate.source_index)]
        )
        assessments.append(
            CandidateAssessment(
                target=assessment.target,
                source_rank=assessment.source_rank,
                selector="document_strength",
                eligible=assessment.eligible,
                reason=assessment.reason,
                details=details,
            )
        )

    return SelectorResult(candidates=ranked, assessments=tuple(assessments))


__all__ = [
    "document_strength_features",
    "select_document_strength",
]
