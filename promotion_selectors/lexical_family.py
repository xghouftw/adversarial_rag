"""Family-aware reranking over exact promoted keywords."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .base import (
    CandidateAssessment,
    SelectorContext,
    SelectorResult,
    contains_token_sequence,
    normalized_tokens,
)
from .nlp import select_nlp_answer_shape

if TYPE_CHECKING:
    from promotion_attack import Candidate, QuestionRecord


def lexical_family_features(candidate: Candidate) -> dict[str, int | float]:
    """Measure newly promoted members of unambiguous lexical families."""
    forms = tuple(
        dict.fromkeys(
            tokens
            for keyword in candidate.simulated_retained_keywords
            if (tokens := normalized_tokens(keyword))
        )
    )
    new_forms = {
        tokens
        for keyword in candidate.newly_retained_keywords
        if (tokens := normalized_tokens(keyword))
    }
    maximal_forms = tuple(
        form
        for form in forms
        if not any(
            len(other) > len(form) and contains_token_sequence(other, form)
            for other in forms
        )
    )
    memberships = {
        form: tuple(
            maximal
            for maximal in maximal_forms
            if contains_token_sequence(maximal, form)
        )
        for form in forms
    }
    positions = {}
    for index, keyword in enumerate(candidate.simulated_retained_keywords):
        form = normalized_tokens(keyword)
        if form:
            positions.setdefault(form, index + 1)

    best = (0, 0.0, 0, 0.0)
    for maximal in maximal_forms:
        family = tuple(
            form
            for form in forms
            if len(memberships[form]) == 1 and maximal in memberships[form]
        )
        promoted = tuple(form for form in family if form in new_forms)
        if len(promoted) < 2:
            continue
        family_positions = tuple(positions[form] for form in promoted)
        score = (
            len(promoted),
            len(promoted) / len(family),
            -min(family_positions),
            -sum(family_positions) / len(family_positions),
        )
        if score > best:
            best = score

    return {
        "family_promoted_count": best[0],
        "family_completeness": best[1],
        "family_first_position": -best[2] if best[0] else 0,
        "family_mean_position": -best[3] if best[0] else 0.0,
    }


def select_lexical_family(
    record: QuestionRecord,
    candidates: Sequence[Candidate],
    context: SelectorContext,
) -> SelectorResult:
    """Keep 1+2 eligibility and rerank for coherent promoted hint families."""
    baseline = select_nlp_answer_shape(record, candidates, context)
    baseline_rank = {
        (candidate.target, candidate.source_index): rank
        for rank, candidate in enumerate(baseline.candidates)
    }
    features_by_key = {
        (candidate.target, candidate.source_index): lexical_family_features(
            candidate
        )
        for candidate in candidates
    }
    candidate_by_key = {
        (candidate.target, candidate.source_rank): candidate
        for candidate in candidates
    }

    def ranking_key(candidate: Candidate) -> tuple[int, float, int, float, int]:
        features = features_by_key[(candidate.target, candidate.source_index)]
        return (
            -int(features["family_promoted_count"]),
            -float(features["family_completeness"]),
            int(features["family_first_position"]),
            float(features["family_mean_position"]),
            baseline_rank[(candidate.target, candidate.source_index)],
        )

    ranked = tuple(sorted(baseline.candidates, key=ranking_key))
    assessments = []
    for assessment in baseline.assessments:
        candidate = candidate_by_key[(assessment.target, assessment.source_rank)]
        assessments.append(
            CandidateAssessment(
                target=assessment.target,
                source_rank=assessment.source_rank,
                selector="lexical_family",
                eligible=assessment.eligible,
                reason=assessment.reason,
                details={
                    **assessment.details,
                    "nlp_answer_shape_reason": assessment.reason,
                    **features_by_key[(candidate.target, candidate.source_index)],
                },
            )
        )
    return SelectorResult(candidates=ranked, assessments=tuple(assessments))


__all__ = ["lexical_family_features", "select_lexical_family"]
