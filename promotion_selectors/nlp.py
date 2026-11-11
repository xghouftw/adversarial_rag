"""Deterministic NLP answer-shape selector."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from promotion_attack import Candidate, QuestionRecord

from .base import (
    CandidateAssessment,
    SelectorContext,
    SelectorResult,
    contains_token_sequence,
    normalized_text,
    normalized_tokens,
)


META_NONANSWER = re.compile(
    r"\b(no|not)\s+(?:\w+\s+){0,3}(?:given|provided|mentioned|specified)\b",
    re.IGNORECASE,
)


class SpacyCandidateQualityAnalyzer:
    """Deterministic answer-shape features; this is not a factual judge."""

    ENTITY_TYPES = {
        "person": {"PERSON", "ORG", "NORP"},
        # spaCy commonly labels hotels, venues, and institutions as ORG even
        # when they are valid answers to a where-question.
        "location": {"GPE", "LOC", "FAC", "ORG"},
        "time": {"DATE", "TIME"},
        "number": {"CARDINAL", "QUANTITY", "ORDINAL", "PERCENT"},
        "money": {"MONEY"},
    }
    MONTH = re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",
        re.IGNORECASE,
    )
    YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")
    CURRENCY = re.compile(
        r"(?:[$£€]\s*\d|\b\d[\d,.]*\s*(?:dollars?|usd|pounds?|euros?)\b)",
        re.IGNORECASE,
    )
    NUMBER = re.compile(r"\b\d[\d,.]*\b")
    INITIALS_ONLY = re.compile(
        r"^(?:(?:[A-Za-z]\.\s*)+|(?:[A-Za-z]\s+){1,3}[A-Za-z])$"
    )

    def __init__(self, nlp: Any):
        self.nlp = nlp

    @staticmethod
    def expected_answer_type(question: str) -> str:
        text = normalized_text(question)
        if re.search(r"\b(how much|cost|price|dollar|amount|salary|worth)\b", text):
            return "money"
        if re.search(r"\b(how many|what number|which number)\b", text):
            return "number"
        if re.search(r"\b(when|what year|which year|what date|which date)\b", text):
            return "time"
        if re.search(
            r"\b(where|what place|which place|what city|which city|"
            r"what country|which country)\b",
            text,
        ):
            return "location"
        if re.search(r"\b(who|whom|whose)\b", text):
            return "person"
        return "open"

    def _span_lemmas(self, span: Any) -> tuple[str, ...]:
        tokens = []
        for token in span:
            if token.is_space or token.is_punct:
                continue
            tokens.extend(normalized_tokens(token.lemma_))
        return tuple(tokens)

    def analyze(
        self,
        record: QuestionRecord,
        candidate: Candidate,
    ) -> Mapping[str, Any]:
        target_doc = self.nlp(candidate.target)
        response_doc = self.nlp(candidate.source_response)
        target_lemmas = self._span_lemmas(target_doc)
        response_lemmas = self._span_lemmas(response_doc)
        question_tokens = normalized_tokens(record.question)

        matching_entity_labels = {
            entity.label_
            for document in (target_doc, response_doc)
            for entity in document.ents
            if self._span_lemmas(entity) == target_lemmas
        }
        noun_chunk_match = False
        try:
            noun_chunk_match = any(
                self._span_lemmas(chunk) == target_lemmas
                for chunk in response_doc.noun_chunks
            )
        except (AttributeError, ValueError):
            pass

        literal_types = set()
        if self.CURRENCY.search(candidate.target):
            literal_types.add("money")
        if self.YEAR.search(candidate.target) or self.MONTH.search(candidate.target):
            literal_types.add("time")
        if self.NUMBER.search(candidate.target):
            literal_types.add("number")

        expected_type = self.expected_answer_type(record.question)
        compatible_labels = self.ENTITY_TYPES.get(expected_type, set())
        if expected_type == "open":
            type_match = 1
        elif matching_entity_labels & compatible_labels or expected_type in literal_types:
            type_match = 2
        elif matching_entity_labels or literal_types:
            type_match = 0
        else:
            type_match = 1

        content_tokens = [
            token for token in target_doc if not token.is_space and not token.is_punct
        ]
        proper_name = bool(content_tokens) and all(
            token.pos_ == "PROPN" for token in content_tokens
        )
        full_short_response = (
            target_lemmas == response_lemmas and 0 < len(response_lemmas) <= 10
        )
        typed_or_entity = bool(literal_types or matching_entity_labels or proper_name)
        if typed_or_entity:
            unit_quality = 3
        elif noun_chunk_match:
            unit_quality = 2
        elif full_short_response:
            unit_quality = 1
        else:
            unit_quality = 0

        if typed_or_entity:
            specificity = 2
        elif len(target_lemmas) >= 2:
            specificity = 1
        else:
            specificity = 0

        question_echo = contains_token_sequence(
            question_tokens,
            normalized_tokens(candidate.target),
        )
        meta_nonanswer = bool(META_NONANSWER.search(candidate.source_response))
        initials_only = bool(self.INITIALS_ONLY.fullmatch(candidate.target.strip()))

        if meta_nonanswer:
            reason = "meta_nonanswer"
        elif initials_only:
            reason = "initials_only"
        elif question_echo:
            reason = "question_echo"
        elif type_match == 0:
            reason = "answer_type_conflict"
        elif unit_quality == 0:
            reason = "not_complete_answer_unit"
        elif specificity == 0:
            reason = "single_generic_word"
        else:
            reason = "eligible"

        return {
            "eligible": reason == "eligible",
            "reason": reason,
            "expected_answer_type": expected_type,
            "matching_entity_labels": sorted(matching_entity_labels),
            "literal_types": sorted(literal_types),
            "noun_chunk_match": noun_chunk_match,
            "full_short_response": full_short_response,
            "question_echo": question_echo,
            "meta_nonanswer": meta_nonanswer,
            "initials_only": initials_only,
            "type_match": type_match,
            "unit_quality": unit_quality,
            "specificity": specificity,
        }


def select_nlp_answer_shape(
    record: QuestionRecord,
    candidates: Sequence[Candidate],
    context: SelectorContext,
) -> SelectorResult:
    """Filter and rank structural candidates by deterministic answer form."""
    if context.candidate_analyzer is None:
        raise ValueError("selector 1+2 requires a candidate quality analyzer")
    analyzed = [
        (candidate, dict(context.candidate_analyzer.analyze(record, candidate)))
        for candidate in candidates
    ]

    for candidate, details in analyzed:
        if not details.get("eligible"):
            continue
        target_tokens = normalized_tokens(candidate.target)
        for other, other_details in analyzed:
            if other is candidate or other.source_index != candidate.source_index:
                continue
            other_tokens = normalized_tokens(other.target)
            if (
                len(other_tokens) > len(target_tokens)
                and contains_token_sequence(other_tokens, target_tokens)
                and other_details.get("eligible")
                and int(other_details.get("unit_quality", 0))
                >= int(details.get("unit_quality", 0))
            ):
                details["eligible"] = False
                details["reason"] = "strict_fragment_of_better_candidate"
                details["superseded_by"] = other.target
                break

    eligible = [
        (candidate, details)
        for candidate, details in analyzed
        if details.get("eligible")
    ]
    ranked = tuple(
        candidate
        for candidate, details in sorted(
            eligible,
            key=lambda item: (
                -int(item[1].get("type_match", 0)),
                -int(item[1].get("unit_quality", 0)),
                -int(item[1].get("specificity", 0)),
                -item[0].simulated_attacked_margin,
                -item[0].surviving_target_votes,
                -item[0].simulated_target_gold_gap,
                normalized_text(item[0].target),
                item[0].source_rank,
                item[0].target,
            ),
        )
    )
    assessments = tuple(
        CandidateAssessment(
            target=candidate.target,
            source_rank=candidate.source_rank,
            selector="nlp_answer_shape",
            eligible=bool(details.get("eligible")),
            reason=str(details.get("reason", "unknown")),
            details=details,
        )
        for candidate, details in analyzed
    )
    return SelectorResult(candidates=ranked, assessments=assessments)
