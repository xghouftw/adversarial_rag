"""Built-in promotion target selector implementations."""

from .base import (
    CandidateAssessment,
    CandidateJudge,
    CandidateQualityAnalyzer,
    JudgeDecision,
    SelectorContext,
    SelectorResult,
    TargetSelector,
    contains_token_sequence,
    normalized_text,
    normalized_tokens,
)
from .document_strength import (
    document_strength_features,
    select_document_strength,
)
from .nlp import SpacyCandidateQualityAnalyzer, select_nlp_answer_shape
from .lexical_family import lexical_family_features, select_lexical_family
from .prometheus import (
    PROMETHEUS_MODEL,
    PROMETHEUS_REVISION,
    PrometheusCandidateJudge,
    RecordedCandidateJudge,
    build_prometheus_prompt,
    parse_prometheus_response,
    select_prometheus_judge,
)
from .raw import select_raw_max_vote


BUILTIN_SELECTORS: dict[str, TargetSelector] = {
    "raw_max_vote": select_raw_max_vote,
    "nlp_answer_shape": select_nlp_answer_shape,
    "document_strength": select_document_strength,
    "lexical_family": select_lexical_family,
    "prometheus_judge": select_prometheus_judge,
}

NLP_ANALYZER_SELECTORS = frozenset(
    {"nlp_answer_shape", "document_strength", "lexical_family"}
)


__all__ = [
    "BUILTIN_SELECTORS",
    "CandidateAssessment",
    "CandidateJudge",
    "CandidateQualityAnalyzer",
    "JudgeDecision",
    "NLP_ANALYZER_SELECTORS",
    "PROMETHEUS_MODEL",
    "PROMETHEUS_REVISION",
    "PrometheusCandidateJudge",
    "RecordedCandidateJudge",
    "SelectorContext",
    "SelectorResult",
    "SpacyCandidateQualityAnalyzer",
    "TargetSelector",
    "build_prometheus_prompt",
    "contains_token_sequence",
    "document_strength_features",
    "lexical_family_features",
    "normalized_text",
    "normalized_tokens",
    "parse_prometheus_response",
    "select_nlp_answer_shape",
    "select_document_strength",
    "select_lexical_family",
    "select_prometheus_judge",
    "select_raw_max_vote",
]
