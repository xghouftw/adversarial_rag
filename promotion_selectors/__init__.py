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
from .nlp import SpacyCandidateQualityAnalyzer, select_nlp_answer_shape
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
    "prometheus_judge": select_prometheus_judge,
}


__all__ = [
    "BUILTIN_SELECTORS",
    "CandidateAssessment",
    "CandidateJudge",
    "CandidateQualityAnalyzer",
    "JudgeDecision",
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
    "normalized_text",
    "normalized_tokens",
    "parse_prometheus_response",
    "select_nlp_answer_shape",
    "select_prometheus_judge",
    "select_raw_max_vote",
]
