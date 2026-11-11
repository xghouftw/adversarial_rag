"""Shared interfaces and result types for promotion target selectors."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

if TYPE_CHECKING:
    from promotion_attack import Candidate, QuestionRecord


@dataclass(frozen=True)
class CandidateAssessment:
    target: str
    source_rank: int
    selector: str
    eligible: bool
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class JudgeDecision:
    score: int | None
    feedback: str
    prompt: str
    response: str
    parse_status: str


@dataclass(frozen=True)
class SelectorResult:
    candidates: tuple[Candidate, ...]
    assessments: tuple[CandidateAssessment, ...] = ()


class CandidateQualityAnalyzer(Protocol):
    def analyze(
        self,
        record: QuestionRecord,
        candidate: Candidate,
    ) -> Mapping[str, Any]: ...


class CandidateJudge(Protocol):
    def judge(
        self,
        record: QuestionRecord,
        candidates: Sequence[Candidate],
    ) -> Sequence[JudgeDecision]: ...


@dataclass(frozen=True)
class SelectorContext:
    candidate_analyzer: CandidateQualityAnalyzer | None = None
    candidate_judge: CandidateJudge | None = None


class TargetSelector(Protocol):
    """Common function signature for policies over a structural pool."""

    def __call__(
        self,
        record: QuestionRecord,
        candidates: Sequence[Candidate],
        context: SelectorContext,
    ) -> SelectorResult: ...


def normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", str(text).casefold()))


def normalized_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def contains_token_sequence(
    container_tokens: Sequence[str],
    phrase_tokens: Sequence[str],
) -> bool:
    if not phrase_tokens or len(phrase_tokens) > len(container_tokens):
        return False
    width = len(phrase_tokens)
    return any(
        tuple(container_tokens[start : start + width]) == tuple(phrase_tokens)
        for start in range(len(container_tokens) - width + 1)
    )
