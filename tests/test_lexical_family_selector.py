import unittest

from promotion_attack import Candidate, QuestionRecord, resolve_target_selector
from promotion_generators.base import GeneratedDocument
from promotion_selectors.base import SelectorContext
from promotion_selectors.lexical_family import (
    lexical_family_features,
    select_lexical_family,
)


class AllEligibleCandidateAnalyzer:
    def analyze(self, record, candidate):
        return {
            "eligible": True,
            "reason": "eligible",
            "type_match": 2,
            "unit_quality": 3,
            "specificity": 2,
        }


def make_candidate(
    target,
    source_rank,
    newly_retained_keywords,
    simulated_retained_keywords,
):
    return Candidate(
        target=target,
        source_index=source_rank - 1,
        source_rank=source_rank,
        source_passage="source",
        source_response="response",
        document=GeneratedDocument(
            strategy="test",
            status="ok",
            text="poison",
            kept_segments=("poison",),
            removed_segments=(),
            target_present=True,
        ),
        clean_target_votes=0,
        surviving_target_votes=0,
        simulated_attacked_target_votes=1,
        clean_threshold=1.0,
        simulated_attacked_threshold=1.0,
        simulated_attacked_margin=0.0,
        maximum_simulated_gold_votes=0,
        simulated_target_gold_gap=1,
        newly_retained_keywords=tuple(newly_retained_keywords),
        simulated_retained_keywords=tuple(simulated_retained_keywords),
    )


class LexicalFamilySelectorTests(unittest.TestCase):
    def test_promoted_family_reranks_over_baseline_order(self):
        record = QuestionRecord(
            dataset="example",
            query_index=0,
            question="What was it?",
            gold_aliases=("Gold",),
            passages=("source", "source"),
            responses=("response", "response"),
            clean_correct=True,
        )
        baseline_first = make_candidate("Alpha", 1, ("Alpha",), ("Alpha",))
        family_candidate = make_candidate(
            "Zulu",
            2,
            ("Zulu Alpha Beta", "Zulu", "Alpha"),
            ("Zulu Alpha Beta", "Zulu", "Alpha"),
        )

        result = select_lexical_family(
            record,
            (baseline_first, family_candidate),
            SelectorContext(candidate_analyzer=AllEligibleCandidateAnalyzer()),
        )

        self.assertEqual(
            [candidate.target for candidate in result.candidates],
            ["Zulu", "Alpha"],
        )
        assessment = next(
            assessment
            for assessment in result.assessments
            if assessment.target == "Zulu"
        )
        self.assertEqual(assessment.details["family_promoted_count"], 3)
        self.assertEqual(assessment.details["family_completeness"], 1.0)

    def test_shared_fragment_ambiguous_between_families_is_not_counted(self):
        candidate = make_candidate(
            "Alice Smith",
            1,
            ("Alice Smith", "James Smith", "Smith"),
            ("Alice Smith", "James Smith", "Smith"),
        )

        features = lexical_family_features(candidate)

        self.assertEqual(features["family_promoted_count"], 0)
        self.assertEqual(resolve_target_selector("5"), "lexical_family")


if __name__ == "__main__":
    unittest.main()
