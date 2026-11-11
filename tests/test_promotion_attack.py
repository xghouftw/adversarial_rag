import json
import re
from pathlib import Path
import tempfile
import unittest

from promotion_attack import (
    AttackConfig,
    LiveSweepResult,
    QuestionRecord,
    SpacyKeywordExtractor,
    TextGeneration,
    aggregate_keyword_sets,
    create_live_run,
    delete_gold_segments,
    execute_live_attack,
    execute_live_candidates,
    finalize_live_run,
    plan_attack,
    save_live_sweep,
)


class FakeToken:
    def __init__(self, text, lemma, pos, whitespace=" "):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.whitespace_ = whitespace


class FakeNlp:
    def __call__(self, text):
        if text == "The politician is Rep. George Santos.":
            return [
                FakeToken("The", "the", "DET"),
                FakeToken("politician", "politician", "NOUN"),
                FakeToken("is", "be", "AUX"),
                FakeToken("Rep.", "Rep.", "PROPN"),
                FakeToken("George", "George", "PROPN"),
                FakeToken("Santos", "Santos", "PROPN", ""),
                FakeToken(".", ".", "PUNCT", ""),
            ]
        raise AssertionError(f"unexpected fake NLP input: {text!r}")


class SimpleTextProcessor:
    def segments(self, passage):
        segments = []
        for line in passage.splitlines():
            segments.extend(
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", line.strip())
                if part.strip()
            )
        return segments

    def lemma_tokens(self, text):
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        return tuple(token[:-1] if token.endswith("s") else token for token in tokens)


class MappingKeywordExtractor:
    def __init__(self, mapping):
        self.mapping = mapping

    def extract(self, response):
        return set(self.mapping[response])


class PromotionAttackTests(unittest.TestCase):
    def setUp(self):
        self.processor = SimpleTextProcessor()
        self.config = AttackConfig()

    def test_aggregation_matches_threshold_and_prompt_order(self):
        aggregation = aggregate_keyword_sets(
            [
                {"Sinha", "temporary president"},
                {"Sinha", "Prasad", "temporary president"},
                {"Prasad"},
            ],
            alpha=0.5,
            beta=3.0,
            stopwords=set(),
        )
        self.assertEqual(aggregation.threshold, 1.5)
        self.assertEqual(
            aggregation.retained_keywords,
            ("temporary president", "Prasad", "Sinha"),
        )
        self.assertEqual(
            aggregation.hints, "temporary president, Prasad, Sinha"
        )

    def test_keyword_extractor_matches_upstream_phrase_and_lemma_behavior(self):
        extractor = SpacyKeywordExtractor(FakeNlp())
        self.assertEqual(
            extractor.extract("The politician is Rep. George Santos."),
            {
                "The politician is Rep. George Santos.",
                "politician",
                "Rep. George Santos",
                "Rep.",
                "George",
                "Santos",
                "",
            },
        )
        self.assertEqual(extractor.extract("I don't know"), set())

    def test_gold_deletion_preserves_non_gold_title_and_target(self):
        generated = delete_gold_segments(
            "Constituent Assembly\n"
            "Rajendra Prasad was elected president. "
            "Sachidananda Sinha served as temporary president.",
            target="Sachidananda Sinha",
            gold_aliases=["Rajendra Prasad"],
            text_processor=self.processor,
        )
        self.assertEqual(generated.status, "ok")
        self.assertEqual(
            generated.text,
            "Constituent Assembly\n"
            "Sachidananda Sinha served as temporary president.",
        )
        self.assertEqual(
            generated.removed_segments,
            ("Rajendra Prasad was elected president.",),
        )

    def test_raw_selector_uses_full_source_response_duplication(self):
        responses = (
            "Sinha response",
            "Gold response 1",
            "Gold response 2",
            "Other 3",
            "Other 4",
            "Other 5",
            "Other 6",
            "Other 7",
            "Other 8",
            "Displaced",
        )
        mapping = {
            "Sinha response": {"Sinha", "temporary president"},
            "Gold response 1": {"Prasad"},
            "Gold response 2": {"Prasad"},
            "Other 3": {"other3"},
            "Other 4": {"other4"},
            "Other 5": {"other5"},
            "Other 6": {"other6"},
            "Other 7": {"other7"},
            "Other 8": {"other8"},
            "Displaced": {"displaced"},
        }
        record = QuestionRecord(
            dataset="open_nq",
            query_index=5,
            question="Who was the first president?",
            gold_aliases=("Prasad",),
            passages=(
                "Prasad was later elected. Sinha served as temporary president.",
                "Prasad was elected.",
                "Prasad was elected again.",
                "unrelated passage",
                "unrelated passage",
                "unrelated passage",
                "unrelated passage",
                "unrelated passage",
                "unrelated passage",
                "displaced",
            ),
            responses=responses,
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        outcome = plan_attack(
            record,
            MappingKeywordExtractor(mapping),
            self.processor,
            stopwords=set(),
            config=self.config,
        )
        self.assertEqual(outcome.status, "selected")
        self.assertEqual(outcome.selected.target, "Sinha")
        self.assertEqual(outcome.selected.source_rank, 1)
        self.assertEqual(outcome.selected.simulated_attacked_target_votes, 2)
        self.assertEqual(outcome.selected.simulated_attacked_threshold, 2.0)
        self.assertIn(
            "temporary president", outcome.selected.simulated_retained_keywords
        )
        self.assertNotIn("Prasad was later elected", outcome.selected.document.text)

    def test_cleanly_retained_target_is_not_a_promotion_candidate(self):
        responses = (
            "response 0",
            "response 1",
            *(f"unknown response {index}" for index in range(2, 10)),
        )
        mapping = {
            response: ({"Sinha"} if index < 2 else {f"other{index}"})
            for index, response in enumerate(responses)
        }
        passages = tuple(
            "Sinha was temporary president." if index < 2 else f"other{index}"
            for index in range(10)
        )
        record = QuestionRecord(
            dataset="open_nq",
            query_index=1,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=passages,
            responses=responses,
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        outcome = plan_attack(
            record,
            MappingKeywordExtractor(mapping),
            self.processor,
            stopwords=set(),
            config=self.config,
        )
        self.assertEqual(outcome.status, "no_candidate")

    def test_short_upstream_record_displaces_its_actual_bottom_passage(self):
        responses = (
            "Sinha response",
            "Gold response 1",
            "Gold response 2",
            "Gold response 3",
            "Gold response 4",
            "Bottom wrong response",
        )
        mapping = {
            "Sinha response": {"Sinha"},
            "Gold response 1": {"Prasad"},
            "Gold response 2": {"Prasad"},
            "Gold response 3": {"Prasad"},
            "Gold response 4": {"Prasad"},
            "Bottom wrong response": {"BottomWrong"},
        }
        record = QuestionRecord(
            dataset="open_nq",
            query_index=64,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=(
                "Sinha served as temporary president.",
                "Prasad was president.",
                "Prasad was president.",
                "Prasad was president.",
                "Prasad was president.",
                "BottomWrong was mentioned.",
            ),
            responses=responses,
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        outcome = plan_attack(
            record,
            MappingKeywordExtractor(mapping),
            self.processor,
            stopwords=set(),
            config=self.config,
        )
        self.assertEqual(outcome.status, "selected")
        self.assertEqual(outcome.selected.target, "Sinha")
        self.assertNotIn(
            "BottomWrong", {candidate.target for candidate in outcome.candidates}
        )
        self.assertEqual(outcome.selected.maximum_simulated_gold_votes, 4)

    def test_live_execution_records_each_causal_stage(self):
        responses = tuple(f"response {index}" for index in range(10))
        mapping = {
            response: ({"Sinha"} if index == 0 else {"Prasad"})
            for index, response in enumerate(responses)
        }
        mapping["injected answer"] = {"Sinha"}
        record = QuestionRecord(
            dataset="open_nq",
            query_index=2,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=(
                "Sinha served as temporary president.",
                *("Prasad was president." for _ in range(9)),
            ),
            responses=responses,
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        extractor = MappingKeywordExtractor(mapping)
        selection = plan_attack(
            record,
            extractor,
            self.processor,
            stopwords=set(),
            config=self.config,
        )
        result = execute_live_attack(
            record,
            selection,
            extractor,
            stopwords=set(),
            config=self.config,
            query_isolated=lambda question, passage: TextGeneration(
                prompt=f"isolated: {question}\n{passage}",
                response="injected answer",
            ),
            query_final=lambda question, hints: TextGeneration(
                prompt=f"final: {question}\n{hints}",
                response="Sinha",
            ),
        )
        self.assertTrue(result.response_realized)
        self.assertTrue(result.promotion_realized)
        self.assertTrue(result.automatic_target_present)
        self.assertFalse(result.automatic_gold_present)
        self.assertTrue(result.automatic_strict_targeted_success)

    def test_exhaustive_sweep_reuses_identical_document_queries(self):
        responses = tuple(f"response {index}" for index in range(10))
        mapping = {
            response: ({"Sinha", "temporary"} if index == 0 else {"Prasad"})
            for index, response in enumerate(responses)
        }
        mapping["injected answer"] = {"Sinha", "temporary"}
        record = QuestionRecord(
            dataset="open_nq",
            query_index=3,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=(
                "Sinha was temporary president.",
                *("Prasad was president." for _ in range(9)),
            ),
            responses=responses,
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        extractor = MappingKeywordExtractor(mapping)
        selection = plan_attack(
            record,
            extractor,
            self.processor,
            stopwords=set(),
            config=self.config,
        )
        isolated_calls = []
        final_calls = []

        def query_isolated(question, passage):
            isolated_calls.append((question, passage))
            return TextGeneration(prompt="isolated", response="injected answer")

        def query_final(question, hints):
            final_calls.append((question, hints))
            return TextGeneration(prompt="final", response="Sinha")

        sweep = execute_live_candidates(
            record,
            selection,
            extractor,
            stopwords=set(),
            config=self.config,
            query_isolated=query_isolated,
            query_final=query_final,
        )
        self.assertEqual(sweep.attempted_candidates, 2)
        self.assertEqual(sweep.unique_poison_documents, 1)
        self.assertEqual(len(isolated_calls), 1)
        self.assertEqual(len(final_calls), 1)
        self.assertTrue(sweep.success_at_1)
        self.assertTrue(sweep.success_at_all)

    def test_live_run_requires_every_frozen_query_before_finalizing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_run = root / "source"
            source_run.mkdir()
            (source_run / "manifest.json").write_text("{}")
            source_manifest = {
                "dataset": "open_nq",
                "dataset_sha256": "dataset-hash",
                "upstream_commit": (
                    "9bc35b2fa5fa7d1088383b2789ec19a512d316b1"
                ),
            }
            live_dir = create_live_run(
                output_root=root / "runs",
                source_run=source_run,
                source_manifest=source_manifest,
                config=self.config,
                query_indices=[4],
                max_candidates=1,
            )
            with self.assertRaisesRegex(ValueError, "missing query results"):
                finalize_live_run(live_dir)

            sweep = LiveSweepResult(
                dataset="open_nq",
                query_index=4,
                question="Question",
                attempted_candidates=0,
                unique_poison_documents=0,
                success_at_1=False,
                success_at_3=False,
                success_at_all=False,
                results=(),
            )
            save_live_sweep(live_dir / "results", sweep)
            finalize_live_run(live_dir)
            manifest = json.loads((live_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
