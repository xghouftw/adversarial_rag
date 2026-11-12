import json
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from promotion_attack import (
    AttackConfig,
    LiveSweepResult,
    QuestionRecord,
    SpacyKeywordExtractor,
    TARGET_SELECTORS,
    TextGeneration,
    aggregate_keyword_sets,
    create_live_run,
    enrich_records_with_certification,
    execute_live_attack,
    execute_live_candidates,
    finalize_live_run,
    one_injection_keyword_frontier,
    plan_attack,
    save_live_sweep,
)
from promotion_generators import delete_gold_segments, gold_alias_substitution
from promotion_selectors.base import JudgeDecision
from promotion_selectors.nlp import SpacyCandidateQualityAnalyzer
from promotion_selectors.prometheus import (
    build_prometheus_prompt,
    parse_prometheus_response,
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


class AnalyzerToken:
    def __init__(self, text, lemma=None, pos="NOUN", punct=False):
        self.text = text
        self.lemma_ = text if lemma is None else lemma
        self.pos_ = pos
        self.is_space = False
        self.is_punct = punct


class AnalyzerSpan(list):
    def __init__(self, tokens, label=""):
        super().__init__(tokens)
        self.label_ = label


class AnalyzerDoc(list):
    def __init__(self, tokens, entities=(), noun_chunks=()):
        super().__init__(tokens)
        self.ents = tuple(entities)
        self.noun_chunks = tuple(noun_chunks)


class AnalyzerNlp:
    def __init__(self, documents):
        self.documents = documents

    def __call__(self, text):
        return self.documents[text]


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


class MappingCandidateAnalyzer:
    def __init__(self, eligible_target):
        self.eligible_target = eligible_target

    def analyze(self, record, candidate):
        eligible = candidate.target == self.eligible_target
        return {
            "eligible": eligible,
            "reason": "eligible" if eligible else "test_rejection",
            "type_match": 2 if eligible else 0,
            "unit_quality": 3 if eligible else 0,
            "specificity": 2 if eligible else 0,
        }


class MappingCandidateJudge:
    def __init__(self, scores):
        self.scores = scores

    def judge(self, record, candidates):
        return tuple(
            JudgeDecision(
                score=self.scores[candidate.target],
                feedback=f"score {self.scores[candidate.target]}",
                prompt=f"prompt for {candidate.target}",
                response=f"Feedback: test [RESULT] {self.scores[candidate.target]}",
                parse_status="ok",
            )
            for candidate in candidates
        )


class PromotionAttackTests(unittest.TestCase):
    def setUp(self):
        self.processor = SimpleTextProcessor()
        self.config = AttackConfig()

    def test_builtin_selectors_share_callable_interface(self):
        self.assertEqual(
            set(TARGET_SELECTORS),
            {"raw_max_vote", "nlp_answer_shape", "prometheus_judge"},
        )
        self.assertTrue(
            all(callable(selector) for selector in TARGET_SELECTORS.values())
        )

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

    def test_certification_frontier_removes_bottom_and_finds_one_short(self):
        frontier = one_injection_keyword_frontier(
            [
                {"kept", "one-short"},
                {"kept"},
                {"other"},
                {"bottom-only"},
            ],
            alpha=0.5,
            beta=3.0,
            stopwords=set(),
        )
        self.assertEqual(frontier.surviving_answering_passages, 3)
        self.assertEqual(frontier.threshold, 2.0)
        self.assertEqual(frontier.already_kept_keywords, ("kept",))
        self.assertEqual(
            set(frontier.one_vote_short_keywords), {"one-short", "other"}
        )
        self.assertNotIn("bottom-only", frontier.keyword_votes)

    def test_checked_certification_artifact_enriches_records(self):
        record = QuestionRecord(
            dataset="open_nq",
            query_index=0,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=("passage one", "passage two"),
            responses=("Sinha", "Prasad"),
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        source_manifest = {
            "dataset": "open_nq",
            "upstream_commit": (
                "9bc35b2fa5fa7d1088383b2789ec19a512d316b1"
            ),
            "parameters": {"alpha": 0.2, "beta": 3.0, "top_k": 10},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certification_run = root / "certification"
            certification_run.mkdir()
            summary = {
                "checks_passed": True,
                "final_answer_mismatches": [],
                "dataset": "open_nq",
                "upstream_commit": source_manifest["upstream_commit"],
                "alpha": 0.2,
                "beta": 3.0,
                "corruption_size": 1,
                "top_k": 10,
                "sample_size": 1,
                "source_run": "/other/root/source-run",
            }
            row = {
                "query_index": 0,
                "question": record.question,
                "answers": list(record.gold_aliases),
                "final_answer": record.clean_final_answer,
                "clean_correct": True,
                "certified": False,
                "certification_outcome": "wrong_answer_found",
                "certify_threshold": 1.2,
                "poison_target_candidates": [
                    {"keyword": "Sinha", "votes": 1, "source_passage_ranks": [1]}
                ],
            }
            (certification_run / "summary.json").write_text(
                json.dumps(summary)
            )
            (certification_run / "certified.json").write_text(
                json.dumps([row])
            )
            enriched, provenance = enrich_records_with_certification(
                records=[record],
                certification_run=certification_run,
                source_run=root / "source-run",
                source_manifest=source_manifest,
            )

        self.assertFalse(enriched[0].certified)
        self.assertEqual(
            enriched[0].certification_one_vote_short_keywords, ("Sinha",)
        )
        self.assertTrue(provenance["checks_passed"])

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

    def test_gold_alias_substitution_replaces_all_distinct_gold_spans(self):
        generated = gold_alias_substitution(
            "What the World Needs Now - song and lyrics by Missi Hale\n"
            "Steve Mazzaro and Shaley Scott are listed.",
            target="Shaley Scott",
            gold_aliases=["Missi Hale", "Steve Mazzaro"],
            text_processor=self.processor,
        )
        self.assertEqual(generated.status, "ok")
        self.assertEqual(
            generated.text,
            "What the World Needs Now - song and lyrics by Shaley Scott\n"
            "Shaley Scott and Shaley Scott are listed.",
        )
        self.assertEqual(
            generated.removed_segments,
            ("Missi Hale", "Steve Mazzaro"),
        )
        self.assertTrue(generated.target_present)

    def test_gold_alias_substitution_prefers_longest_overlapping_alias(self):
        generated = gold_alias_substitution(
            "Assembly\nDr. Rajendra Prasad was elected president. "
            "Sachidananda Sinha was temporary president.",
            target="Sachidananda Sinha",
            gold_aliases=["Rajendra Prasad", "Dr. Rajendra Prasad"],
            text_processor=self.processor,
        )
        self.assertEqual(generated.status, "ok")
        self.assertEqual(
            generated.text,
            "Assembly\nSachidananda Sinha was elected president. "
            "Sachidananda Sinha was temporary president.",
        )
        self.assertEqual(generated.removed_segments, ("Dr. Rajendra Prasad",))

    def test_gold_alias_substitution_preserves_currency_punctuation(self):
        generated = gold_alias_substitution(
            "Federal jurisdiction starts at $75,000. "
            "CAFA has a separate $5 million threshold.",
            target="$5 million",
            gold_aliases=["$75,000", "75,000 dollars"],
            text_processor=self.processor,
        )
        self.assertEqual(generated.status, "ok")
        self.assertEqual(
            generated.text,
            "Federal jurisdiction starts at $5 million. "
            "CAFA has a separate $5 million threshold.",
        )
        self.assertNotIn("$$", generated.text)

    def test_gold_alias_substitution_requires_target_and_gold_in_source(self):
        target_absent = gold_alias_substitution(
            "Rajendra Prasad was elected president.",
            target="Sachidananda Sinha",
            gold_aliases=["Rajendra Prasad"],
            text_processor=self.processor,
        )
        self.assertEqual(target_absent.status, "target_absent_from_source")

        gold_absent = gold_alias_substitution(
            "Sachidananda Sinha was temporary president.",
            target="Sachidananda Sinha",
            gold_aliases=["Rajendra Prasad"],
            text_processor=self.processor,
        )
        self.assertEqual(gold_absent.status, "no_gold_alias_in_source")

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

    def test_nlp_and_prometheus_selectors_are_parallel_over_structural_pool(self):
        responses = tuple(f"response {index}" for index in range(10))
        mapping = {
            response: ({"Sinha", "temporary"} if index == 0 else {"Prasad"})
            for index, response in enumerate(responses)
        }
        record = QuestionRecord(
            dataset="open_nq",
            query_index=7,
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

        nlp_outcome = plan_attack(
            record,
            extractor,
            self.processor,
            stopwords=set(),
            config=AttackConfig(target_selector="1+2"),
            candidate_analyzer=MappingCandidateAnalyzer("Sinha"),
        )
        judge_outcome = plan_attack(
            record,
            extractor,
            self.processor,
            stopwords=set(),
            config=AttackConfig(target_selector="3"),
            candidate_judge=MappingCandidateJudge(
                {"Sinha": 2, "temporary": 5}
            ),
        )

        self.assertEqual(nlp_outcome.selected.target, "Sinha")
        self.assertEqual(judge_outcome.selected.target, "temporary")
        self.assertEqual(len(nlp_outcome.selector_assessments), 2)
        self.assertEqual(len(judge_outcome.selector_assessments), 2)

    def test_prometheus_prompt_and_result_parser_are_reproducible(self):
        candidate_record = QuestionRecord(
            dataset="open_nq",
            query_index=4,
            question="Who was president?",
            gold_aliases=("Prasad",),
            passages=("Sinha was temporary president.", "bottom"),
            responses=("Sinha", "bottom"),
            clean_final_answer="Prasad",
            clean_correct=True,
        )
        mapping = {"Sinha": {"Sinha"}, "bottom": {"bottom"}}
        outcome = plan_attack(
            candidate_record,
            MappingKeywordExtractor(mapping),
            self.processor,
            stopwords=set(),
            config=AttackConfig(alpha=1.0),
        )
        prompt = build_prometheus_prompt(candidate_record, outcome.selected)
        decision = parse_prometheus_response(
            prompt, "Feedback: specific wrong person [RESULT] 5"
        )
        self.assertIn("Accepted gold answers: Prasad", prompt)
        self.assertEqual(decision.score, 5)
        self.assertEqual(decision.feedback, "specific wrong person")

    def test_nlp_expected_answer_types_are_deterministic(self):
        expected = SpacyCandidateQualityAnalyzer.expected_answer_type
        self.assertEqual(expected("Who wrote the novel?"), "person")
        self.assertEqual(expected("Where was the film made?"), "location")
        self.assertEqual(expected("When did trade stop?"), "time")
        self.assertEqual(expected("How many episodes aired?"), "number")
        self.assertEqual(expected("What was the dollar amount?"), "money")
        self.assertEqual(expected("What was the ship called?"), "open")

    def test_nlp_shape_checks_reject_wrong_type_and_initials(self):
        year = AnalyzerToken("2013", pos="NUM")
        canadian = AnalyzerToken("Canadian", pos="PROPN")
        g_token = AnalyzerToken("G", pos="PROPN")
        period = AnalyzerToken(".", lemma=".", pos="PUNCT", punct=True)
        fairmont_tokens = [
            AnalyzerToken(word, pos="PROPN")
            for word in ("Fairmont", "Chateau", "Whistler")
        ]
        documents = {
            "2013": AnalyzerDoc([year], [AnalyzerSpan([year], "DATE")]),
            "2013, Canadian": AnalyzerDoc(
                [year, canadian],
                [AnalyzerSpan([year], "DATE"), AnalyzerSpan([canadian], "NORP")],
            ),
            "G.": AnalyzerDoc([g_token, period]),
            "G. S. W. (no last name given)": AnalyzerDoc([g_token, period]),
            "Fairmont Chateau Whistler": AnalyzerDoc(
                fairmont_tokens,
                [AnalyzerSpan(fairmont_tokens, "ORG")],
            ),
        }
        analyzer = SpacyCandidateQualityAnalyzer(AnalyzerNlp(documents))
        record = QuestionRecord(
            dataset="open_nq",
            query_index=11,
            question="where was the film made",
            gold_aliases=("British Columbia",),
            passages=("p", "bottom"),
            responses=("r", "bottom"),
            clean_correct=True,
        )

        year_result = analyzer.analyze(
            record,
            SimpleNamespace(target="2013", source_response="2013, Canadian"),
        )
        initials_result = analyzer.analyze(
            record,
            SimpleNamespace(
                target="G.", source_response="G. S. W. (no last name given)"
            ),
        )
        place_result = analyzer.analyze(
            record,
            SimpleNamespace(
                target="Fairmont Chateau Whistler",
                source_response="Fairmont Chateau Whistler",
            ),
        )

        self.assertEqual(year_result["reason"], "answer_type_conflict")
        self.assertEqual(initials_result["reason"], "meta_nonanswer")
        self.assertTrue(place_result["eligible"])

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
