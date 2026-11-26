import unittest
from importlib.util import find_spec

from defense_selectors import LexicalFamilySelector
from promotion_attack import (
    AttackConfig, QuestionRecord, SpacyKeywordExtractor, SpacyTextProcessor,
    TextGeneration, execute_live_candidates, plan_attack,
)
from scripts.run_lexical_family_pilot import evaluate_question, frozen_sweep_success, pilot_indices


class PairedAnswerTests(unittest.TestCase):
    def test_replay_scores_the_full_archived_prefix(self):
        candidates = [
            {"arms": [
                {"condition": "clean", "selector": "exact", "automatic_strict_targeted_success": True},
                {"condition": "frozen", "selector": "exact", "automatic_strict_targeted_success": rank == 4},
                {"condition": "frozen", "selector": "lexical_family", "automatic_strict_targeted_success": rank == 3},
            ]}
            for rank in range(1, 5)
        ]
        self.assertEqual(frozen_sweep_success(candidates), {
            "exact": {"success_at_1": False, "success_at_3": False, "success_at_all": True},
            "lexical_family": {"success_at_1": False, "success_at_3": True, "success_at_all": True},
        })

    def test_live_limit_is_a_cap_and_explicit_indices_preserve_order(self):
        archived = [2, 3, 5]
        self.assertEqual(pilot_indices(archived, limit=100), archived)
        self.assertEqual(pilot_indices(archived, limit=2), [2, 3])
        self.assertEqual(pilot_indices(archived, query_indices=[5, 2]), [5, 2])
        with self.assertRaisesRegex(ValueError, "No archived result"):
            pilot_indices(archived, query_indices=[4])

    def test_abstention_and_identical_prompts(self):
        question = {
            "gold_aliases": ["Ontario"],
            "target": "Victoria",
            "arms": [{"prompt": prompt} for prompt in (None, "A", "A", "B")],
        }
        calls = []

        def query(prompt):
            calls.append(prompt)
            return {"A": "Ontario", "B": "Victoria"}[prompt]

        evaluate_question(question, query)
        self.assertEqual(calls, ["A", "B"])
        self.assertEqual(question["distinct_final_calls"], 2)
        self.assertEqual(question["arms"][0]["answer"], "I don't know.")
        self.assertEqual([a["gold_substring_present"] for a in question["arms"]],
                         [False, True, True, False])
        self.assertEqual([a["automatic_strict_targeted_success"] for a in question["arms"]],
                         [False, False, False, True])


@unittest.skipUnless(find_spec("en_core_web_sm"), "requires pinned spaCy English model")
class LexicalFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import spacy

        cls.nlp = spacy.load("en_core_web_sm")
        cls.extractor = SpacyKeywordExtractor(cls.nlp)
        cls.selector = LexicalFamilySelector(cls.nlp)

    def select(self, hints, *responses):
        votes = [self.extractor.extract(text) for text in responses]
        return self.selector.select(hints, responses, votes)

    def test_hotel_and_separate_town(self):
        hotel = "Fairmont Chateau Whistler"
        hints = (hotel, "Whistler", "Fairmont", "Chateau")
        self.assertEqual(self.select(hints, hotel, hotel), (hotel,))
        for responses in [(hotel, "Whistler"), (f"The {hotel} is in Whistler.",)]:
            with self.subTest(responses=responses):
                self.assertEqual(self.select(hints, *responses), (hotel, "Whistler"))

    def test_shared_surname_abstention_and_order(self):
        hints = ("James Smith", "December", "Alice Smith", "Smith", "James", "Alice")
        result = self.select(
            hints, "Alice Smith arrived in December.", "James Smith", "I don't know, James"
        )
        self.assertEqual(result, ("James Smith", "December", "Alice Smith", "Smith"))

    def test_lemma_anchor_and_standalone_inflection(self):
        hints = ("large red fox", "fox", "large", "red")
        self.assertEqual(self.select(hints, "Large red foxes ran."), (hints[0],))
        self.assertEqual(self.select(hints, "Large red foxes ran.", "The fox slept."), hints[:2])

    def test_final_prompt_uses_selected_hints_after_displacement(self):
        hotel = "Fairmont Chateau Whistler"
        responses = (hotel,) + ("Victoria",) * 8 + ("Whistler",)
        record = QuestionRecord(
            dataset="example", query_index=0, question="Where is it?",
            gold_aliases=("Victoria",), passages=responses, responses=responses,
            clean_correct=True,
        )
        config = AttackConfig()
        selection = plan_attack(
            record, self.extractor, SpacyTextProcessor(self.nlp), set(), config,
        )
        for selector in (None, self.selector):
            with self.subTest(selector=selector):
                result = execute_live_candidates(
                    record, selection, self.extractor, set(), config,
                    query_isolated=lambda *_: TextGeneration("test", hotel),
                    query_final=lambda _, hints: TextGeneration(hints, "Victoria"),
                    hint_selector=selector, max_candidates=1,
                ).results[0]
                expected = result.attacked_retained_keywords
                self.assertIn("Whistler", expected)
                if selector:
                    expected = (hotel, "Victoria")
                self.assertEqual(result.final_hint_keywords, expected)
                self.assertEqual(result.final_prompt, ", ".join(expected))
