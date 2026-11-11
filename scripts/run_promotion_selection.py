#!/usr/bin/env python3
"""Run a modular one-passage promotion target selector."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path

import spacy
from nltk.corpus import stopwords

from promotion_attack import (
    AttackConfig,
    SpacyKeywordExtractor,
    SpacyTextProcessor,
    enrich_records_with_certification,
    load_upstream_question_records,
    plan_attacks,
    resolve_target_selector,
    save_selection_run,
)
from promotion_selectors.nlp import SpacyCandidateQualityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select mechanically promotable one-passage targets from a pinned "
            "RobustRAG reproduction run. Selectors 1 and 1+2 make no LLM "
            "calls; selector 3 loads Prometheus-2-7B."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--certification-run",
        type=Path,
        help=(
            "Optional checked certification replay containing certified.json "
            "and summary.json. The command refuses artifacts whose replay "
            "checks did not pass."
        ),
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument(
        "--target-selector",
        default="1",
        help="1/raw_max_vote, 1+2/nlp_answer_shape, or 3/prometheus_judge",
    )
    parser.add_argument("--document-generator", default="delete_gold_segments")
    parser.add_argument(
        "--include-clean-incorrect",
        action="store_true",
        help="Profile all questions instead of requiring a correct clean answer.",
    )
    parser.add_argument(
        "--prometheus-model",
        default="prometheus-eval/prometheus-7b-v2.0",
    )
    parser.add_argument(
        "--prometheus-revision",
        default="66ffb1fc20beebfb60a3964a957d9011723116c5",
    )
    parser.add_argument("--prometheus-cache-dir", type=Path)
    parser.add_argument("--prometheus-batch-size", type=int, default=4)
    parser.add_argument("--prometheus-max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, source_manifest = load_upstream_question_records(
        source_run=args.source_run,
        dataset_path=args.dataset_path,
        runner_dir=args.runner_dir,
    )
    certification_provenance = None
    if args.certification_run is not None:
        records, certification_provenance = enrich_records_with_certification(
            records=records,
            certification_run=args.certification_run,
            source_run=args.source_run,
            source_manifest=source_manifest,
        )
    source_parameters = source_manifest["parameters"]
    config = AttackConfig(
        alpha=(
            float(source_parameters["alpha"])
            if args.alpha is None
            else args.alpha
        ),
        beta=(
            float(source_parameters["beta"])
            if args.beta is None
            else args.beta
        ),
        top_k=int(source_parameters["top_k"]),
        attack_budget=1,
        target_selector=args.target_selector,
        document_generator=args.document_generator,
        require_clean_correct=not args.include_clean_incorrect,
    )
    config.validate()

    nlp = spacy.load("en_core_web_sm")
    keyword_extractor = SpacyKeywordExtractor(nlp)
    text_processor = SpacyTextProcessor(nlp)
    selector_name = resolve_target_selector(config.target_selector)
    candidate_analyzer = (
        SpacyCandidateQualityAnalyzer(nlp)
        if selector_name == "nlp_answer_shape"
        else None
    )
    candidate_judge = None
    if selector_name == "prometheus_judge":
        from promotion_selectors.prometheus import PrometheusCandidateJudge

        candidate_judge = PrometheusCandidateJudge(
            model_name=args.prometheus_model,
            revision=args.prometheus_revision,
            cache_dir=args.prometheus_cache_dir,
            batch_size=args.prometheus_batch_size,
            max_new_tokens=args.prometheus_max_new_tokens,
        )
    stopword_set = set(stopwords.words("english"))
    outcomes = plan_attacks(
        records=records,
        keyword_extractor=keyword_extractor,
        text_processor=text_processor,
        stopwords=stopword_set,
        config=config,
        candidate_analyzer=candidate_analyzer,
        candidate_judge=candidate_judge,
    )
    environment = {
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "spacy",
                "en-core-web-sm",
                "nltk",
                "joblib",
                "torch",
                "transformers",
            )
        },
        "certification": certification_provenance,
        "selector_model": (
            candidate_judge.provenance()
            if candidate_judge is not None
            else None
        ),
    }
    output_dir = save_selection_run(
        output_root=args.output_root,
        source_run=args.source_run,
        source_manifest=source_manifest,
        config=config,
        outcomes=outcomes,
        extra_manifest=environment,
    )
    statuses = Counter(outcome.status for outcome in outcomes)
    assessment_reasons = Counter(
        assessment.reason
        for outcome in outcomes
        for assessment in outcome.selector_assessments
    )
    summary = {
        "output_dir": str(output_dir),
        "source_run": str(args.source_run),
        "config": {
            "alpha": config.alpha,
            "beta": config.beta,
            "target_selector": config.target_selector,
            "resolved_target_selector": selector_name,
            "document_generator": config.document_generator,
            "require_clean_correct": config.require_clean_correct,
        },
        "statuses": dict(statuses),
        "selected_questions": sum(
            outcome.selected is not None for outcome in outcomes
        ),
        "candidate_rows": sum(outcome.candidate_count for outcome in outcomes),
        "structural_candidate_rows": sum(
            len(outcome.selector_assessments) or outcome.candidate_count
            for outcome in outcomes
        ),
        "selector_rejected_rows": sum(
            not assessment.eligible
            for outcome in outcomes
            for assessment in outcome.selector_assessments
        ),
        "certified_questions": sum(
            outcome.certified is True for outcome in outcomes
        ),
        "certification_frontier_rows": sum(
            len(outcome.certification_one_vote_short_keywords)
            for outcome in outcomes
        ),
        "selector_assessment_reasons": dict(assessment_reasons),
    }
    print(json.dumps(summary, indent=2))
    print("PROMOTION_SELECTION_RESULT=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
