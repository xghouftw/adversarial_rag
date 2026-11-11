#!/usr/bin/env python3
"""Run the cache-only one-passage promotion target selector."""

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
    load_upstream_question_records,
    plan_attacks,
    save_selection_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select mechanically promotable one-passage targets from a pinned "
            "RobustRAG reproduction run. This command makes no LLM calls."
        )
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--target-selector", default="raw_max_vote")
    parser.add_argument("--document-generator", default="delete_gold_segments")
    parser.add_argument(
        "--include-clean-incorrect",
        action="store_true",
        help="Profile all questions instead of requiring a correct clean answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, source_manifest = load_upstream_question_records(
        source_run=args.source_run,
        dataset_path=args.dataset_path,
        runner_dir=args.runner_dir,
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
    stopword_set = set(stopwords.words("english"))
    outcomes = plan_attacks(
        records=records,
        keyword_extractor=keyword_extractor,
        text_processor=text_processor,
        stopwords=stopword_set,
        config=config,
    )
    environment = {
        package: importlib.metadata.version(package)
        for package in ("spacy", "en-core-web-sm", "nltk", "joblib")
    }
    output_dir = save_selection_run(
        output_root=args.output_root,
        source_run=args.source_run,
        source_manifest=source_manifest,
        config=config,
        outcomes=outcomes,
        extra_manifest={"packages": environment},
    )
    statuses = Counter(outcome.status for outcome in outcomes)
    summary = {
        "output_dir": str(output_dir),
        "source_run": str(args.source_run),
        "config": {
            "alpha": config.alpha,
            "beta": config.beta,
            "target_selector": config.target_selector,
            "document_generator": config.document_generator,
            "require_clean_correct": config.require_clean_correct,
        },
        "statuses": dict(statuses),
        "selected_questions": sum(
            outcome.selected is not None for outcome in outcomes
        ),
        "candidate_rows": sum(outcome.candidate_count for outcome in outcomes),
    }
    print(json.dumps(summary, indent=2))
    print("PROMOTION_SELECTION_RESULT=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
