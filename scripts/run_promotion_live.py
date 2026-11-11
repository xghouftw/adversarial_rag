#!/usr/bin/env python3
"""Run an explicitly bounded live promotion-attack pilot on a GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import spacy
import torch
from nltk.corpus import stopwords

from promotion_attack import (
    AttackConfig,
    SpacyKeywordExtractor,
    SpacyTextProcessor,
    UpstreamQueryAdapter,
    create_live_run,
    enrich_records_with_certification,
    execute_live_candidates,
    file_sha256,
    finalize_live_run,
    load_upstream_question_records,
    plan_attacks,
    resolve_target_selector,
    save_live_sweep,
)
from promotion_selectors.nlp import SpacyCandidateQualityAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded live promotion pilot from a frozen selection "
            "artifact. This command loads Mistral-7B and makes GPU calls."
        )
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=int)
    scope.add_argument("--query-indices", type=int, nargs="+")
    parser.add_argument(
        "--max-candidates",
        default="1",
        help="Positive integer for success@k, or 'all' for exhaustive search.",
    )
    return parser.parse_args()


def candidate_signatures(outcome: dict) -> list[tuple]:
    return [
        (
            candidate["target"],
            candidate["source_rank"],
            candidate["document"]["text"],
            candidate["simulated_attacked_margin"],
            tuple(candidate["simulated_retained_keywords"]),
        )
        for candidate in outcome["candidates"]
    ]


def parse_max_candidates(value: str) -> int | None:
    if value.casefold() == "all":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("--max-candidates must be positive or 'all'")
    return parsed


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    max_candidates = parse_max_candidates(args.max_candidates)

    selection_manifest = json.loads(
        (args.selection_run / "manifest.json").read_text()
    )
    if selection_manifest.get("status") != "complete":
        raise ValueError("selection run is not complete")
    if selection_manifest.get("kind") not in {
        "cache_only_promotion_selection",
        "llm_judged_promotion_selection",
    }:
        raise ValueError("--selection-run is not a promotion selection artifact")
    saved_outcomes = json.loads(
        (args.selection_run / "selections.json").read_text()
    )
    saved_by_index = {
        int(outcome["query_index"]): outcome for outcome in saved_outcomes
    }

    source_run = Path(selection_manifest["source_run"])
    source_manifest_path = source_run / "manifest.json"
    if (
        file_sha256(source_manifest_path)
        != selection_manifest["source_manifest_sha256"]
    ):
        raise ValueError("source-run manifest changed after target selection")
    records, source_manifest = load_upstream_question_records(
        source_run=source_run,
        dataset_path=args.dataset_path,
        runner_dir=args.runner_dir,
    )
    certification = selection_manifest.get("environment", {}).get(
        "certification"
    )
    if certification is not None:
        certification_run = Path(certification["certification_run"])
        if (
            file_sha256(certification_run / "summary.json")
            != certification["certification_summary_sha256"]
            or file_sha256(certification_run / "certified.json")
            != certification["certified_json_sha256"]
        ):
            raise ValueError(
                "certification artifact changed after target selection"
            )
        records, current_certification = enrich_records_with_certification(
            records=records,
            certification_run=certification_run,
            source_run=source_run,
            source_manifest=source_manifest,
        )
        if current_certification != certification:
            raise ValueError("certification provenance changed after selection")
    saved_config = selection_manifest["config"]
    config = AttackConfig(**saved_config)
    config.validate()

    nlp = spacy.load("en_core_web_sm")
    extractor = SpacyKeywordExtractor(nlp)
    processor = SpacyTextProcessor(nlp)
    selector_name = resolve_target_selector(config.target_selector)
    candidate_analyzer = (
        SpacyCandidateQualityAnalyzer(nlp)
        if selector_name == "nlp_answer_shape"
        else None
    )
    candidate_judge = None
    if selector_name == "prometheus_judge":
        from promotion_selectors.prometheus import RecordedCandidateJudge

        candidate_judge = RecordedCandidateJudge(saved_outcomes)
    stopword_set = set(stopwords.words("english"))
    selections = plan_attacks(
        records=records,
        keyword_extractor=extractor,
        text_processor=processor,
        stopwords=stopword_set,
        config=config,
        candidate_analyzer=candidate_analyzer,
        candidate_judge=candidate_judge,
    )

    # Refuse live calls if current code/environment changes the frozen ranking.
    for selection in selections:
        saved = saved_by_index.get(selection.query_index)
        if saved is None:
            raise ValueError(
                f"selection artifact is missing query {selection.query_index}"
            )
        current_signatures = [
            (
                candidate.target,
                candidate.source_rank,
                candidate.document.text,
                candidate.simulated_attacked_margin,
                candidate.simulated_retained_keywords,
            )
            for candidate in selection.candidates
        ]
        if current_signatures != candidate_signatures(saved):
            raise ValueError(
                f"frozen candidate ranking changed for query {selection.query_index}"
            )

    selected_pairs = [
        (record, selection)
        for record, selection in zip(records, selections)
        if selection.selected is not None
    ]
    if args.query_indices is not None:
        requested = list(args.query_indices)
        if len(set(requested)) != len(requested):
            raise ValueError("--query-indices must be unique")
        pairs_by_index = {
            record.query_index: (record, selection)
            for record, selection in selected_pairs
        }
        missing = [index for index in requested if index not in pairs_by_index]
        if missing:
            raise ValueError(f"queries have no frozen candidate: {missing}")
        selected_pairs = [pairs_by_index[index] for index in requested]
    else:
        selected_pairs = selected_pairs[: args.limit]

    if not selected_pairs:
        raise ValueError("no selected questions to run")
    if not torch.cuda.is_available():
        raise RuntimeError("select a GPU runtime before running live attacks")
    current_gpu = torch.cuda.get_device_name(0)
    source_gpu = str(source_manifest.get("environment", {}).get("gpu", ""))
    source_gpu_family = next(
        (family for family in ("L4", "A100") if family in source_gpu),
        None,
    )
    if source_gpu_family is None:
        raise RuntimeError(
            f"source run does not identify a supported L4/A100 GPU: {source_gpu!r}"
        )
    if source_gpu_family not in current_gpu:
        raise RuntimeError(
            f"select an {source_gpu_family} runtime to match the source run; "
            f"current GPU is {current_gpu!r}"
        )

    query_indices = [record.query_index for record, _ in selected_pairs]
    environment = {
        "gpu": current_gpu,
        "source_gpu": source_gpu,
        "packages": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "transformers",
                "spacy",
                "en-core-web-sm",
                "nltk",
                "joblib",
            )
        },
        "selection_run": str(args.selection_run),
        "selection_manifest_sha256": file_sha256(
            args.selection_run / "manifest.json"
        ),
    }
    live_dir = create_live_run(
        output_root=args.output_root,
        source_run=source_run,
        source_manifest=source_manifest,
        config=config,
        query_indices=query_indices,
        max_candidates=max_candidates,
        extra_manifest=environment,
    )

    from src.models import create_model  # type: ignore

    model = create_model(
        "mistral7b",
        cache_path=str(live_dir / "mistral-attack-cache.z"),
    )
    for record, selection in selected_pairs:
        adapter = UpstreamQueryAdapter(model, record)
        sweep = execute_live_candidates(
            record=record,
            selection=selection,
            keyword_extractor=extractor,
            stopwords=stopword_set,
            config=config,
            query_isolated=adapter.query_isolated,
            query_final=adapter.query_final,
            max_candidates=max_candidates,
        )
        save_live_sweep(live_dir / "results", sweep)
        model.dump_cache()
        print(
            json.dumps(
                {
                    "query_index": record.query_index,
                    "attempted_candidates": sweep.attempted_candidates,
                    "unique_poison_documents": sweep.unique_poison_documents,
                    "success_at_1": sweep.success_at_1,
                    "success_at_3": sweep.success_at_3,
                    "success_at_all": sweep.success_at_all,
                }
            ),
            flush=True,
        )

    finalize_live_run(live_dir)
    print(json.dumps({"completed_live_run": str(live_dir)}, indent=2))


if __name__ == "__main__":
    main()
