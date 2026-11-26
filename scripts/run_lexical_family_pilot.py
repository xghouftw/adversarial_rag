"""Paired final-answer evaluation on clean and previously recorded responses."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from defense_selectors import LexicalFamilySelector
from promotion_attack import (
    SpacyKeywordExtractor, aggregate_keyword_sets, contains_normalized_phrase, contains_upstream_answer,
    load_upstream_question_records,
)
from reproduction.robustrag_setup import file_sha256


def prepare_question(record, frozen, params, extractor, selector, stopwords, template):
    if (frozen["query_index"], frozen["dataset"], frozen["question"]) != (
        record.query_index, record.dataset, record.question
    ):
        raise ValueError(f"Frozen result does not match question {record.query_index}")
    # The frozen run displaced the last processed passage, including on short lists.
    responses_by_condition = {
        "clean": record.responses,
        "frozen": (*record.responses[:-1], frozen["injected_response"]),
    }
    arms = []
    for condition, responses in responses_by_condition.items():
        votes = [extractor.extract(text) for text in responses]
        exact = aggregate_keyword_sets(votes, params["alpha"], params["beta"], stopwords)
        hints = {
            "exact": exact.retained_keywords,
            "lexical_family": selector.select(exact.retained_keywords, responses, votes),
        }
        for name, keywords in hints.items():
            prompt = template.format(query_str=record.question, hints=", ".join(keywords))
            if condition == "frozen" and name == "exact":
                if list(keywords) != frozen["attacked_retained_keywords"] or prompt != frozen["final_prompt"]:
                    raise ValueError(f"Frozen hint or prompt replay mismatch on question {record.query_index}")
            arms.append({
                "condition": condition, "selector": name, "hints": list(keywords),
                "answering_count": exact.answering_count, "threshold": exact.threshold,
                # The original promotion runner calls query_final even on an empty hint bag.
                "prompt": prompt if condition == "frozen" or exact.answering_count else None,
            })
    return {
        "query_index": record.query_index, "question": record.question,
        "gold_aliases": record.gold_aliases, "target": frozen["target"],
        "source_rank": frozen["source_rank"], "arms": arms,
        "archived_stages": {key: frozen[key] for key in
                            ("response_realized", "promotion_realized", "attacked_gold_retained")},
    }


def evaluate_question(question, query, answers=None):
    # Each arm gets the same decoding settings. Identical prompts share one call.
    if answers is None:
        answers = {}
    previous_size = len(answers)
    for arm in question["arms"]:
        prompt = arm["prompt"]
        if prompt is None:
            answer = "I don't know."
        else:
            if prompt not in answers:
                answers[prompt] = query(prompt)
            answer = answers[prompt]
        arm["answer"] = answer
        arm["gold_substring_present"] = any(
            contains_upstream_answer(answer, alias)
            for alias in question["gold_aliases"] if alias.strip()
        )
        arm["automatic_gold_present"] = arm["gold_substring_present"]
        arm["automatic_target_present"] = contains_normalized_phrase(answer, question["target"])
        arm["automatic_strict_targeted_success"] = (
            arm["automatic_target_present"] and not arm["automatic_gold_present"]
        )
    question["distinct_final_calls"] = len(answers) - previous_size


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def frozen_sweep_success(candidates):
    """Compute the original question-level metrics over the saved candidate order."""
    flags_by_selector = {}
    for candidate in candidates:
        for arm in candidate["arms"]:
            if arm["condition"] == "frozen":
                flags_by_selector.setdefault(arm["selector"], []).append(arm["automatic_strict_targeted_success"])
    return {
        name: {"success_at_1": flags[0], "success_at_3": any(flags[:3]), "success_at_all": any(flags)}
        for name, flags in flags_by_selector.items()
    }


def pilot_indices(archived, limit=None, query_indices=None):
    """Select archived questions in the same order as the original live runner."""
    if not archived or len(set(archived)) != len(archived):
        raise ValueError("Expected a nonempty archive of unique query indices.")
    if query_indices is not None:
        if not query_indices or len(set(query_indices)) != len(query_indices):
            raise ValueError("Explicit query indices must be nonempty and unique.")
        missing = [index for index in query_indices if index not in archived]
        if missing:
            raise ValueError(f"No archived result for query indices {missing}.")
        return list(query_indices)
    if limit is None or limit < 1:
        raise ValueError("The pilot question limit must be positive.")
    return archived[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "dataset-file", "runner", "frozen-run", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=int)
    scope.add_argument("--query-indices", nargs="+", type=int)
    args = parser.parse_args()

    import spacy
    import torch
    from nltk.corpus import stopwords

    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select an A100 GPU in the experiment environment.")
    frozen_manifest = json.loads((args.frozen_run / "manifest.json").read_text())
    source_hash = file_sha256(args.source_run / "manifest.json")
    if (frozen_manifest["status"] != "complete"
            or frozen_manifest["source_manifest_sha256"] != source_hash
            or frozen_manifest["config"]["attack_budget"] != 1):
        raise ValueError("Expected a completed, matching one-passage result archive.")
    archived = frozen_manifest["query_indices"]
    indices = pilot_indices(archived, args.limit, args.query_indices)
    records, source = load_upstream_question_records(args.source_run, args.dataset_file, args.runner)
    params = source["parameters"]
    if (params["model"] != "mistralai/Mistral-7B-Instruct-v0.2" or not params["greedy_decoding"]
            or any(params[k] != frozen_manifest["config"][k] for k in ("alpha", "beta", "top_k"))):
        raise ValueError("Source model or aggregation parameters do not match the pilot.")
    from src.models import create_model
    from src.prompt_template import MISTRAL_TMPL

    nlp = spacy.load("en_core_web_sm")
    extractor = SpacyKeywordExtractor(nlp)
    selector = LexicalFamilySelector(nlp)
    stop = set(stopwords.words("english"))
    questions = []
    hashes = {}
    by_index = {r.query_index: r for r in records}
    for index in indices:
        path = args.frozen_run / "results" / f"query-{index:03d}.json"
        hashes[path.name] = file_sha256(path)
        sweep = json.loads(path.read_text())
        if not sweep["results"] or len(sweep["results"]) != sweep["attempted_candidates"]:
            raise ValueError(f"Incomplete archived sweep for query {index}")
        candidates = [
            prepare_question(by_index[index], frozen, params, extractor, selector, stop, MISTRAL_TMPL["qa-hint"])
            for frozen in sweep["results"]
        ]
        questions.append(candidates)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_root / f"live-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    print("Live pilot output:", out, flush=True)
    manifest = {
        "status": "running", "kind": "paired_final_answer_pilot",
        "dataset": source["dataset"], "dataset_sha256": source["dataset_sha256"],
        "query_indices": indices, "sample_size": len(indices), "parameters": params,
        "requested_limit": args.limit, "requested_query_indices": args.query_indices,
        "archived_question_count": len(archived), "source_question_count": len(records),
        "source_run": str(args.source_run), "source_manifest_sha256": source_hash,
        "frozen_run": str(args.frozen_run), "frozen_manifest_sha256": file_sha256(args.frozen_run / "manifest.json"),
        "frozen_result_sha256": hashes, "frozen_config": frozen_manifest["config"],
        "upstream_commit": source["upstream_commit"], "gpu": torch.cuda.get_device_name(0),
        "source_gpu": source["environment"]["gpu"], "prompt_template": MISTRAL_TMPL["qa-hint"],
        "selector_sha256": file_sha256(Path(__file__).resolve().parents[1] / "defense_selectors.py"),
        "packages": {k: version(k) for k in ("torch", "transformers", "spacy", "en-core-web-sm")},
    }
    write_json(out / "manifest.json", manifest)
    try:
        model = create_model("mistral7b")
        manifest["model_revision"] = model.model.config._commit_hash
        manifest["generation_kwargs"] = model.generation_kwargs
        strict_counts = {}
        scores = {}
        total_calls = 0
        for candidates in questions:
            answers = {}
            for candidate in candidates:
                evaluate_question(candidate, model.query, answers)
            first = candidates[0]
            for arm in first["arms"]:
                key = f"{arm['condition']}/{arm['selector']}"
                scores[key] = scores.get(key, 0) + int(arm["automatic_gold_present"])
            success = frozen_sweep_success(candidates)
            for name, metrics in success.items():
                totals = strict_counts.setdefault(name, dict.fromkeys(metrics, 0))
                for metric, value in metrics.items():
                    totals[metric] += int(value)
            index = first["query_index"]
            # Keep the original top-one fields available at their existing paths.
            result = dict(first, results=candidates, success=success, distinct_final_calls=len(answers))
            write_json(out / f"query-{index:03d}.json", result)
            total_calls += len(answers)
            print(f"Finished Q{index} ({len(answers)} final calls)", flush=True)
        summary = {"question_count_per_arm": len(indices), "gold_substring_counts": scores,
                   "gold_substring_rank": 1, "frozen_strict_success_counts": strict_counts,
                   "distinct_final_calls": total_calls}
        write_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
