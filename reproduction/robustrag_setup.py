"""Prepare the pinned RobustRAG runner and checksum-verified datasets.

The upstream source is vendored under ``third_party/`` so Colab runs never
need to clone GitHub.  A temporary runner copy receives the compatibility
change used by the reproduction: imports needed only by the unused GPT paths
are deferred, while the Mistral path stays identical to upstream.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "9bc35b2fa5fa7d1088383b2789ec19a512d316b1"
UPSTREAM_SOURCE_PATH = Path("third_party/robustrag_upstream")
DATA_BASE_URL = (
    "https://raw.githubusercontent.com/inspire-group/RobustRAG/"
    f"{UPSTREAM_COMMIT}/data"
)

SOURCE_HASHES = {
    "main.py": "f96fa6def3738ef388862280f6ea79c32f2055ccb92982c92f2b3b6cfa68171d",
    "src/attack.py": "ca0acff4d59802130c621e76516e76b6400ffa0c62698b9d7a625d2ce2ffac75",
    "src/dataset_utils.py": "ceecc4faaace8fc1dcb03f4995b46b9440fe3faf473a0b5df1db6f68a643fc14",
    "src/defense.py": "b8b1a84082c31d7510c148fc097f07ef507ed9f171c9cc57f31dacd7165632ef",
    "src/helper.py": "bf6d543b4b50257f9a331b7b4bdd44c650ca9618c8ebe4b256f1e45896d68c3c",
    "src/models.py": "c0215220c383decc76512a0797cb63f1f5d5c62e68d269a2e56fed7fc21d4df2",
    "src/prompt_template.py": "3f94a4d715f5d51fa30150e29640dd7df421b98aaae6c488d3e15773e4de5bc8",
}

DATASETS = {
    "realtimeqa": (
        "realtimeqa.json",
        "0924fd01734f90e7910d6cb5802d387aaf5811eeba650667a8e72d6b94d7941e",
    ),
    "open_nq": (
        "open_nq.json",
        "32a71eab0a71f67e8f07ca8fddf8d9ac4d18c841d92df3f77a0e0cfa11ad8f11",
    ),
}

_MODEL_IMPORT_REPLACEMENTS = (
    (
        "import litellm\nfrom litellm import batch_completion\nfrom openai import OpenAI\n",
        "",
    ),
    (
        "        self.client = OpenAI()",
        "        from openai import OpenAI\n        self.client = OpenAI()",
    ),
    (
        "        result = batch_completion(",
        "        from litellm import batch_completion\n        result = batch_completion(",
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_vendored_source(source_dir: Path) -> None:
    """Refuse to run if the committed upstream snapshot has changed."""
    source_dir = Path(source_dir)
    for relative_name, expected_hash in SOURCE_HASHES.items():
        source_path = source_dir / relative_name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing vendored RobustRAG source: {source_path}")
        actual_hash = file_sha256(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Vendored source checksum mismatch for {source_path}: {actual_hash}"
            )


def _replace_once(path: Path, before: str, after: str) -> None:
    source = path.read_text()
    if source.count(before) != 1:
        raise RuntimeError(f"Unexpected pinned source in {path}: {before!r}")
    path.write_text(source.replace(before, after, 1))


def build_runner(source_dir: Path, runner_dir: Path) -> None:
    """Copy verified source and apply only the documented import deferrals."""
    source_dir = Path(source_dir).resolve()
    runner_dir = Path(runner_dir).resolve()
    verify_vendored_source(source_dir)
    if (
        runner_dir == source_dir
        or source_dir in runner_dir.parents
        or runner_dir in source_dir.parents
    ):
        raise ValueError("Runner directory must be outside the vendored source tree.")

    runner_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{runner_dir.name}-", dir=runner_dir.parent)
    )
    try:
        for relative_name in SOURCE_HASHES:
            source_path = source_dir / relative_name
            destination = temporary / relative_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)

        models_path = temporary / "src/models.py"
        for before, after in _MODEL_IMPORT_REPLACEMENTS:
            _replace_once(models_path, before, after)

        if runner_dir.exists():
            shutil.rmtree(runner_dir)
        temporary.replace(runner_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(
        f"Prepared runner from vendored RobustRAG commit {UPSTREAM_COMMIT}: "
        f"{runner_dir}"
    )


def ensure_datasets(data_dir: Path, dataset_names: tuple[str, ...] = tuple(DATASETS)) -> None:
    """Reuse verified shared data, downloading only a missing pinned dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in dataset_names:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown RobustRAG dataset: {dataset_name}")
        filename, expected_hash = DATASETS[dataset_name]
        destination = data_dir / filename
        if destination.exists():
            actual_hash = file_sha256(destination)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Refusing to overwrite {destination}; checksum is {actual_hash}"
                )
            print("Reusing verified dataset:", destination)
            continue

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", dir=data_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            urllib.request.urlretrieve(f"{DATA_BASE_URL}/{filename}", temporary)
            actual_hash = file_sha256(temporary)
            if actual_hash != expected_hash:
                raise RuntimeError(f"Checksum mismatch for {filename}: {actual_hash}")
            temporary.replace(destination)
            print("Saved verified dataset:", destination)
        finally:
            if temporary.exists():
                temporary.unlink()


def prepare_robustrag(
    project_dir: Path,
    runner_dir: Path,
    data_dir: Path,
    dataset_names: tuple[str, ...] = tuple(DATASETS),
) -> Path:
    """Prepare code and data and return the committed upstream source path."""
    source_dir = Path(project_dir) / UPSTREAM_SOURCE_PATH
    build_runner(source_dir, runner_dir)
    ensure_datasets(data_dir, dataset_names)
    return source_dir
