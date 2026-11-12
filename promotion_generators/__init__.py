"""Built-in promotion document generators."""

from .base import DocumentGenerator, GeneratedDocument, TextProcessor
from .delete_gold import delete_gold_segments
from .gold_alias import gold_alias_substitution


DOCUMENT_GENERATORS: dict[str, DocumentGenerator] = {
    "delete_gold_segments": delete_gold_segments,
    "gold_alias_substitution": gold_alias_substitution,
}


def register_document_generator(name: str, generator: DocumentGenerator) -> None:
    if not name or name in DOCUMENT_GENERATORS:
        raise ValueError(f"document generator already registered or invalid: {name!r}")
    DOCUMENT_GENERATORS[name] = generator


__all__ = [
    "DOCUMENT_GENERATORS",
    "DocumentGenerator",
    "GeneratedDocument",
    "TextProcessor",
    "delete_gold_segments",
    "gold_alias_substitution",
    "register_document_generator",
]
