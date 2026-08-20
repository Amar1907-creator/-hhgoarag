"""Explicit validation for MSMARCO-XI records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

REQUIRED_FIELDS = (
    "source_lang", "target_lang", "meta", "query", "Answer", "query_id",
    "query_type", "passages", "Eng_Query", "Eng_Answer",
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_record(record: Any) -> ValidationResult:
    """Validate documented fields without throwing away or mutating the source row."""
    result = ValidationResult()
    if not isinstance(record, Mapping):
        result.errors.append("record is not an object")
        return result
    for name in REQUIRED_FIELDS:
        if name not in record:
            result.errors.append(f"missing required field: {name}")
    if result.errors:
        return result
    for name in ("source_lang", "target_lang", "query", "Answer", "query_type", "Eng_Query", "Eng_Answer"):
        if not isinstance(record[name], str):
            result.errors.append(f"field {name} must be a string")
        elif not record[name].strip():
            result.warnings.append(f"field {name} is empty")
    if not isinstance(record["query_id"], int) or isinstance(record["query_id"], bool):
        result.errors.append("field query_id must be an integer")
    if not isinstance(record["meta"], Mapping):
        result.errors.append("field meta must be an object")
    if not isinstance(record["passages"], Mapping):
        result.errors.append("field passages must be an object")
        return result
    passages = record["passages"]
    expected = ("is_selected", "English_passages", "Translated_passages")
    for name in expected:
        if name not in passages:
            result.errors.append(f"missing passages.{name}")
        elif not isinstance(passages[name], list):
            result.errors.append(f"passages.{name} must be a list")
    if any(name not in passages or not isinstance(passages.get(name), list) for name in expected):
        return result
    sizes = {name: len(passages[name]) for name in expected}
    if len(set(sizes.values())) != 1:
        result.errors.append(f"passage list lengths differ: {sizes}")
        return result
    for index, value in enumerate(passages["Translated_passages"]):
        if not isinstance(value, str):
            result.errors.append(f"Translated_passages[{index}] must be a string")
        elif not value.strip():
            result.warnings.append(f"Translated_passages[{index}] is empty")
    for index, value in enumerate(passages["English_passages"]):
        if not isinstance(value, str):
            result.errors.append(f"English_passages[{index}] must be a string")
    for index, value in enumerate(passages["is_selected"]):
        if value not in (0, 1, False, True):
            result.errors.append(f"is_selected[{index}] must be 0 or 1")
    return result
