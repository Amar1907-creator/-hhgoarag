"""Leakage-safe retrieval metrics over explicit query-to-positive mappings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def evaluate(rankings: Mapping[str, Sequence[str]], positives: Mapping[str, set[str]], cutoffs: tuple[int, ...] = (1, 5, 10)) -> dict[str, float | int]:
    eligible = {query_id: labels for query_id, labels in positives.items() if labels and query_id in rankings}
    if not eligible: raise ValueError("no queries have both rankings and positive labels")
    result: dict[str, float | int] = {"queries": len(eligible)}
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = sum(bool(set(rankings[query_id][:cutoff]) & labels) for query_id, labels in eligible.items()) / len(eligible)
    reciprocal_ranks = []
    for query_id, labels in eligible.items():
        rank = next((position for position, passage_id in enumerate(rankings[query_id], start=1) if passage_id in labels), None)
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result["mrr"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return result
