# from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def pairwise_combinations(items: Sequence[str]) -> List[Tuple[str, str]]:
    """Return all unique unordered pairs from a sequence."""
    pairs: List[Tuple[str, str]] = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((items[i], items[j]))
    return pairs


def safe_divide(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 on zero denominator."""
    if denominator == 0:
        return 0.0
    return numerator / denominator

