"""Role 4 (HUANG, Ze): small analysis helpers; full bootstrap analysis is future work."""

from collections import Counter
from collections.abc import Iterable

from .contracts import RunResult


def paired_attainment_counts(pairs: Iterable[tuple[RunResult, RunResult]]) -> dict[str, int]:
    """Keep all four outcomes; an attainment is historical, not final retention."""
    counts = Counter({"both": 0, "diagonal_only": 0, "rotated_only": 0, "neither": 0})
    for diagonal, rotated in pairs:
        d = diagonal.first_attainment is not None
        r = rotated.first_attainment is not None
        key = "both" if d and r else "diagonal_only" if d else "rotated_only" if r else "neither"
        counts[key] += 1
    return dict(counts)
