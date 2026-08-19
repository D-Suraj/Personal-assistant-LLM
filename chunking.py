import re
from typing import Protocol

import numpy as np


class SupportsEncode(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


_UNIT_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?=[ \t]+)|(?<=\n)(?=[ \t]*\S)"
)
_COSINE_DISTANCE_EPSILON = 1e-6


def _split_oversized_unit(text: str, limit: int) -> list[str]:
    """Split a large unit at a nearby line/word boundary when possible."""
    pieces: list[str] = []
    remaining = text
    minimum_preferred_break = max(1, limit // 2)

    while len(remaining) > limit:
        newline_at = remaining.rfind("\n", 0, limit)
        whitespace_at = max(
            remaining.rfind(" ", 0, limit),
            remaining.rfind("\t", 0, limit),
        )
        preferred_at = max(newline_at, whitespace_at)
        break_at = (
            preferred_at + 1 if preferred_at >= minimum_preferred_break else limit
        )
        pieces.append(remaining[:break_at])
        remaining = remaining[break_at:]

    if remaining:
        pieces.append(remaining)
    return pieces


def _candidate_units(text: str, limit: int) -> list[str]:
    units: list[str] = []
    pending_whitespace = ""
    for candidate in _UNIT_BOUNDARY_RE.split(text):
        if not candidate.strip():
            pending_whitespace += candidate
            continue
        candidate = pending_whitespace + candidate
        pending_whitespace = ""
        units.extend(_split_oversized_unit(candidate, limit))
    if pending_whitespace and units:
        previous = units.pop()
        units.extend(_split_oversized_unit(previous + pending_whitespace, limit))
    return units


def _semantic_distances(
    units: list[str],
    embedder: SupportsEncode,
    context_window: int,
) -> np.ndarray:
    contexts = [
        "".join(
            units[
                max(0, index - context_window) : min(
                    len(units), index + context_window + 1
                )
            ]
        ).strip()
        for index in range(len(units))
    ]
    try:
        vectors = np.asarray(embedder.encode(contexts), dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The embedding model returned invalid semantic vectors."
        ) from exc

    if vectors.ndim != 2 or vectors.shape[0] != len(units) or vectors.shape[1] == 0:
        raise ValueError(
            "The embedding model must return one non-empty vector per semantic unit."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("The embedding model returned non-finite semantic vectors.")

    left = vectors[:-1]
    right = vectors[1:]
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    similarities = np.ones(len(units) - 1, dtype=np.float32)
    np.divide(
        np.sum(left * right, axis=1),
        denominators,
        out=similarities,
        where=denominators > 0,
    )
    return 1.0 - np.clip(similarities, -1.0, 1.0)


def _overlap_from_previous(units: list[str], budget: int) -> str:
    if budget <= 0:
        return ""

    selected: list[str] = []
    for unit in reversed(units):
        candidate = "".join([unit, *selected])
        if len(candidate) > budget:
            break
        selected.insert(0, unit)

    overlap = "".join(selected)
    if overlap:
        return overlap

    previous = "".join(units)
    suffix = previous[-budget:]
    boundary = re.search(r"\s+", suffix)
    if boundary and boundary.end() <= max(1, budget // 3):
        suffix = suffix[boundary.end() :]
    return suffix if suffix.strip() else ""


def semantic_chunk_text(
    text: str,
    embedder: SupportsEncode,
    *,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
    breakpoint_percentile: float,
    min_breakpoint_distance: float,
    context_window: int,
) -> list[str]:
    """Create bounded chunks using embedding distance between neighboring units."""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars - 1.")
    if min_chars <= 0:
        raise ValueError("min_chars must be greater than zero.")
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError("breakpoint_percentile must be between 0 and 100.")
    if not 0 <= min_breakpoint_distance <= 2:
        raise ValueError("min_breakpoint_distance must be between 0 and 2.")
    if context_window < 0:
        raise ValueError("context_window cannot be negative.")

    clean = text.strip()
    if not clean:
        return []

    core_limit = max_chars - overlap_chars
    min_chars = min(min_chars, core_limit)
    if len(clean) <= max_chars:
        return [clean]

    preferred_unit_limit = overlap_chars or core_limit // 3
    unit_limit = max(1, min(core_limit, max(80, preferred_unit_limit)))
    units = _candidate_units(clean, unit_limit)
    if len(units) == 1:
        return units

    distances = _semantic_distances(units, embedder, context_window)
    distance_threshold = float(np.percentile(distances, breakpoint_percentile))
    distance_threshold = max(distance_threshold, min_breakpoint_distance)

    core_chunks: list[list[str]] = []
    break_reasons: list[str] = []
    current: list[str] = []
    current_length = 0
    for index, unit in enumerate(units):
        distance = float(distances[index - 1]) if current else 0.0
        semantic_break = (
            bool(current)
            and current_length >= min_chars
            and distance > _COSINE_DISTANCE_EPSILON
            and distance >= distance_threshold
        )
        current_limit = max_chars if not core_chunks else core_limit
        size_break = bool(current) and current_length + len(unit) > current_limit
        if semantic_break or size_break:
            core_chunks.append(current)
            break_reasons.append("semantic" if semantic_break else "size")
            current = []
            current_length = 0
        current.append(unit)
        current_length += len(unit)

    if current:
        core_chunks.append(current)

    if (
        len(core_chunks) > 1
        and break_reasons[-1] == "size"
        and len("".join(core_chunks[-1])) < min_chars
    ):
        previous = core_chunks[-2]
        final = core_chunks[-1]
        previous_limit = max_chars if len(core_chunks) == 2 else core_limit
        if len("".join(previous + final)) <= previous_limit:
            previous.extend(final)
            core_chunks.pop()
            break_reasons.pop()
        else:
            previous_length = len("".join(previous))
            final_length = len("".join(final))
            while previous and final_length < min_chars:
                unit = previous[-1]
                if final_length + len(unit) > core_limit:
                    break
                if previous_length - len(unit) < min_chars:
                    break
                final.insert(0, previous.pop())
                previous_length -= len(unit)
                final_length += len(unit)

    chunks = ["".join(core_chunks[0])]
    for index in range(1, len(core_chunks)):
        core_text = "".join(core_chunks[index])
        separator_size = 2
        overlap_budget = min(
            overlap_chars,
            max(0, max_chars - len(core_text) - separator_size),
        )
        overlap = _overlap_from_previous(core_chunks[index - 1], overlap_budget)
        chunk = f"{overlap}\n\n{core_text}" if overlap else core_text
        chunks.append(chunk)

    return chunks
