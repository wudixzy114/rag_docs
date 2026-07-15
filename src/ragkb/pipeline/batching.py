"""Deterministic size-aware batching shared by LLM pipeline stages."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def pack_by_size(items: Iterable[T], size_of: Callable[[T], int], *,
                 max_items: int, max_chars: int) -> list[list[T]]:
    """Pack in input order under both count and character budgets."""
    batches: list[list[T]] = []
    current: list[T] = []
    current_size = 0
    for item in items:
        item_size = max(1, size_of(item))
        if current and (len(current) >= max_items or current_size + item_size > max_chars):
            batches.append(current)
            current, current_size = [], 0
        current.append(item)
        current_size += item_size
    if current:
        batches.append(current)
    return batches
