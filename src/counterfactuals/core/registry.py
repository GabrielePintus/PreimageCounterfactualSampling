"""Type-safe registry utilities for pluggable components."""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Simple string-to-class registry with runtime validation."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        """Register a factory under a stable public name."""
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = factory

    def get(self, name: str) -> Callable[..., T]:
        """Return a registered factory or raise with a helpful error."""
        if name not in self._items:
            options = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: [{options}]")
        return self._items[name]

    def create(self, name: str, **kwargs: Any) -> T:
        """Instantiate a registered factory with keyword arguments."""
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        """List all registered public names."""
        return sorted(self._items.keys())
