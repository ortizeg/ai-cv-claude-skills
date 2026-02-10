"""Dependency resolution for skills."""

from __future__ import annotations

from whet.core.skill import Skill


def resolve_dependencies(
    requested: list[str],
    available: dict[str, Skill],
) -> list[str]:
    """Resolve skill dependencies, returning an ordered install list.

    Given a list of requested skill names, adds any required dependencies
    and returns a topologically-sorted list.
    """
    resolved: list[str] = []
    seen: set[str] = set()

    def _visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)

        skill = available.get(name)
        if skill and skill.metadata:
            for dep in skill.metadata.dependencies.requires:
                if dep in available:
                    _visit(dep)

        resolved.append(name)

    for name in requested:
        _visit(name)

    return resolved
