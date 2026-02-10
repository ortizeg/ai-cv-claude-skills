"""Search and filter skills."""

from __future__ import annotations

from whet.core.skill import Skill


def search_skills(
    skills: list[Skill],
    query: str | None = None,
    category: str | None = None,
    tag: str | None = None,
) -> list[Skill]:
    """Search skills by query, category, or tag.

    Query matches against name, description, and tags.
    """
    results = skills

    if category:
        results = [s for s in results if s.category == category]

    if tag:
        results = [s for s in results if tag in s.tags]

    if query:
        q = query.lower()
        results = [
            s
            for s in results
            if q in s.name.lower()
            or q in s.description.lower()
            or any(q in t.lower() for t in s.tags)
        ]

    return results
