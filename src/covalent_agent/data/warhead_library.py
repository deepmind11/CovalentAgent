"""Warhead library: query interface over data/warheads.json."""

from __future__ import annotations

from .loaders import get_residue_properties, get_warhead_classes


class WarheadLibrary:
    """Query interface for the curated warhead class library."""

    def __init__(self) -> None:
        self._warheads = get_warhead_classes()
        self._residue_props = get_residue_properties()

    @property
    def all_warheads(self) -> list[dict]:
        return list(self._warheads)

    def get_warheads_for_residue(self, residue_type: str) -> list[dict]:
        """Return warhead classes that target a given residue type.

        Args:
            residue_type: e.g. "cysteine", "lysine", "serine"
        """
        rt = residue_type.lower()
        return [w for w in self._warheads if rt in w["target_residues"]]

    def get_residue_properties(self, residue_type: str) -> dict | None:
        """Return chemical properties for a residue type."""
        return self._residue_props.get(residue_type.lower())

    def get_warhead_by_name(self, name: str) -> dict | None:
        """Look up a warhead class by its name (case-insensitive)."""
        name_lower = name.lower()
        for w in self._warheads:
            if w["name"].lower() == name_lower:
                return w
        return None

    def score_warhead_for_context(
        self,
        warhead: dict,
        residue_type: str,
        ligandability: float,
    ) -> float:
        """Heuristic score (0-1) for a warhead given target context.

        Considers reactivity match, selectivity, and ligandability.
        """
        reactivity_scores = {"low": 0.3, "moderate": 0.6, "high": 0.9}
        selectivity_scores = {"low": 0.3, "moderate": 0.6, "high": 0.9, "broad": 0.4}

        react = reactivity_scores.get(warhead.get("reactivity", "moderate"), 0.5)
        select = selectivity_scores.get(warhead.get("selectivity", "moderate"), 0.5)

        # Prefer moderate reactivity for drug-like molecules
        reactivity_penalty = abs(react - 0.6) * 0.3

        # Bonus for having approved drug examples
        precedent_bonus = min(len(warhead.get("examples", [])) * 0.1, 0.3)

        score = (
            0.3 * select
            + 0.2 * (1.0 - reactivity_penalty)
            + 0.2 * ligandability
            + 0.15 * precedent_bonus
            + 0.15 * (1.0 if residue_type.lower() in warhead.get("target_residues", []) else 0.0)
        )
        return round(min(max(score, 0.0), 1.0), 3)
