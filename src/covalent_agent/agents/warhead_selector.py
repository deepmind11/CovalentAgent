"""WarheadSelector agent: recommends optimal warhead classes for a target residue.

Takes a WarheadSelectionInput (residue type, ligandability, structural context,
protein name) and returns scored WarheadRecommendation models ranked by
suitability.
"""

from __future__ import annotations

import logging

import anthropic

from covalent_agent.config import settings
from covalent_agent.data.warhead_library import WarheadLibrary
from covalent_agent.schemas import (
    WarheadRecommendation,
    WarheadSelectionInput,
    WarheadSelectionResult,
)

logger = logging.getLogger(__name__)

_MAX_RECOMMENDATIONS = 5
_MIN_RECOMMENDATIONS = 3


class WarheadSelectorAgent:
    """Recommend warhead classes for covalent drug design.

    Workflow:
        1. Query WarheadLibrary for warheads compatible with the target residue.
        2. Score each warhead using the library's heuristic scorer.
        3. Ask Claude to generate a detailed rationale for the top picks.
        4. Return sorted WarheadRecommendation models.
    """

    def __init__(self) -> None:
        self._library = WarheadLibrary()
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def run(self, input: WarheadSelectionInput) -> WarheadSelectionResult:
        """Execute the warhead selection pipeline."""
        compatible = self._library.get_warheads_for_residue(input.residue_type)

        if not compatible:
            logger.warning(
                "No warheads found for residue type '%s'. "
                "Returning empty recommendations.",
                input.residue_type,
            )
            return WarheadSelectionResult(
                target_residue=input.residue_type,
                recommendations=[],
            )

        scored = _score_and_sort(
            warheads=compatible,
            residue_type=input.residue_type,
            ligandability=input.ligandability_score,
            library=self._library,
        )

        top_warheads = scored[:_MAX_RECOMMENDATIONS]

        rationales = await self._generate_rationales(
            warheads=top_warheads,
            input=input,
        )

        recommendations = [
            _build_recommendation(warhead, score, rationales.get(warhead["name"], ""))
            for warhead, score in top_warheads
        ]

        return WarheadSelectionResult(
            target_residue=input.residue_type,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Claude rationale generation
    # ------------------------------------------------------------------

    async def _generate_rationales(
        self,
        warheads: list[tuple[dict, float]],
        input: WarheadSelectionInput,
    ) -> dict[str, str]:
        """Use Claude to explain why each warhead suits this target context.

        Returns a mapping of warhead name to rationale string.
        """
        warhead_summaries = "\n".join(
            f"- {w['name']} (score: {s:.3f}, reactivity: {w['reactivity']}, "
            f"selectivity: {w['selectivity']}, mechanism: {w['mechanism']})"
            for w, s in warheads
        )

        prompt = (
            f"You are an expert medicinal chemist specializing in covalent drug design.\n\n"
            f"Target protein: {input.protein_name}\n"
            f"Target residue type: {input.residue_type}\n"
            f"Ligandability score: {input.ligandability_score:.2f}\n"
            f"Structural context: {input.structural_context}\n\n"
            f"The following warhead classes scored highest for this target:\n"
            f"{warhead_summaries}\n\n"
            f"For EACH warhead, provide a concise rationale (2-3 sentences) explaining:\n"
            f"1. Why this warhead is suitable for {input.residue_type} in {input.protein_name}\n"
            f"2. Key advantages and any concerns for this specific context\n"
            f"3. Relevant precedent from approved drugs or clinical candidates\n\n"
            f"Format your response as:\n"
            f"WARHEAD_NAME: rationale text\n"
            f"(one entry per warhead, separated by blank lines)"
        )

        try:
            response = await self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_rationales(
                response.content[0].text,
                [w["name"] for w, _ in warheads],
            )
        except Exception:
            logger.exception("Claude rationale generation failed; using fallback rationales.")
            return {
                w["name"]: (
                    f"{w['name']} targets {input.residue_type} via {w['mechanism']}. "
                    f"Reactivity: {w['reactivity']}, selectivity: {w['selectivity']}."
                )
                for w, _ in warheads
            }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _score_and_sort(
    warheads: list[dict],
    residue_type: str,
    ligandability: float,
    library: WarheadLibrary,
) -> list[tuple[dict, float]]:
    """Score each warhead and return (warhead, score) pairs sorted descending."""
    scored = [
        (w, library.score_warhead_for_context(w, residue_type, ligandability))
        for w in warheads
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def _build_recommendation(
    warhead: dict,
    score: float,
    rationale: str,
) -> WarheadRecommendation:
    """Convert a raw warhead dict + score into a WarheadRecommendation model."""
    return WarheadRecommendation(
        warhead_class=warhead["name"],
        smarts=warhead["smarts"],
        reactivity=warhead["reactivity"],
        selectivity=warhead["selectivity"],
        score=score,
        rationale=rationale,
        examples=warhead.get("examples", []),
        mechanism=warhead["mechanism"],
    )


def _parse_rationales(text: str, warhead_names: list[str]) -> dict[str, str]:
    """Parse Claude's response into a name-to-rationale mapping.

    Expected format:
        WARHEAD_NAME: rationale text
        (blank line separator)

    Falls back gracefully if parsing fails for a given warhead.
    """
    rationales: dict[str, str] = {}
    name_lower_map = {name.lower(): name for name in warhead_names}

    current_name: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        # Check if this line starts a new warhead entry
        matched_name = _match_warhead_line(stripped, name_lower_map)
        if matched_name is not None:
            # Save previous entry
            if current_name is not None:
                rationales[current_name] = " ".join(current_lines).strip()
            current_name = matched_name
            # Extract text after the colon on the same line
            colon_idx = stripped.find(":")
            remainder = stripped[colon_idx + 1 :].strip() if colon_idx != -1 else ""
            current_lines = [remainder] if remainder else []
        elif stripped and current_name is not None:
            current_lines.append(stripped)

    # Save final entry
    if current_name is not None:
        rationales[current_name] = " ".join(current_lines).strip()

    return rationales


def _match_warhead_line(
    line: str,
    name_lower_map: dict[str, str],
) -> str | None:
    """Check if a line starts with a known warhead name followed by a colon.

    Returns the canonical warhead name if matched, None otherwise.
    """
    line_lower = line.lower()
    for lower_name, canonical in name_lower_map.items():
        # Match patterns like "Acrylamide:" or "**Acrylamide**:" (markdown bold)
        clean = line_lower.replace("*", "").replace("#", "").strip()
        if clean.startswith(lower_name) and ":" in clean:
            return canonical
    return None
