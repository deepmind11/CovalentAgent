"""Data loaders for warhead library and reactive residue database."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


@lru_cache(maxsize=1)
def load_warheads() -> dict:
    """Load the warhead class definitions from data/warheads.json."""
    with open(DATA_DIR / "warheads.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_reactive_residues() -> dict:
    """Load known druggable residues from data/reactive_residues.json."""
    with open(DATA_DIR / "reactive_residues.json") as f:
        return json.load(f)


def get_warhead_classes() -> list[dict]:
    """Return the list of warhead class definitions."""
    return load_warheads()["warhead_classes"]


def get_residue_properties() -> dict:
    """Return residue property definitions keyed by residue name."""
    return load_warheads()["residue_properties"]


def get_known_druggable_cysteines() -> list[dict]:
    """Return the list of known druggable cysteine entries."""
    return load_reactive_residues()["known_druggable_cysteines"]


def lookup_protein(protein_name: str) -> dict | None:
    """Look up a protein by name in the reactive residues database.

    Returns the first matching entry, or None if not found.
    """
    name_upper = protein_name.upper()
    for entry in get_known_druggable_cysteines():
        if entry["protein"].upper() == name_upper:
            return entry
    return None


def lookup_residue(protein_name: str, residue: str) -> dict | None:
    """Look up a specific protein + residue combination.

    Args:
        protein_name: e.g. "KRAS"
        residue: e.g. "C12"

    Returns the matching entry or None.
    """
    name_upper = protein_name.upper()
    residue_upper = residue.upper()
    for entry in get_known_druggable_cysteines():
        if (
            entry["protein"].upper() == name_upper
            and entry["residue"].upper() == residue_upper
        ):
            return entry
    return None
