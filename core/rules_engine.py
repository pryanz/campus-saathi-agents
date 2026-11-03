"""Versioned, data-driven rules engine.

All eligibility logic lives in ``data/rules/rule_pack_<version>.json``. This module
only loads, validates and evaluates those packs against a ``StudentProfile``.
"""
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from typing import List, Optional

from core.profile import StudentProfile

DEFAULT_RULE_PACK_VERSION = "v1"
DEFAULT_RULES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rules"
)


class RulePackError(ValueError):
    """Raised when a rule pack is missing or malformed."""


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    name: str
    station: str
    original_copies: int
    photocopies: int
    cutoff_date: Optional[str] = None
    issued_on_or_after: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "station": self.station,
            "copies": {"original": self.original_copies, "photocopy": self.photocopies},
            "cutoff_date": self.cutoff_date,
            "issued_on_or_after": self.issued_on_or_after,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Checklist:
    profile: StudentProfile
    rule_pack_version: str
    documents: List[ChecklistItem] = field(default_factory=list)
    required_stations: List[str] = field(default_factory=list)

    def documents_by_station(self) -> dict:
        grouped = {station: [] for station in self.required_stations}
        for doc in self.documents:
            grouped.setdefault(doc.station, []).append(doc)
        return grouped

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "rule_pack_version": self.rule_pack_version,
            "required_stations": list(self.required_stations),
            "documents": [d.to_dict() for d in self.documents],
        }


# ---------------------------------------------------------------- loading

def _pack_path(version: str, rules_dir: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", version or ""):
        raise RulePackError(f"Invalid rule pack version: {version!r}")
    return os.path.join(rules_dir, f"rule_pack_{version}.json")


@lru_cache(maxsize=None)
def _load_cached(version: str, rules_dir: str) -> str:
    path = _pack_path(version, rules_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        raise RulePackError(f"Rule pack '{version}' not found at {path}")
    pack = json.loads(raw)
    validate_rule_pack(pack)
    if pack["version"] != version:
        raise RulePackError(
            f"Rule pack file for '{version}' declares version '{pack['version']}'"
        )
    return raw


def load_rule_pack(version: str = DEFAULT_RULE_PACK_VERSION, rules_dir: Optional[str] = None) -> dict:
    """Load and validate a rule pack. Returns a fresh copy, safe to mutate."""
    return json.loads(_load_cached(version, rules_dir or DEFAULT_RULES_DIR))


# ------------------------------------------------------------- validation

def validate_rule_pack(pack: dict) -> None:
    """Check a rule pack against data/rules/rule_pack.schema.json's constraints.

    Hand-written (no jsonschema dependency) and stricter than the schema where
    cross-references matter: station ids, unique ids, criteria field names.
    """
    if not isinstance(pack, dict):
        raise RulePackError("Rule pack must be a JSON object")
    for key in ("version", "stations", "documents"):
        if key not in pack:
            raise RulePackError(f"Rule pack is missing required key '{key}'")
    extra = set(pack) - {"version", "academic_year", "notes", "stations", "documents"}
    if extra:
        raise RulePackError(f"Unknown top-level keys: {sorted(extra)}")
    if not isinstance(pack["version"], str) or not pack["version"]:
        raise RulePackError("'version' must be a non-empty string")

    station_ids = []
    for s in pack["stations"]:
        if not isinstance(s, dict) or not isinstance(s.get("id"), str):
            raise RulePackError(f"Invalid station entry: {s!r}")
        if set(s) - {"id", "when"}:
            raise RulePackError(f"Unknown keys in station '{s['id']}'")
        _validate_criteria(s.get("when"), f"station '{s['id']}'")
        station_ids.append(s["id"])
    _require_unique(station_ids, "station")

    doc_ids = []
    allowed = {"id", "name", "copies", "station", "when", "cutoff_date",
               "issued_on_or_after", "notes"}
    for d in pack["documents"]:
        if not isinstance(d, dict):
            raise RulePackError(f"Invalid document entry: {d!r}")
        doc_id = d.get("id")
        for key in ("id", "name", "copies", "station"):
            if key not in d:
                raise RulePackError(f"Document '{doc_id}' is missing '{key}'")
        if set(d) - allowed:
            raise RulePackError(f"Unknown keys in document '{doc_id}': {sorted(set(d) - allowed)}")
        copies = d["copies"]
        if (not isinstance(copies, dict) or set(copies) != {"original", "photocopy"}
                or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                       for v in copies.values())):
            raise RulePackError(f"Document '{doc_id}' needs copies {{original, photocopy}} as non-negative ints")
        if d["station"] not in station_ids:
            raise RulePackError(f"Document '{doc_id}' references unknown station '{d['station']}'")
        for date_key in ("cutoff_date", "issued_on_or_after"):
            if date_key in d:
                try:
                    date.fromisoformat(d[date_key])
                except (TypeError, ValueError):
                    raise RulePackError(f"Document '{doc_id}' has invalid {date_key}: {d[date_key]!r}")
        _validate_criteria(d.get("when"), f"document '{doc_id}'")
        doc_ids.append(doc_id)
    _require_unique(doc_ids, "document")


def _require_unique(ids: list, kind: str) -> None:
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise RulePackError(f"Duplicate {kind} ids: {sorted(dupes)}")


def _validate_criteria(criteria, where: str) -> None:
    if criteria is None:
        return
    if not isinstance(criteria, dict):
        raise RulePackError(f"'when' in {where} must be an object")
    valid_fields = StudentProfile.field_names()
    for key, cond in criteria.items():
        if key not in valid_fields:
            raise RulePackError(f"'when' in {where} uses unknown profile field '{key}'")
        values = cond["not"] if isinstance(cond, dict) else cond
        if isinstance(cond, dict) and set(cond) != {"not"}:
            raise RulePackError(f"'when' in {where}: only 'not' is supported, got {sorted(cond)}")
        if isinstance(values, (str, bool)):
            continue
        if not isinstance(values, list) or not values:
            raise RulePackError(f"'when.{key}' in {where} must be a non-empty list")


# ------------------------------------------------------------- evaluation

def _norm(value):
    return value.strip().casefold() if isinstance(value, str) else value


def matches(criteria: Optional[dict], profile: StudentProfile) -> bool:
    """True if the profile satisfies every criterion (AND across fields, OR within a list)."""
    if not criteria:
        return True
    for key, cond in criteria.items():
        actual = _norm(getattr(profile, key))
        negate = isinstance(cond, dict)
        values = cond["not"] if negate else cond
        if not isinstance(values, list):
            values = [values]
        hit = actual in [_norm(v) for v in values]
        if hit == negate:
            return False
    return True


def build_checklist(profile: StudentProfile,
                    rule_pack_version: str = DEFAULT_RULE_PACK_VERSION,
                    rules_dir: Optional[str] = None) -> Checklist:
    """Personalised document checklist and required stations for a student profile."""
    pack = load_rule_pack(rule_pack_version, rules_dir)

    documents = [
        ChecklistItem(
            id=d["id"],
            name=d["name"],
            station=d["station"],
            original_copies=d["copies"]["original"],
            photocopies=d["copies"]["photocopy"],
            cutoff_date=d.get("cutoff_date"),
            issued_on_or_after=d.get("issued_on_or_after"),
            notes=d.get("notes", ""),
        )
        for d in pack["documents"]
        if matches(d.get("when"), profile)
    ]

    # A station is required when its own criteria match, or when a required
    # document is verified there. Order follows the rule pack.
    needed = {d.station for d in documents}
    stations = [
        s["id"] for s in pack["stations"]
        if matches(s.get("when"), profile) or s["id"] in needed
    ]

    return Checklist(
        profile=profile,
        rule_pack_version=pack["version"],
        documents=documents,
        required_stations=stations,
    )
