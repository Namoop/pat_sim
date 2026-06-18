"""Scan amplitude envelope profile names and parsing."""

from __future__ import annotations

ENVELOPE_SMOOTH = 0
ENVELOPE_LINEAR = 1
ENVELOPE_COSINE = 2

SCAN_ENVELOPE_PROFILES: tuple[str, ...] = ("smooth", "linear", "cosine")

_PROFILE_IDS: dict[str, int] = {
    "smooth": ENVELOPE_SMOOTH,
    "linear": ENVELOPE_LINEAR,
    "cosine": ENVELOPE_COSINE,
}


def parse_scan_envelope_profile(value: str) -> int:
    """Return numeric profile id for *value* (case-insensitive)."""
    key = value.strip().lower()
    if key not in _PROFILE_IDS:
        valid = ", ".join(SCAN_ENVELOPE_PROFILES)
        raise ValueError(
            f"scan_envelope_profile must be one of {{{valid}}}; got {value!r}"
        )
    return _PROFILE_IDS[key]
