"""Safe, well-formed mutation of the evaluator rubric XML.

Extracted from ``evolve.py`` so both the single-lineage propose path and the
Pareto-frontier ``gepa_evolve`` loop (Phase 2) share one correct implementation.

Correctness guarantees (Phase 0 of the RQGM fidelity rebuild):

* The version bump targets ONLY the ``<evaluator ... version="...">`` opening
  tag — never the XML prolog ``<?xml version="1.0"?>`` (the old
  ``re.sub(r'version="[^"]*"', ..., count=1)`` corrupted the prolog).
* New ``<criterion>`` entries are de-duplicated by ``id`` against the parent
  rubric, so a criterion cannot be appended twice across epochs.
* The mutated text is parsed with ``xml.etree`` and rejected if not well-formed.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

# Matches the version attribute *inside the <evaluator ...> opening tag only*.
_EVALUATOR_VERSION_RE = re.compile(r'(<evaluator\b[^>]*?\bversion=")[^"]*(")', re.DOTALL)
# Matches the <evaluator ...> opening tag (to inject a version attr if missing).
_EVALUATOR_OPEN_RE = re.compile(r"(<evaluator\b[^>]*?)(\s*>)", re.DOTALL)
_CRITERION_ID_RE = re.compile(r'<criterion\b[^>]*?\bid="([^"]+)"')
# The <rubric>...</rubric> block (the only structure mutation touches besides
# the evaluator version attribute).
_RUBRIC_BLOCK_RE = re.compile(r"<rubric\b[^>]*>.*?</rubric>", re.DOTALL)


class RubricValidationError(ValueError):
    """Raised when a mutated rubric is not well-formed XML."""


def existing_criterion_ids(rubric_text: str) -> set[str]:
    """Return the set of ``<criterion id="...">`` ids already in the rubric."""
    return set(_CRITERION_ID_RE.findall(rubric_text))


def set_evaluator_version(rubric_text: str, version: str) -> str:
    """Set ``version="..."`` on the ``<evaluator>`` tag ONLY (never the prolog)."""
    new_text, n = _EVALUATOR_VERSION_RE.subn(rf'\g<1>{version}\g<2>', rubric_text)
    if n:
        return new_text
    # No version attribute on <evaluator>; inject one.
    new_text, n = _EVALUATOR_OPEN_RE.subn(rf'\g<1> version="{version}"\g<2>', rubric_text, count=1)
    return new_text if n else rubric_text


def validate_rubric_xml(rubric_text: str) -> bool:
    """Validate the structural integrity of a (mutated) evaluator rubric.

    The seed rubric intentionally embeds pseudo-XML *placeholders* in free text
    (e.g. ``<float 0..1>`` and ``<id>`` inside ``<output_contract>``), so the
    document is not well-formed as a whole by design. We therefore validate the
    parts that mutation actually touches and that the old buggy regex used to
    corrupt:

    1. The XML prolog, if present, must be the canonical ``<?xml version="1.0"``
       declaration (the old regex rewrote it to the rubric version string).
    2. There must be exactly one balanced ``<evaluator> ... </evaluator>`` wrapper.
    3. The ``<rubric> ... </rubric>`` block (which holds the appended criteria)
       must be well-formed XML on its own.
    """
    stripped = rubric_text.lstrip()
    if stripped.startswith("<?xml"):
        prolog = stripped.split("?>", 1)[0]
        if 'version="1.0"' not in prolog:
            return False
    if rubric_text.count("</evaluator>") != 1 or not _EVALUATOR_OPEN_RE.search(rubric_text):
        return False
    block = _RUBRIC_BLOCK_RE.search(rubric_text)
    if block is None:
        return False
    try:
        ET.fromstring(block.group(0))
    except ET.ParseError:
        return False
    return True


def render_criteria(new_criteria: list[dict[str, Any]], epoch: int) -> str:
    """Render ``new_criteria`` dicts to ``<criterion>`` XML fragments."""
    out = ""
    for c in new_criteria:
        cid = str(c.get("id", "evolved_criterion"))
        text = str(c.get("text", "")).strip()
        weight = c.get("weight", 0.10)
        out += (
            f'    <criterion id="{cid}" weight="{weight}" origin="gepa" epoch_added="{epoch}">\n'
            f"      {text}\n"
            f"    </criterion>\n"
        )
    return out


def mutate_rubric_text(
    champion_text: str,
    new_criteria: list[dict[str, Any]],
    version: str,
    epoch: int,
    *,
    validate: bool = True,
) -> str:
    """Append de-duplicated ``new_criteria`` and bump the evaluator version.

    Parameters
    ----------
    champion_text : str
        The parent rubric XML.
    new_criteria : list[dict]
        Candidate criteria (``{"id", "text", "weight"?}``). Any whose ``id`` is
        already present in ``champion_text`` is skipped (dedup).
    version : str
        New value for the ``<evaluator version="...">`` attribute.
    epoch : int
        Stamped on each appended criterion as ``epoch_added``.
    validate : bool
        If True (default) the result is parsed and :class:`RubricValidationError`
        is raised when it is not well-formed XML.

    Returns
    -------
    str
        The mutated, well-formed rubric XML.
    """
    present = existing_criterion_ids(champion_text)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set(present)
    for c in new_criteria:
        cid = str(c.get("id", "evolved_criterion"))
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(c)

    crit_xml = render_criteria(deduped, epoch)
    out = champion_text
    if "</rubric>" in out:
        out = out.replace("</rubric>", crit_xml + "  </rubric>", 1)
    else:
        out = out + "\n" + crit_xml
    out = set_evaluator_version(out, version)

    if validate and not validate_rubric_xml(out):
        raise RubricValidationError(
            f"mutated rubric for version {version!r} is not well-formed XML"
        )
    return out
