"""Industrial entity extraction.

Method: regular expressions for structured identifiers (asset IDs, dates, regulatory references),
a curated domain lexicon with regex variants for components / faults / maintenance actions, simple
negation handling ("no corrosion was found"), and sentence-level rules for safety precautions.

Every row records whether the value was EXTRACTED verbatim from text or INFERRED by a rule
(e.g. equipment type derived from the ID prefix). This is NOT a trained NER model; its limits are
documented in the README and measured on the Evaluation page.
"""
from __future__ import annotations

import re

import pandas as pd

from .text_preprocessor import split_sentences

ASSET_RE = re.compile(r"\b[PMCBVT]-\d{3}\b")
REG_RE = re.compile(r"\bIKI-(?:SAF|REG|PRS|ENV)-\d{2,3}\b")
DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
MONTH_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b")
WO_RE = re.compile(r"\bWO-\d{4}-\d{3,4}\b")
NAME_RE = re.compile(r"\b((?:(?:feedwater|cooling water|booster|circulation|instrument air|air|steam|standby) )?(?:pump|compressor|boiler|motor))\s+([PMCBVT]-\d{3})\b", re.I)
DEPT_RE = re.compile(r"(?:Department|Dept)\s*:\s*([A-Za-z &]+?)\s*(?:\n|$|\.)", re.I)
PERSON_RE = re.compile(r"(?:Reported by|Reviewed by|Inspector|Prepared by)\s*:?\s*([A-Za-z ]+?)(?:\n|$|\.)", re.I)
DEPT_MENTION_RE = re.compile(r"\b(Inspection Department|Mechanical Maintenance|Electrical Maintenance|Operations)\b")

TYPE_BY_PREFIX = {"P": "Pump", "M": "Motor", "C": "Compressor", "B": "Boiler", "V": "Valve", "T": "Tank"}

# canonical name -> regex alternatives (matched case-insensitively, longest match wins per type)
COMPONENTS = {
    "mechanical seal": [r"mechanical seals?"], "seal face": [r"seal faces?"], "bearing": [r"bearings?"],
    "impeller": [r"impellers?"], "shaft sleeve": [r"shaft sleeves?"], "shaft": [r"shafts?"],
    "coupling guard": [r"coupling guards?"], "coupling": [r"couplings?"], "suction strainer": [r"suction strainers?"],
    "oil cooler": [r"oil coolers?"], "oil filter": [r"oil filters?"], "air filter": [r"air filters?"],
    "cooling fan": [r"cooling fans?"], "rotor": [r"rotors?"], "winding": [r"windings?"],
    "terminal box": [r"terminal box(?:es)?"], "pressure relief valve": [r"(?:pressure )?relief valves?"],
    "thermocouple": [r"thermocouples?"], "burner": [r"burners?"], "flame scanner": [r"flame scanners?"],
    "fuel valve": [r"fuel (?:supply )?valves?"], "furnace": [r"furnaces?"], "tube": [r"tubes?"],
    "economiser": [r"economi[sz]ers?"], "header": [r"headers?"], "shell": [r"\bshell\b"],
    "gasket": [r"gaskets?"], "flange": [r"flanges?"], "gland": [r"glands?"],
}
FAULTS = {
    "bearing failure": [r"bearing failures?", r"failed bearings?"], "seal leakage": [r"seal (?:leakage|leaks?)"],
    "leakage": [r"leakage", r"leaks?", r"leaking"], "overheating": [r"overheat(?:ing|ed|s)?", r"over-temperature"],
    "cavitation": [r"cavitation"], "vibration": [r"vibrations?"], "misalignment": [r"misalign\w*"],
    "imbalance": [r"imbalance", r"unbalance"], "noise": [r"noise", r"noisy"], "corrosion": [r"corrosion", r"corroded"],
    "fouling": [r"foul(?:ing|ed)"], "scaling": [r"scale deposits?", r"scaling"], "drift": [r"drift"],
    "trip": [r"\btripped\b", r"\btrips?\b"], "flame failure": [r"flame failure"],
}
ACTIONS = {
    "replace": [r"replac\w*"], "lubricate": [r"lubricat\w*", r"(?:re-)?greased"], "clean": [r"\bclean(?:ed|ing|s)?\b"],
    "inspect": [r"\b(?:re-)?inspect(?:ed|ing|s)?\b"], "realign": [r"realign\w*"], "calibrate": [r"calibrat\w*"],
    "tighten": [r"tighten\w*"], "flush": [r"flush\w*"], "test": [r"\btest(?:ed|ing|s)?\b"], "repair": [r"repair\w*"],
    "drain": [r"drain\w*"], "purge": [r"purg\w*"], "dry": [r"\bdry(?:ing)?\b", r"\bdried\b"], "adjust": [r"adjust\w*"],
    "overhaul": [r"overhaul\w*"], "balance": [r"\bbalanc(?:e|ed|ing)\b"],
}
SAFETY_RE = re.compile(r"lock ?out|lockout|tagout|\bisolate\b|\bwear\b|gloves|eye protection|depressuri[sz]e|work permit|"
                       r"must not|never (?:bypass|remove)|zero voltage|vent all|allow .{0,30}cool", re.I)
NEGATION_RE = re.compile(r"\b(?:no|not|without|nor)\b(?:\W+\w+){0,3}\W*$", re.I)


def _compile(lex: dict[str, list[str]]) -> list[tuple[str, re.Pattern]]:
    return [(canon, re.compile(pat if pat.startswith("\\b") else r"\b" + pat + r"\b", re.I))
            for canon, pats in lex.items() for pat in pats]


_COMP, _FAULT, _ACT = _compile(COMPONENTS), _compile(FAULTS), _compile(ACTIONS)


def _lexicon_matches(sentence: str, patterns, check_negation: bool = False) -> list[tuple[str, str]]:
    """Return (canonical, surface) matches, dropping spans nested in a longer match of the same type."""
    spans = []
    for canon, rx in patterns:
        for m in rx.finditer(sentence):
            if check_negation and NEGATION_RE.search(sentence[:m.start()]):
                continue
            spans.append((m.start(), m.end(), canon, m.group(0)))
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    kept: list[tuple[int, int, str, str]] = []
    for s in spans:
        if not any(s[0] < k[1] and k[0] < s[1] for k in kept):
            kept.append(s)
    return [(k[2], k[3]) for k in sorted(kept)]


def extract_from_sentence(sentence: str) -> list[dict]:
    """All entities of one sentence as dicts with entity_type, value, kind."""
    out: list[dict] = []
    add = lambda t, v, kind="extracted": out.append({"entity_type": t, "value": v, "kind": kind})
    for m in ASSET_RE.finditer(sentence):
        add("asset_id", m.group(0))
        add("equipment_type", TYPE_BY_PREFIX[m.group(0)[0]], "inferred")  # inferred from the ID prefix
    for m in NAME_RE.finditer(sentence):
        add("equipment_name", m.group(1).strip().lower())
    for canon, _ in _lexicon_matches(sentence, _COMP):
        add("component", canon)
    for canon, _ in _lexicon_matches(sentence, _FAULT, check_negation=True):
        add("fault", canon)
    for canon, _ in _lexicon_matches(sentence, _ACT):
        add("maintenance_action", canon)
    for m in REG_RE.finditer(sentence):
        add("regulatory_reference", m.group(0))
    for rx in (DATE_RE, MONTH_RE):
        for m in rx.finditer(sentence):
            is_insp = bool(re.search(r"inspect|analysis|test|due", sentence, re.I))
            add("inspection_date" if is_insp else "date", m.group(0))
    for m in WO_RE.finditer(sentence):
        add("work_order", m.group(0))
    if SAFETY_RE.search(sentence) and len(sentence) < 400:
        add("safety_precaution", sentence.strip())
    for m in DEPT_MENTION_RE.finditer(sentence):
        add("department", m.group(1))
    return out


def extract_entities(chunks: pd.DataFrame) -> pd.DataFrame:
    """Run extraction over all chunks; one row per (entity, evidence sentence)."""
    cols = ["entity_type", "value", "kind", "document", "doc_id", "chunk_id", "location", "evidence"]
    if chunks is None or chunks.empty:
        return pd.DataFrame(columns=cols)
    rows, seen = [], set()
    for ch in chunks.itertuples(index=False):
        text = str(ch.text)
        header_hits = [("department", m.group(1).strip(), text) for m in DEPT_RE.finditer(text)]
        header_hits += [("personnel", m.group(1).strip(), text) for m in PERSON_RE.finditer(text)]
        for et, val, _ in header_hits:
            key = (ch.doc_id, et, val.lower(), "header")
            if key not in seen:
                seen.add(key)
                rows.append(dict(entity_type=et, value=val, kind="extracted", document=ch.document, doc_id=ch.doc_id,
                                 chunk_id=ch.chunk_id, location=ch.location, evidence=f"{et}: {val}"))
        for sent in split_sentences(text):
            for e in extract_from_sentence(sent):
                key = (ch.doc_id, e["entity_type"], e["value"].lower(), sent)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({**e, "document": ch.document, "doc_id": ch.doc_id, "chunk_id": ch.chunk_id,
                             "location": ch.location, "evidence": sent[:300]})
    return pd.DataFrame(rows, columns=cols)
