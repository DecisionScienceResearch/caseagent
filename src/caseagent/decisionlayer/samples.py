"""Ingest DecisionLayer's sample simulations as local cases.

The simulator page lists its samples in ``/static/js/simulate_cases.json``
(key -> file name) and serves each as JSON under ``/samples/arbitration_cases/``,
with exhibit files alongside. These are public static files; no key is needed.

A sample looks like::

    {
      "arbitration_rules": "...", "governing_contract": "...",
      "question_for_arbitration": "...", "financial_demand_usd": 42000.0,
      "other_relief": "...",
      "plaintiff": {"name": "..."}, "respondent": {"name": "..."},
      "plaintiff_argument":   {"argument": "...", "documents": [], "images": []},
      "respondent_argument":  {"argument": "...", ...},
      "plaintiff_rebuttal":   {"rebuttal": "...", ...},
      "respondent_rebuttal":  {"rebuttal": "...", ...},
      "attachments": {
        "plaintiff_documents": ["attachments/<key>/Exhibit A-1 - ....pdf", ...],
        "respondent_documents": [...], "plaintiff_rebuttal_documents": [...],
        "respondent_rebuttal_documents": [...]
      }
    }

The four argument texts become the case's baseline filings; the four attachment
lists become the exhibits filed with each phase.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from ..core.casefile import CaseBundle, save_case
from ..core.documents import normalize_text
from ..core.models import CaseFile, Party, Stage
from .client import DEFAULT_BASE_URL

INDEX_PATH = "/static/js/simulate_cases.json"
SAMPLES_PATH = "/samples/arbitration_cases/"
SAMPLE_ARGUMENTS_FILENAME = "sample-arguments.md"

# Sample JSON keys for each phase: (argument object, text key, attachments list).
_PHASES = {
    Stage.CLAIM: ("plaintiff_argument", "argument", "plaintiff_documents"),
    Stage.ANSWER: ("respondent_argument", "argument", "respondent_documents"),
    Stage.REPLY: ("plaintiff_rebuttal", "rebuttal", "plaintiff_rebuttal_documents"),
    Stage.RESPONSE: ("respondent_rebuttal", "rebuttal", "respondent_rebuttal_documents"),
}


class SampleError(Exception):
    pass


class SampleStore:
    """Reads the sample index, sample JSON, and exhibit files from the DecisionLayer site."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=60.0, transport=transport)

    def close(self) -> None:
        self._http.close()

    def index(self) -> dict[str, str]:
        return dict(self._get(INDEX_PATH).json())

    def fetch(self, key: str) -> dict[str, Any]:
        index = self.index()
        if key not in index:
            raise SampleError(f"No sample named {key!r}. Available: {', '.join(sorted(index))}.")
        return self._get(SAMPLES_PATH + quote(index[key])).json()

    def attachment(self, relative_path: str) -> bytes:
        encoded = "/".join(quote(segment) for segment in relative_path.split("/"))
        return self._get(SAMPLES_PATH + encoded).content

    def _get(self, path: str) -> httpx.Response:
        try:
            response = self._http.get(path)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SampleError(f"Could not fetch {self.base_url}{path}: {exc}") from exc
        return response


def title_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("-", " ").replace("_", " ")
    words = [word if word.isupper() else word.capitalize() for word in stem.split()]
    return " ".join(words).replace("Saas", "SaaS")


def sample_to_case(key: str, raw: Mapping[str, Any], title: str | None = None) -> CaseFile:
    """Convert a sample's JSON to a CaseFile: the dispute, contract and exhibits only.

    The sample's four written arguments are deliberately left out; the agent
    writes every filing itself. Pure; downloads nothing.
    """
    evidence: dict[Stage, list[str]] = {}
    attachments = raw.get("attachments") or {}
    for stage, (_section, _text_key, documents_key) in _PHASES.items():
        names = [_attachment_name(ref) for ref in attachments.get(documents_key) or []]
        evidence[stage] = [name for name in names if name]

    return CaseFile(
        key=key,
        title=title or key,
        claimant=Party(name=str((raw.get("plaintiff") or {}).get("name") or "Claimant").strip()),
        respondent=Party(
            name=str((raw.get("respondent") or {}).get("name") or "Respondent").strip()
        ),
        question_for_arbitration=normalize_text(str(raw.get("question_for_arbitration") or "")),
        financial_demand_usd=raw.get("financial_demand_usd") or 0,
        other_relief=normalize_text(str(raw.get("other_relief") or "")) or None,
        arbitration_clause=normalize_text(str(raw.get("arbitration_rules") or "")) or None,
        governing_contract=normalize_text(str(raw.get("governing_contract") or "")) or None,
        evidence=evidence,
        source={
            "kind": "decisionlayer-sample",
            "key": key,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def attachment_refs(raw: Mapping[str, Any]) -> dict[str, str]:
    """Map each exhibit file name to its path under the samples directory, deduplicated."""
    refs: dict[str, str] = {}
    for _stage, (_section, _text_key, documents_key) in _PHASES.items():
        for ref in (raw.get("attachments") or {}).get(documents_key) or []:
            path = _attachment_path(ref)
            name = _attachment_name(ref)
            if not path or not name:
                continue
            if name in refs and refs[name] != path:
                raise SampleError(
                    f"Two different exhibit files are both named {name!r}: {refs[name]} and {path}"
                )
            refs[name] = path
    return refs


def _attachment_path(ref: Any) -> str | None:
    if isinstance(ref, str):
        return ref
    if isinstance(ref, Mapping):
        value = ref.get("file") or ref.get("path")
        return str(value) if value else None
    return None


def _attachment_name(ref: Any) -> str | None:
    path = _attachment_path(ref)
    return Path(path).name if path else None


def ingest_sample(
    key: str,
    raw: Mapping[str, Any],
    cases_dir: Path,
    read_attachment: Callable[[str], bytes],
    title: str | None = None,
) -> CaseBundle:
    """Write a sample as a case folder, downloading its exhibits.

    The sample's own arguments go into ``sample-arguments.md`` for a person to
    read next to the agent's work. Nothing in caseagent reads that file.
    """
    case = sample_to_case(key, raw, title=title)
    refs = attachment_refs(raw)
    if case.source is not None:
        case.source["attachments"] = len(refs)
    bundle = save_case(case, cases_dir / key)
    for name, path in refs.items():
        bundle.evidence_path(name).write_bytes(read_attachment(path))
    (bundle.root / SAMPLE_ARGUMENTS_FILENAME).write_text(
        sample_arguments_markdown(case, raw), encoding="utf-8"
    )
    return bundle


def sample_arguments_markdown(case: CaseFile, raw: Mapping[str, Any]) -> str:
    lines = [
        f"# {case.title}: the sample's own arguments",
        "",
        "Reference only. These are the human-written arguments that ship with the "
        "DecisionLayer sample. caseagent never reads this file: every filing it "
        "produces is written by the agent from the dispute, the contract and the exhibits.",
        "",
    ]
    for stage, (section, text_key, _documents_key) in _PHASES.items():
        block = raw.get(section) or {}
        text = normalize_text(str(block.get(text_key) or block.get("argument") or ""))
        lines += [f"## {stage.title} ({case.party(stage.role).name})", "", text or "_empty_", ""]
    return "\n".join(lines)


def fetch_sample(key: str, cases_dir: Path, store: SampleStore) -> CaseBundle:
    index = store.index()
    if key not in index:
        raise SampleError(f"No sample named {key!r}. Available: {', '.join(sorted(index))}.")
    raw = store.fetch(key)
    return ingest_sample(
        key, raw, cases_dir, store.attachment, title=title_from_filename(index[key])
    )


def sample_key(value: str) -> str:
    """Normalize a user-typed sample name to an index key."""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
