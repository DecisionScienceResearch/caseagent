"""Case folders on disk.

A case is a directory::

    cases/<key>/
        case.json        the CaseFile
        evidence/        exhibit files named in case.json
        contract.pdf     optional, when the contract is a file rather than text

``CaseBundle`` pairs the parsed CaseFile with that directory so callers can
resolve file paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .documents import ExtractedText, extract_text, normalize_text
from .models import CaseFile, Party, Stage

CASE_FILENAME = "case.json"
EVIDENCE_DIRNAME = "evidence"


class CaseError(Exception):
    """A case folder is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class CaseBundle:
    case: CaseFile
    root: Path

    @property
    def evidence_dir(self) -> Path:
        return self.root / EVIDENCE_DIRNAME

    def evidence_path(self, name: str) -> Path:
        return self.evidence_dir / name

    def evidence_paths(self, stage: Stage) -> list[Path]:
        return [self.evidence_path(name) for name in self.case.evidence.get(stage, [])]

    @property
    def contract_path(self) -> Path | None:
        return self.root / self.case.contract_file if self.case.contract_file else None

    def contract(self) -> ExtractedText:
        """Contract text: inline text if present, otherwise extracted from the file."""
        if self.case.governing_contract and self.case.governing_contract.strip():
            return ExtractedText(normalize_text(self.case.governing_contract))
        if self.contract_path is not None:
            return extract_text(self.contract_path)
        return ExtractedText("")

    def missing_files(self) -> list[str]:
        missing = [
            f"{EVIDENCE_DIRNAME}/{name}"
            for name in self.case.evidence_names()
            if not self.evidence_path(name).is_file()
        ]
        if self.contract_path is not None and not self.contract_path.is_file():
            missing.append(str(self.case.contract_file))
        return missing


def save_case(case: CaseFile, root: Path) -> CaseBundle:
    root.mkdir(parents=True, exist_ok=True)
    (root / EVIDENCE_DIRNAME).mkdir(exist_ok=True)
    payload = case.model_dump(mode="json", exclude_none=True)
    (root / CASE_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return CaseBundle(case=case, root=root)


def load_case(root: Path) -> CaseBundle:
    """Load a case folder (or a path to its case.json) and check its files exist."""
    if root.is_file():
        root = root.parent
    path = root / CASE_FILENAME
    if not path.is_file():
        raise CaseError(f"No {CASE_FILENAME} in {root}")
    try:
        case = CaseFile.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CaseError(f"{path} is not a valid case file:\n{exc}") from exc
    bundle = CaseBundle(case=case, root=root)
    missing = bundle.missing_files()
    if missing:
        raise CaseError(
            f"{path} names files that are not in the case folder: " + ", ".join(missing)
        )
    return bundle


def resolve_case(ref: str, cases_dir: Path) -> CaseBundle:
    """Find a case by key under ``cases_dir`` or by an explicit path."""
    candidates = [Path(ref).expanduser(), cases_dir / ref]
    for candidate in candidates:
        if (candidate / CASE_FILENAME).is_file() or (
            candidate.is_file() and candidate.name == CASE_FILENAME
        ):
            return load_case(candidate)
    raise CaseError(
        f"No case named {ref!r}. Looked for {CASE_FILENAME} in "
        + " and ".join(str(c) for c in candidates)
        + ". Run `caseagent samples fetch` or `caseagent case init`."
    )


def list_cases(cases_dir: Path) -> list[CaseBundle]:
    if not cases_dir.is_dir():
        return []
    bundles = []
    for child in sorted(cases_dir.iterdir()):
        if (child / CASE_FILENAME).is_file():
            bundles.append(load_case(child))
    return bundles


def scaffold_case(root: Path, key: str, title: str | None = None) -> CaseBundle:
    """Write an empty case folder for a dispute of your own.

    The blank fields are filled in by ``caseagent next`` (it asks for them) or by
    editing case.json.
    """
    if (root / CASE_FILENAME).exists():
        raise CaseError(f"{root / CASE_FILENAME} already exists")
    case = CaseFile(
        key=key,
        title=title or key,
        claimant=Party(name=""),
        respondent=Party(name=""),
        question_for_arbitration="",
        financial_demand_usd="0.00",
        evidence={stage: [] for stage in Stage},
        source={"kind": "own"},
    )
    return save_case(case, root)


def add_evidence(bundle: CaseBundle, stage: Stage, paths: list[Path]) -> CaseBundle:
    """Copy files into evidence/ and list them under the phase that files them."""
    case = bundle.case
    names = list(case.evidence.get(stage, []))
    bundle.evidence_dir.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if not source.is_file():
            raise CaseError(f"Not a file: {source}")
        target = bundle.evidence_path(source.name)
        if target.resolve() != source.resolve():
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise CaseError(
                    f"A different file named {source.name!r} is already in {bundle.evidence_dir}"
                )
            target.write_bytes(source.read_bytes())
        if source.name not in names:
            names.append(source.name)
    evidence = {**case.evidence, stage: names}
    return save_case(case.model_copy(update={"evidence": evidence}), bundle.root)
