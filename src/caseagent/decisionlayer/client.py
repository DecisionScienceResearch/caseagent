"""HTTP client for the DecisionLayer public API.

Written against the OpenAPI document at ``/api/v1/openapi.json`` and the guides
at ``/api/run-a-simulation`` and ``/api/create-a-case``:

* every request carries ``Authorization: Bearer dvarb_...``;
* form endpoints are multipart, with repeatable file fields;
* a file of 8 MiB or more goes through ``POST /api/v1/uploads`` and is sent as
  a ticket in the field's ``*_ticket(s)`` sibling;
* errors are ``{"error": {"status", "message", "details", "reason"}}``; a 5xx
  that carries that body came from the API itself and is replayed for the same
  ``Idempotency-Key``, so only bodiless gateway failures are retried;
* creates accept an ``Idempotency-Key`` so a retry cannot file twice; upload
  tickets a failed create did not consume are cancelled;
* ``POST /api/v1/cases/{id}/sign`` returns a signing URL for the arbitration
  terms when the key owner's ``next_action`` is ``sign_terms``.
"""

from __future__ import annotations

import mimetypes
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__

DEFAULT_BASE_URL = "https://www.decisionlayer.ai"
TEST_KEY_PREFIX = "dvarb_test_"
LARGE_FILE_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024
UPLOAD_BATCH_SIZE = 100

# Each multipart file field and the field that carries its upload tickets.
TICKET_FIELDS = {
    "contract_file": "contract_file_ticket",
    "plaintiff_documents": "plaintiff_documents_tickets",
    "respondent_documents": "respondent_documents_tickets",
    "plaintiff_rebuttal_documents": "plaintiff_rebuttal_documents_tickets",
    "respondent_rebuttal_documents": "respondent_rebuttal_documents_tickets",
    "evidence": "evidence_tickets",
    "evidence_response_files": "evidence_response_files_tickets",
    "counterclaim_files": "counterclaim_files_tickets",
}

# Gateway failures worth retrying; see ``_is_api_error`` for the exception.
RETRY_STATUSES = {502, 503, 504}
RETRY_DELAYS = (2.0, 5.0, 10.0)
# A retried create returns 409 while the first attempt still holds the key's
# lease (about a minute), then replays the stored result.
LEASE_WAIT_SECONDS = 15.0
LEASE_RETRIES = 5


class DecisionLayerError(Exception):
    """An error response from the API, or a request it would certainly reject."""

    def __init__(
        self,
        status: int,
        message: str,
        details: Sequence[str] = (),
        reason: str | None = None,
    ) -> None:
        self.status = status
        self.message = message
        self.details = list(details)
        self.reason = reason
        super().__init__(self.describe())

    def describe(self) -> str:
        head = (
            f"DecisionLayer API error {self.status}: {self.message}"
            if self.status
            else self.message
        )
        return "\n".join([head, *(f"  - {detail}" for detail in self.details)])

    @classmethod
    def from_response(cls, response: httpx.Response) -> DecisionLayerError:
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            return cls(
                status=int(error.get("status") or response.status_code),
                message=str(error.get("message") or response.reason_phrase),
                details=[str(detail) for detail in error.get("details") or []],
                reason=error.get("reason"),
            )
        text = response.text.strip()
        return cls(response.status_code, text[:500] or response.reason_phrase)


class SimulationTimeout(Exception):
    """The award was not ready in time; the simulation may still finish."""

    def __init__(self, simulation_id: str, waited: float) -> None:
        self.simulation_id = simulation_id
        super().__init__(
            f"Simulation {simulation_id} was still running after {waited:.0f}s. "
            f"Check it later with `caseagent sim result {simulation_id}`."
        )


class _ApiModel(BaseModel):
    # The API may add fields; keep them rather than fail.
    model_config = ConfigDict(extra="allow")


class Principal(_ApiModel):
    id: str
    email: str
    first_name: str = ""
    last_name: str = ""
    can_file_cases: bool = False


class Simulation(_ApiModel):
    id: str
    status: str  # processing | retrying | ready | failed
    page_url: str | None = None
    result_url: str | None = None
    message: str | None = None
    created_at: datetime | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("ready", "failed")


class SimulationResult(_ApiModel):
    id: str
    case_id: str
    text: str
    published_at: datetime | None = None
    pdf_url: str | None = None
    page_url: str | None = None


class ArbitrationCase(_ApiModel):
    id: str
    status: str
    role: str
    current_turn: str | None = None
    next_round: int | None = None
    accepted_fields: list[str] = Field(default_factory=list)
    action_required: bool = False
    # complete_form | sign_terms | pay_filing_fee | verify_identity | respond
    next_action: str | None = None
    action_url: str | None = None
    view_url: str | None = None
    question_for_arbitration: str = ""
    financial_demand_usd: str = "0.00"
    other_relief: str | None = None
    argument: str | None = None
    claimant_email: str | None = None
    respondent_type: str = "individual"
    respondent_first_name: str = ""
    respondent_last_name: str = ""
    test: bool = False

    @property
    def respondent_name(self) -> str:
        return f"{self.respondent_first_name} {self.respondent_last_name}".strip()


class EvidenceFile(_ApiModel):
    """A file attached to a response round; the API serves its name, not the file."""

    name: str
    kind: str = "supporting"  # supporting | evidence_response | counterclaim


class CaseResponse(_ApiModel):
    id: str
    case_id: str
    round: int
    submitted_by: str
    argument: str | None = None
    evidence_response: str | None = None
    evidence_demands: str | None = None
    counterclaim_argument: str | None = None
    counterclaim_amount_usd: str | None = None
    counterclaim_other_relief: str | None = None
    evidence_files: list[EvidenceFile] = Field(default_factory=list)
    submitted_at: datetime | None = None
    # When the next round's response is due; null after the final round.
    due_at: datetime | None = None


class SigningSession(_ApiModel):
    """A URL for signing the arbitration terms. Open it before it expires."""

    signing_url: str
    expires_at: datetime | None = None
    provider: str | None = None  # signwell | hosted


class ResponseCreated(_ApiModel):
    response: CaseResponse
    case: ArbitrationCase


class Decision(_ApiModel):
    id: str
    case_id: str
    text: str
    published_at: datetime | None = None
    pdf_url: str | None = None


class SimulationFailed(Exception):
    """The simulator reported that award generation stopped."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        super().__init__(
            f"Simulation {simulation.id} failed: {simulation.message or 'no reason given'}"
        )


@dataclass(frozen=True)
class Attachment:
    """A local file bound for one multipart file field."""

    field: str
    path: Path


def _is_api_error(response: httpx.Response) -> bool:
    """True when the body is the API's own error envelope, i.e. the API handled the request."""
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and isinstance(body.get("error"), dict)


class DecisionLayerClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise DecisionLayerError(
                0,
                "No DecisionLayer API key. Set DECISIONLAYER_API_KEY (create one at /settings/api-keys).",
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": f"caseagent/{__version__}",
            },
            timeout=timeout,
            transport=transport,
        )
        # Upload URLs point at object storage and must not receive the API key.
        self._storage = httpx.Client(timeout=300.0, transport=transport)

    @property
    def is_test_key(self) -> bool:
        return self.api_key.startswith(TEST_KEY_PREFIX)

    def close(self) -> None:
        self._http.close()
        self._storage.close()

    def __enter__(self) -> DecisionLayerClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ account

    def me(self) -> Principal:
        return Principal.model_validate(self._request("GET", "/api/v1/me").json())

    # -------------------------------------------------------------- simulations

    def create_simulation(
        self,
        fields: Mapping[str, str],
        attachments: Sequence[Attachment] = (),
        idempotency_key: str | None = None,
    ) -> Simulation:
        """File every argument and response in one request and queue the award.

        Requires a production key; a test key gets 403.
        """
        response = self._post_form("/api/v1/simulations", fields, attachments, idempotency_key)
        return Simulation.model_validate(response.json())

    def get_simulation(self, simulation_id: str) -> Simulation:
        response = self._request("GET", f"/api/v1/simulations/{simulation_id}")
        return Simulation.model_validate(response.json())

    def get_simulation_result(self, simulation_id: str) -> SimulationResult:
        response = self._request("GET", f"/api/v1/simulations/{simulation_id}/result")
        return SimulationResult.model_validate(response.json())

    def list_simulations(
        self, limit: int = 50, cursor: str | None = None
    ) -> tuple[list[Simulation], str | None]:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        response = self._request("GET", "/api/v1/simulations", params=params)
        simulations = [Simulation.model_validate(item) for item in response.json()]
        return simulations, response.headers.get("X-Next-Cursor")

    def wait_for_simulation(
        self,
        simulation_id: str,
        timeout: float = 900.0,
        interval: float = 10.0,
        on_poll: Callable[[Simulation], None] | None = None,
    ) -> Simulation:
        """Poll until the award is public. Raises on failure or timeout."""
        started = time.monotonic()
        while True:
            simulation = self.get_simulation(simulation_id)
            if on_poll is not None:
                on_poll(simulation)
            if simulation.status == "ready":
                return simulation
            if simulation.status == "failed":
                raise SimulationFailed(simulation)
            waited = time.monotonic() - started
            if waited >= timeout:
                raise SimulationTimeout(simulation_id, waited)
            self._sleep(interval)

    # -------------------------------------------------------------------- cases

    def get_case(self, case_id: str) -> ArbitrationCase:
        return ArbitrationCase.model_validate(
            self._request("GET", f"/api/v1/cases/{case_id}").json()
        )

    def list_responses(self, case_id: str) -> list[CaseResponse]:
        """The response thread, oldest first."""
        response = self._request("GET", f"/api/v1/cases/{case_id}/responses")
        return [CaseResponse.model_validate(item) for item in response.json()]

    def create_response(
        self,
        case_id: str,
        fields: Mapping[str, str],
        attachments: Sequence[Attachment] = (),
        idempotency_key: str | None = None,
    ) -> ResponseCreated:
        """Submit this key's turn on a case. Each round can be submitted once."""
        response = self._post_form(
            f"/api/v1/cases/{case_id}/responses", fields, attachments, idempotency_key
        )
        return ResponseCreated.model_validate(response.json())

    def get_decision(self, case_id: str) -> Decision:
        return Decision.model_validate(
            self._request("GET", f"/api/v1/cases/{case_id}/decision").json()
        )

    def sign_case(self, case_id: str) -> SigningSession:
        """A signing URL for the party whose ``next_action`` is ``sign_terms``.

        Call again for a fresh URL once it expires. 409 when this party is not
        waiting on a signature (a test case never is).
        """
        response = self._request("POST", f"/api/v1/cases/{case_id}/sign")
        return SigningSession.model_validate(response.json())

    def claim_case(self, case_id: str, verification_code: str) -> ArbitrationCase:
        """Become the respondent on a production case with the code from the notice.

        Until then the case reads as 404 with reason ``not_claimed``. The code
        is single-use, but the same account may send it again. 403 with reason
        ``invalid_code`` when it is rejected; the claimant cannot claim.
        """
        response = self._request(
            "POST",
            f"/api/v1/cases/{case_id}/claim",
            json={"verification_code": verification_code},
        )
        return ArbitrationCase.model_validate(response.json())

    # ------------------------------------------------------------------ uploads

    def upload_files(self, paths: Sequence[Path]) -> list[str]:
        """Upload files through the two-step protocol and return their tickets, in order."""
        tickets: list[str] = []
        for start in range(0, len(paths), UPLOAD_BATCH_SIZE):
            batch = list(paths[start : start + UPLOAD_BATCH_SIZE])
            descriptors = [{"filename": path.name, "size": path.stat().st_size} for path in batch]
            response = self._request("POST", "/api/v1/uploads", json={"files": descriptors})
            sessions = response.json()["uploads"]
            if len(sessions) != len(batch):
                raise DecisionLayerError(
                    0, f"Asked for {len(batch)} upload sessions but received {len(sessions)}."
                )
            for path, session in zip(batch, sessions, strict=True):
                body = path.read_bytes()
                put = self._storage.put(
                    session["upload_url"],
                    content=body,
                    headers={
                        "Content-Type": session["content_type"],
                        "Content-Length": str(len(body)),
                    },
                )
                if put.status_code >= 400:
                    raise DecisionLayerError(
                        put.status_code, f"Uploading {path.name} failed: {put.text[:300]}"
                    )
                tickets.append(session["ticket"])
        return tickets

    def cancel_uploads(self, tickets: Sequence[str]) -> int:
        """Discard step-one uploads the API has not consumed; returns how many were removed."""
        deleted = 0
        for start in range(0, len(tickets), UPLOAD_BATCH_SIZE):
            batch = list(tickets[start : start + UPLOAD_BATCH_SIZE])
            response = self._request("POST", "/api/v1/uploads/cancel", json={"tickets": batch})
            deleted += int(response.json().get("deleted") or 0)
        return deleted

    # ---------------------------------------------------------------- internals

    def _post_form(
        self,
        path: str,
        fields: Mapping[str, str],
        attachments: Sequence[Attachment],
        idempotency_key: str | None,
    ) -> httpx.Response:
        parts, tickets = self._form_parts(fields, attachments)
        headers = {"Idempotency-Key": idempotency_key or str(uuid.uuid4())}
        try:
            return self._request("POST", path, files=parts, headers=headers)
        except DecisionLayerError as exc:
            # Tickets the API did not consume would otherwise linger until they expire.
            # A 409 may mean a first attempt is still being processed, so leave those alone.
            if tickets and exc.status != 409:
                with suppress(DecisionLayerError):
                    self.cancel_uploads(tickets)
            raise

    def _form_parts(
        self, fields: Mapping[str, str], attachments: Sequence[Attachment]
    ) -> tuple[list[tuple[str, tuple[Any, ...]]], list[str]]:
        """Build multipart parts; also return the upload tickets they carry, if any.

        Text fields are sent as parts without a filename, which keeps the body
        multipart even when no file is attached (these endpoints do not accept
        urlencoded forms). If any file needs the ticket protocol, every file in
        the request uses it, because the API rejects mixed upload sources.
        """
        parts: list[tuple[str, tuple[Any, ...]]] = [
            (name, (None, value)) for name, value in fields.items() if value not in (None, "")
        ]
        sizes = []
        for attachment in attachments:
            if attachment.field not in TICKET_FIELDS:
                raise DecisionLayerError(0, f"{attachment.field!r} is not a file field of the API.")
            if not attachment.path.is_file():
                raise DecisionLayerError(0, f"Attachment not found: {attachment.path}")
            size = attachment.path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise DecisionLayerError(
                    413, f"{attachment.path.name} is over the 100 MiB per-file limit."
                )
            if size == 0:
                raise DecisionLayerError(0, f"{attachment.path.name} is empty.")
            sizes.append(size)

        tickets: list[str] = []
        if any(size >= LARGE_FILE_BYTES for size in sizes):
            tickets = self.upload_files([attachment.path for attachment in attachments])
            for attachment, ticket in zip(attachments, tickets, strict=True):
                parts.append((TICKET_FIELDS[attachment.field], (None, ticket)))
        else:
            for attachment in attachments:
                mime = mimetypes.guess_type(attachment.path.name)[0] or "application/octet-stream"
                parts.append(
                    (attachment.field, (attachment.path.name, attachment.path.read_bytes(), mime))
                )
        return parts, tickets

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        files: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Send a request, retrying only where a repeat cannot cause a second filing."""
        retryable = method == "GET" or bool(headers and "Idempotency-Key" in headers)
        delays = list(RETRY_DELAYS) if retryable else []
        lease_retries = LEASE_RETRIES if retryable and method != "GET" else 0
        retried = False
        while True:
            try:
                response = self._http.request(
                    method, path, params=params, json=json, files=files, headers=headers
                )
            except httpx.TransportError as exc:
                if not delays:
                    raise DecisionLayerError(0, f"Could not reach {self.base_url}: {exc}") from exc
                self._sleep(delays.pop(0))
                retried = True
                continue
            if response.status_code in RETRY_STATUSES and delays and not _is_api_error(response):
                self._sleep(delays.pop(0))
                retried = True
                continue
            if response.status_code == 409 and retried and lease_retries:
                lease_retries -= 1
                self._sleep(LEASE_WAIT_SECONDS)
                continue
            if response.status_code >= 400:
                raise DecisionLayerError.from_response(response)
            return response
