"""caseagent: compose arbitration filings with an LLM and test them on DecisionLayer.

The package is split along one boundary that matters:

* ``caseagent.core`` is the agent itself: the case model, what each party can
  see at each phase, the prompts, and the brief -> draft -> review loop. It
  knows nothing about HTTP or the DecisionLayer API.
* ``caseagent.llm`` is model access (OpenRouter or any OpenAI-compatible API).
* ``caseagent.decisionlayer`` is all the API work: the HTTP client, sample-case
  ingestion, and the mapping between core filings and API payloads.

``caseagent.cli`` wires the three together.
"""

__version__ = "0.1.0"
