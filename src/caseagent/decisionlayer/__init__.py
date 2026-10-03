"""Everything that talks to DecisionLayer.

* ``client``    - the HTTP client for the public API (``/api/v1``).
* ``samples``   - ingest the simulator's sample cases into local case folders.
* ``simulator`` - map a played case onto ``POST /api/v1/simulations`` and fetch the award.
* ``live``      - map filings onto the turns of a real (or test-key) case.

The agent in ``caseagent.core`` never imports this package.
"""
