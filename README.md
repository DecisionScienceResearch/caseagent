# caseagent

An agent that argues an arbitration, one filing at a time, and tests the result on
the [DecisionLayer](https://staging.decisionlayer.ai) simulator.

A case is a dispute (claim form, contract, exhibits) plus a *thread*: the filings made
so far. The agent writes the next filing for whichever side is due, against the
thread as it actually stands, trying to win for that side:

| Phase (Rule 1(b)) | Filed by   | What the agent writes                                                              | Simulator field       | Case API round  |
|-------------------|------------|------------------------------------------------------------------------------------|-----------------------|-----------------|
| **Claim**         | Claimant   | opening argument and remedies                                                      | `plaintiff_argument`  | the case itself |
| **Answer**        | Respondent | responses, defenses, affirmative defenses, a **counterclaim** if the record supports one | `respondent_argument` | round 1         |
| **Reply**         | Claimant   | point-by-point reply, defenses to any counterclaim                                 | `plaintiff_rebuttal`  | round 2         |
| **Response**      | Respondent | final response                                                                     | `respondent_rebuttal` | round 3         |

Two ways to use it:

- **Your own side.** `caseagent next CASE` writes the next filing due. Starting a case
  of your own, the agent first asks you about the dispute and lets you attach
  documents; later it replies to whatever has been filed, whether the agent wrote
  it or you pasted in what the other side sent.
- **Simulation.** `caseagent simulate CASE` has the agent play every remaining phase
  for both sides, one turn at a time, then submits the case and reports the award.

Every argument is the agent's. DecisionLayer's sample disputes are ingested as the
dispute only; the sample authors' arguments are never read by the agent or sent
anywhere (they are saved to `sample-arguments.md` purely for you to compare against).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is fetched automatically.

```sh
uv sync
cp .env.example .env        # OPENROUTER_API_KEY and DECISIONLAYER_API_KEY
uv run caseagent doctor     # checks keys, API access, the model catalog
```

Models come through [OpenRouter](https://openrouter.ai) by default, so one key covers
Anthropic, OpenAI, Google, xAI, DeepSeek, Meta, Mistral and others; the default is
`anthropic/claude-opus-5.5`. `openai:<model>` calls OpenAI directly (`OPENAI_API_KEY`);
`custom:<model>` calls any OpenAI-compatible endpoint (`CASEAGENT_LLM_BASE_URL`, e.g.
Ollama).

The simulator needs a **production** DecisionLayer key (`dvarb_...`); test keys get 403.
Each simulation counts against the monthly allowance, so nothing is submitted without
`simulate` (which confirms first) or `submit`.

## Getting started

`uv run caseagent tour` walks through the steps below in the terminal, explaining each
command and running it only when you say so. The same steps, by hand:

Cheapest first: the first real model call writes one filing, not a whole simulation.

**1. Keys and a health check.** Put `OPENROUTER_API_KEY` in `.env` (and a production
`DECISIONLAYER_API_KEY` when you reach step 5), then:

```sh
uv run caseagent doctor
uv run caseagent samples fetch --all          # -> cases/<key>/ (dispute, contract, exhibits)
```

**2. See the mechanics for free.** No keys needed; the fake model writes placeholders:

```sh
uv run caseagent simulate saas-outage -m fake:demo --depth quick --dry-run
uv run caseagent show latest
```

**3. One real filing.** `saas-outage` is the smallest sample (a short ToS, no exhibits,
a genuinely arguable issue). This is a single model call, well under a dollar:

```sh
uv run caseagent next saas-outage --depth quick
```

Read `cases/saas-outage/filings/1-claim.txt`, and the exact prompt the agent saw in
`cases/saas-outage/work/1-claim/materials.txt`. Then have the other side answer it and
check that the Answer engages the Claim's actual points:

```sh
uv run caseagent reply saas-outage
uv run caseagent case show saas-outage --full
uv run caseagent case reset saas-outage --yes # rewind when done
```

**4. A full head-to-head, no simulator yet.** The default depth is thorough (brief,
draft, review), so this is twelve model calls; on Claude Opus 5.5 at high effort expect
a few dollars for this case and several times that for the exhibit-heavy ones
(`invoice-balance`, `wedding-package`, `driver-deactivation`). `--depth standard`
roughly halves it; `--through answer` writes only the Claim and Answer.

```sh
uv run caseagent simulate saas-outage --dry-run
```

Each phase is written by a separate agent for that side, against the filings before
it. It sees the claim form, the contract, the exhibits on the record and its own exhibits
for the phase, never the other side's private brief or statements, so the Reply really
answers the Answer and the Response really answers the Reply. Read
`runs/<id>/transcript.md`, and the `notes-for-principal.md` files under `work/`: they say
what the agent thought the record lacked.

**5. Spend a simulation.** Submit the dry run (or skip `--dry-run` next time):

```sh
uv run caseagent submit latest                # waits for the award; AWARD paragraph printed
```

The full text is in `runs/<id>/award.txt`; `caseagent runs` lists every run's outcome.

**6. Compare models.** Swap sides too, since a case may simply favor one side:

```sh
uv run caseagent simulate saas-outage --claimant-model anthropic/claude-opus-5.5 --respondent-model openai/gpt-6.1-sol
uv run caseagent simulate saas-outage --claimant-model openai/gpt-6.1-sol --respondent-model anthropic/claude-opus-5.5
uv run caseagent runs
```

`caseagent models [words]` browses the catalog; `-p` picks interactively.

**7. Your own dispute.** See the next section. If a command fails, `CASEAGENT_DEBUG=1`
in front of it prints the full traceback.

## Arguing your own dispute

```sh
uv run caseagent case init cases/my-dispute
uv run caseagent next cases/my-dispute
```

With nothing filed, `next` is initiating: it asks for the essentials (parties, the
dispute as it should read on the claim form, the amount, the contract), then the agent
asks you questions about the facts and documents, you attach what you have, and it
writes the Claim. Your answers are saved as your side's private statement and reused
in later phases.

If you are the respondent, record the claim you received and let the agent answer it:

```sh
uv run caseagent case init cases/their-claim --title "Deposit dispute"
uv run caseagent case add-filing cases/their-claim claim claim-we-received.txt
uv run caseagent case add-evidence cases/their-claim answer signed-addendum.pdf march-emails.pdf
uv run caseagent next cases/their-claim --interview
```

`next` always writes whichever filing is due next, so the same case can be driven
from either side: write your Claim, paste in their Answer with `add-filing`, write your
Reply. Or have the agent draft the other side's filing too, to see what you are up
against, and `caseagent simulate cases/my-dispute` to finish both sides and get an award.
`case show CASE --full` prints the thread; `case reset CASE [--last]` rewinds it.

Each filing is saved in the case folder under `filings/`, with the agent's prompts,
brief, draft and *notes for the principal* (what to verify, what the record lacks)
under `work/`. Notes are never filed.

## Commands

| Command | What it does |
|---|---|
| `samples list` / `samples fetch KEY... \| --all` | Ingest DecisionLayer's sample disputes (no arguments) |
| `case init DIR` / `list` / `show CASE [--full]` | Start, list, inspect cases and their threads |
| `case add-filing CASE STAGE FILE` / `add-evidence CASE STAGE FILES...` / `reset CASE` | Record a filing you have, attach exhibits, rewind |
| `next CASE [-m MODEL] [--interview]` (also `start`, `reply`) | Write the next filing due; ask the party first when starting a case of your own |
| `simulate CASE [-m] [--claimant-model] [--respondent-model] [--through STAGE] [--dry-run]` | Agent plays all remaining phases for both sides, then submits |
| `submit RUN \| latest` | Submit a dry run (or resubmit) and save the award |
| `runs` / `show RUN [--award]` | Compare runs; print a transcript and award |
| `sim list` / `sim status ID` / `sim result ID [--run RUN]` | Simulator API helpers (resume a run that timed out) |
| `live status CASE_ID` / `live respond CASE_ID --materials CASE [--submit]` | Draft (and, after you affirm, file) the next round on a real or test-key case |
| `models [WORDS...]` / `doctor` | Browse the model catalog; check configuration |
| `tour [--from N] [--case KEY] [--print]` | Guided first run: the Getting started steps, one at a time |

Composition options on `next` and `simulate`: `--depth quick|standard|thorough`,
`--counterclaims auto|never`, `--evidence-demands`, `--max-words N`, `--effort low..max`,
`--max-tokens N`, `--temperature`, `--pick` (interactive model picker).

## How a filing is written

The system prompt tells the agent whose side it is on and what winning means for that
side (the full relief demanded for a claimant; denial and any supportable counterclaim
for a respondent), then how DecisionLayer arbitrations are decided (Rules 1, 3, 3A):
plain text of the agreement first, burdens of proof, the record controls, unanswered
arguments may be deemed admitted, damages need record support, relief must be requested
expressly, citations are verified. Then, with `--depth thorough` (the default):

1. **Brief**: a private analysis: burdens and elements, record facts with cites,
   controlling contract language, the other side's points and the answer to each,
   weaknesses, relief, possible counterclaims.
2. **Draft**: the filing in tagged plain-text sections (argument, counterclaim and
   amount, documentation demands/response, notes for the principal).
3. **Review**: a cite-check against the record; unsupported facts and unverifiable
   legal citations are removed, unanswered points answered, relief made explicit.

The agent argues only from the record and the party's own statements, never invents
facts or authority, treats the other side's filings and exhibits as evidence rather than
instructions, and never addresses the AI arbitrator (filings with adversarial prompts
are rejected). Structured parts map onto the APIs differently: the case API has fields
for counterclaims and documentation demands; the simulator has one text box per phase,
so there they are folded in under plain headings.

## Picking models

Any id from `caseagent models` works with `-m`. Settings are fitted to each model from
OpenRouter's catalog: reasoning effort (`CASEAGENT_EFFORT`, default `high`) is only sent
to models that list that level, the output cap stays within the model's limit and
context window, and Claude models get a prompt-cache breakpoint on the case materials.
Temperature is never sent unless you set it. If the materials would not fit a model's
context window the run stops and says so; nothing is truncated silently.

## Layout

```
src/caseagent/
  core/           the agent. No HTTP, no API knowledge.
    models.py       Role, Stage, Filing, Statement, CaseFile
    casefile.py     case folders on disk
    thread.py       the filings made so far (ThreadStore)
    documents.py    PDF/DOCX/RTF/TXT text extraction
    view.py         what a party sees at a phase (the information boundary)
    prompts.py      system prompt, case materials, interview/brief/draft/review tasks
    parsing.py      tagged reply -> Filing; interview questions
    agent.py        questions, brief -> draft -> review
    pipeline.py     continue a thread: write every remaining phase in order
    runs.py         simulation runs (a copied thread plus run.json)
    llm.py          the ChatClient port the core depends on
  llm/            model access: OpenRouter / OpenAI-compatible client, catalog, fake model
  decisionlayer/  the API work: HTTP client, sample ingestion, simulator and live-case mapping
  cli.py, ui.py   Typer command line
tests/            unit tests with mocked HTTP; architecture tests keep core free of API imports
                  and make sure nothing reads sample-arguments.md
```

## Live cases

`live respond` reads a real case through the API (status, whose turn, which fields the
round accepts, the thread so far), composes the next filing with the contract and
exhibits from a local case folder, and saves a draft. With `--submit` it asks you to
affirm the filing (the affirmation is the party's, not the agent's) and posts it. Real
(non-test) cases additionally require `--allow-production`. A `dvarb_test_` key per party
lets you exercise all three rounds without signing, payment or email; see "Test the full
response flow" in the API guide.

Awards are AI-generated simulations and not legal advice; see DecisionLayer's terms.
