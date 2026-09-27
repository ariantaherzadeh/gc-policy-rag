# gc-policy-rag

RAG demo over Government of Canada Treasury Board policies, built on Cohere's stack.
Portfolio project for a Cohere Forward Deployed Engineer (Sovereign AI) interview.

The full build spec is in `docs/spec.md`. Read it before making changes.

## Division of labour (from the spec)

The agent writes the plumbing. The owner makes the judgment calls. Never finalize these:
- Chunking strategy: keep the chunker pluggable with a simple default; the real strategy is the owner's.
- Eval questions and expected answers: file format plus 2 example rows only.
- Error analysis and README results: never invent metrics or findings.

## Conventions

- Plain Python, no LangChain/LlamaIndex. Raw SQL via psycopg 3, no ORM.
- Check Cohere model IDs and API params against the docs (cohere-docs MCP or
  `https://docs.cohere.com/<page>.md`), never from memory.
- `COHERE_API_KEY` comes only from the environment / `.env`. Never read, print or commit it.
- Trial key budget: 1,000 calls/month. Batch, cache, and avoid unnecessary calls.
- **Never call the Cohere API without asking the owner first**, including tests and smoke checks.
  Say what will run and roughly how many calls it costs, then wait for a yes.

## Workflow: learning-oriented PRs

The owner is building this to learn, so work is split into small PRs they review one at a time.
- Before coding a PR, briefly explain the options and let the owner make the design calls.
- One PR at a time (~150–400 lines); don't start the next until the current one is merged.
- Every PR ends with something the owner can run and see.
- PR descriptions include: what/why, a suggested reading order, commands to try it,
  API calls used (with running total), and a few "check yourself" interview-style questions.
- After each merge the owner writes their own notes in `docs/decisions.md`. Don't write entries for them.

## Running

- `docker compose up -d db` starts Postgres + pgvector.
- `uv sync` installs dependencies; run code with `uv run python -m app.<module>`.
