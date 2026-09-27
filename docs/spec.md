# RAG demo app — build spec

## Purpose

A small, local RAG (retrieval-augmented generation) system over a fixed set of
Government of Canada Treasury Board policy documents. It's a portfolio project for a
Cohere Forward Deployed Engineer interview (Sovereign AI / public sector), built
on Cohere's stack.

What the project needs to show, in priority order:
1. A working retrieve → rerank → generate pipeline with citations.
2. **Measured** retrieval quality (evals, plus an ablation with and without rerank).
3. Handling of real enterprise problems: bilingual EN/FR content, superseded
   documents, and "I don't know" when the answer isn't in the corpus.

## Division of labour — read this first

The owner is building this partly to learn. **The agent writes the plumbing. The
owner makes the judgment calls.** Leave these as clearly marked TODOs or pluggable
defaults, and don't finalize them yourself:

- **Chunking strategy** (size, overlap, where to split). Implement a pluggable chunker
  with a simple default, and leave the real strategy to the owner.
- **Eval questions and expected answers.** Provide the file format and 2 example rows only.
- **Error analysis and README results.** Never invent metrics or findings.

## Stack

| Piece | Choice |
|---|---|
| Language | Python 3.12 |
| API | FastAPI |
| Model calls | `cohere` Python SDK (Embed, Rerank, Chat) |
| Storage | Postgres + pgvector (Docker) |
| UI | One static HTML page with vanilla JS, served by FastAPI |
| Packaging | Docker Compose (API + Postgres) |

**Don't use** LangChain or LlamaIndex. Each stage should be plain Python the owner can explain.

## Architecture

```
Ingest (once):   fetch → data/raw/ → parse → chunk → embed (batched, cached) → pgvector
Per question:    UI → POST /chat → rewrite follow-up → vector search (top 50, filtered)
                 → rerank (top 8) → Chat with documents → answer + citations → UI
```

### Provider seam

Hide each stage behind a small interface, with Cohere as the default provider, so
another provider can be swapped in through config:

```python
class Embedder(Protocol):
    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]: ...

class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]: ...

class Generator(Protocol):
    def answer(self, question: str, chunks: list[Chunk], history: list[Turn]) -> Answer: ...
```

`Answer` = text plus **normalized** citations (`chunk_id`, `start`, `end`, `text`), so
the UI never depends on one provider's citation format. Store the embedding model name
with each vector and never mix models in one index. Changing models means a full re-index.

## Corpus

- A fixed set of 6–10 documents, listed in `rag/corpus/manifest.yaml`. No upload feature.
- Manifest fields: `id`, `title`, `url`, `language` (en/fr), `published`,
  `status` (current/superseded), `format` (html/pdf), `access_level` (optional).
- `fetch.py` downloads everything into `data/raw/`, which is gitignored. Commit the manifest, not the files.
- Prefer the **HTML** versions on the TBS site: they have clean numbered sections in both languages.
  Include **1–2 PDFs** on purpose (e.g. a table-heavy Estimates document).
- The corpus must include at least one **EN/FR pair** and one **current/superseded pair**.
- Candidates, to be confirmed by the owner (the agent should find and verify the URLs):
  Directive on Automated Decision-Making, Guide on the use of generative AI,
  Policy on Government Security, Directive on Security Management,
  Policy on Service and Digital, one Estimates PDF.

## Functional requirements

- **Ingest:** parse, chunk, and embed in batches with a content-hash cache on disk, so
  re-running ingest never re-embeds unchanged chunks. Upsert into pgvector along with the metadata.
- **Retrieve:** embed the query, then run a cosine search for the top 50 with SQL metadata filters
  (default: `status = current`; optionally language and access level).
- **Rerank:** keep the top 8. It must be possible to turn rerank off (`--no-rerank` / config flag) for the ablation.
- **Generate:** Cohere Chat with a `documents` param, so citations come back natively.
  Abstain when the retrieved passages don't contain the answer.
- **Chat follow-ups:** rewrite each question into a standalone query using the conversation history before retrieving.
- **API:** `POST /chat` returning `{answer, citations[], retrieved[]}`, and `GET /health`.
  Include the retrieved chunks in the response for debugging.
- **UI:** a chat box, the answer, and clickable citations that show the source passage and document title.

## Evals

- `rag/evals/questions.yaml`: each row has `question`, `expected_answer`,
  `expected_sources` (doc id + section), `should_abstain`.
- `python -m app.evals` runs every question and:
  - **automatically** scores recall@8 against `expected_sources`, citation match, and correct abstention
  - prints each Q / expected / actual / citations so the owner can hand-grade
    (correct / partial / wrong + a note)
  - saves each run to `rag/evals/runs/<timestamp>.jsonl` with its config (chunker, rerank on/off),
    so runs can be compared.
- No LLM-as-judge for now.

## Constraints

- **Trial API key: 1,000 calls/month total.** Trial rate limits: Chat 20/min,
  Rerank 10/min, Embed 2,000 inputs/min. Batch embeddings, cache aggressively, and
  run evals sequentially with backoff.
- The API key comes only from the `COHERE_API_KEY` env var. It never goes to the browser and is never committed.
- Check current model IDs and API parameters against the Cohere docs, not from memory.
  Use `https://docs.cohere.com/<page>.md` or the docs MCP in `.mcp.json`. Use a
  **multilingual** Embed model, since the corpus is EN/FR.

## Suggested layout

```
rag/
  app/  providers/  ingest.py  retrieve.py  chat.py  api.py  evals.py  config.py
  corpus/manifest.yaml
  evals/questions.yaml   evals/runs/
  static/index.html
  data/raw/  data/cache/        # gitignored
  docker-compose.yml  pyproject.toml  README.md
```

## Milestones

1. **One document end to end.** Ingest one document and get a cited answer from the command line.
2. **Full corpus + metadata.** Manifest, fetch, filters, API, and UI.
3. **Eval harness.** Automated checks, hand-grading output, saved runs, and the rerank ablation.
4. **Differentiators.** Bilingual retrieval check, superseded-version filtering, optional access-level filter.
5. **Packaging.** `docker compose up` runs everything, plus a README with the architecture and run instructions.
   Leave the results section blank for the owner.
6. **Stretch:** agentic retrieval. Expose `search_documents` as a Chat tool and let the model decide when and what to search.

## Done means

- A fresh clone plus a `COHERE_API_KEY` gets to a working chat at `localhost:8000` with only
  `docker compose up`, `python -m app.fetch`, and `python -m app.ingest`.
- `python -m app.evals` runs with and without rerank and writes comparable run files.
- Out-of-corpus questions get an abstention, not an invented answer.
- Superseded documents are never cited under the default filter.
