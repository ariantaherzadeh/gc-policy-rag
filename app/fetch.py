"""Download every document in the manifest into data/raw/.

    uv run python -m app.fetch            # skip files already downloaded
    uv run python -m app.fetch --force    # re-download everything

Only talks to the TBS website, never to Cohere.
"""

import argparse
import time

import httpx

from app.corpus import RAW_DIR, load_manifest, raw_path

# canada.ca rejects requests without browser-style Accept headers. We identify the
# project honestly rather than pretending to be a browser.
HEADERS = {
    "User-Agent": "gc-policy-rag/0.1 (+https://github.com/ariantaherzadeh/gc-policy-rag)",
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-CA,en;q=0.9,fr-CA;q=0.8",
}
DELAY_SECONDS = 1.0  # be polite to a government website


def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=30) as client:
        for doc in load_manifest():
            path = raw_path(doc)
            if path.exists() and not force:
                print(f"skip   {doc.id} (already at {path.relative_to(RAW_DIR.parent.parent)})")
                continue

            response = client.get(doc.url)
            response.raise_for_status()
            # The site's firewall answers some blocked requests with HTTP 200 and an error page.
            if b"Request Rejected" in response.content[:500]:
                raise RuntimeError(f"{doc.id}: request rejected by the site's firewall")

            path.write_bytes(response.content)
            print(f"saved  {doc.id} ({len(response.content):,} bytes)")
            time.sleep(DELAY_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    fetch_all(force=parser.parse_args().force)
