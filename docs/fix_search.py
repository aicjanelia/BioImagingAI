#!/usr/bin/env python3
"""
Post-render fix for Quarto's search.json.

1. Splits each level-2 entry into separate entries for its level-3+
   subsections, so results link to the exact subsection anchor.
2. Removes the (hidden) per-chapter bibliography text from every entry.

Usage (normally run by Quarto as a post-render script):
    python fix_search.py [output_dir]
Output dir defaults to $QUARTO_PROJECT_OUTPUT_DIR, then "_site".

Requires: pip install beautifulsoup4 lxml
"""
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

REF_SELECTORS = "#refs, div.references, div.csl-bib-body, div.csl-entry"
SUB_LEVELS = ("level3", "level4", "level5", "level6")


BLOCK_TAGS = ["p", "li", "div", "pre", "figcaption", "table", "tr",
              "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "dt", "dd"]


def clean_text(node, blocks=True):
    """Newline between block elements; inline elements join with no break."""
    if blocks:
        for el in node.find_all(BLOCK_TAGS):
            el.append("\n")
            el.insert(0, "\n")
    text = node.get_text("")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def own_content(section):
    """Text of a section excluding nested subsections and bibliography."""
    clone = BeautifulSoup(str(section), "lxml").find("section")
    for sub in list(clone.find_all("section", recursive=True)):
        if sub.decomposed:  # already removed along with a parent subsection
            continue
        if sub is not clone and any(c in SUB_LEVELS for c in sub.get("class", [])):
            sub.decompose()
    for ref in clone.select(REF_SELECTORS):
        ref.decompose()
    h = clone.find(re.compile(r"^h[1-6]$"))  # heading is added back by caller
    if h:
        h.decompose()
    return clean_text(clone)


def heading_text(section):
    h = section.find(re.compile(r"^h[1-6]$"))
    return clean_text(h, blocks=False) if h else ""


def process(out_dir):
    sj = out_dir / "search.json"
    if not sj.exists():
        print("fix_search: no search.json found, nothing to do")
        return
    entries = json.loads(sj.read_text(encoding="utf-8"))
    soups = {}
    result, seen = [], set()
    n_split = n_refs = 0

    def add(e):
        if e["objectID"] not in seen:
            seen.add(e["objectID"])
            result.append(e)

    for e in entries:
        href = e.get("href", "")
        page, _, frag = href.partition("#")
        page_path = out_dir / unquote(page)
        if not frag or not page_path.exists():
            add(e)
            continue
        if page not in soups:
            soups[page] = BeautifulSoup(
                page_path.read_text(encoding="utf-8"), "lxml")
        section = soups[page].find("section", id=unquote(frag))
        if section is None:
            add(e)
            continue

        had_refs = bool(section.select(REF_SELECTORS))
        n_refs += had_refs

        if any(c in SUB_LEVELS for c in section.get("class", [])):
            # Already a level-3+ entry (e.g. re-run): rebuild flat
            subs = []
        else:
            subs = [s for s in section.find_all("section")
                    if any(c in SUB_LEVELS for c in s.get("class", []))]

        head = heading_text(section) or e.get("section", "")
        body = own_content(section)
        main = dict(e)
        main["section"] = head
        main["text"] = (head + "\n" + body).strip() if body else head
        add(main)

        for s in subs:
            sid = s.get("id")
            if not sid:
                continue
            shead = heading_text(s)
            sbody = own_content(s)
            if not shead and not sbody:
                continue
            sub = dict(e)
            sub["objectID"] = f"{page}#{sid}"
            sub["href"] = f"{page}#{sid}"
            sub["section"] = shead
            sub["text"] = (shead + "\n" + sbody).strip() if sbody else shead
            add(sub)
            n_split += 1

    # Same byte format Quarto writes: 2-space indent, raw UTF-8, no BOM,
    # no trailing newline. Validate before replacing the original file.
    payload = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
    json.loads(payload.decode("utf-8"))
    tmp = sj.with_suffix(".json.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, sj)
    print(f"fix_search: {len(entries)} -> {len(result)} entries "
          f"({n_split} subsections split out, "
          f"bibliography stripped from {n_refs} entries)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "QUARTO_PROJECT_OUTPUT_DIR", "_site")
    try:
        process(Path(out))
    except Exception as exc:  # leave Quarto's original search.json untouched
        print(f"fix_search: FAILED ({exc!r}); original search.json kept",
              file=sys.stderr)
