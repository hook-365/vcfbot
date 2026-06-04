#!/usr/bin/env python3
"""vcfbot accuracy regression harness.

Runs a fixed battery of realistic engineer questions (upgrading / new
deployments / understanding the product) plus the flagship sizing question
through the live /api/chat pipeline, and reports per-question:

  - retrieved sources (count, with page + section)
  - inline citation count (groundedness signal)
  - whether the model honestly said "not in the context" (IDK)
  - light BEHAVIORAL smoke checks — see CASES below

The checks are deliberately NOT version-specific facts (page numbers, exact
GB/host counts) — those rot when Broadcom updates the docs, which is the bias
this project avoids. They assert behaviour we never want to regress (answers
stay grounded; the flagship keeps VCF Operations OUT of an "optional" section).
Edit CASES freely; this is a smoke test, not ground truth.

Usage:
  python scripts/eval.py                  # all cases against :8129
  python scripts/eval.py --category upgrade
  python scripts/eval.py --quick          # flagship only
  python scripts/eval.py --url http://host:8765/api/chat
  python scripts/eval.py --json results.json

Exit code is non-zero if any case fails a check — usable in CI / pre-deploy.
Note: each question is a real chat completion (cost depends on CHAT_PROVIDER;
~$0.01/q on Haiku). Stdlib only — no extra deps, runs anywhere. The chat model
is non-deterministic, so a borderline case can flap ±1 between runs — re-run
once before treating a single failure as a real regression.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8129/api/chat"

# Each case: category, question, and optional checks.
#   must_cite   — answer must contain >=1 inline [.. p.N ..] citation
#   not_idk     — answer must NOT say "not in the context" (it IS answerable)
#   include     — list of substrings expected somewhere (case-insensitive);
#                 keep these STABLE/version-neutral (component names, not numbers)
#   not_optional— component name that must NOT sit under an "Optional" heading
CASES = [
    {
        "cat": "flagship",
        "q": ("What are the RAM, CPU and storage requirements for all components "
              "of a minimally supported simple VCF deployment?"),
        "must_cite": True, "not_idk": True,
        "include": ["sddc manager", "vcenter", "nsx manager", "vcf operations"],
        "not_optional": "VCF Operations",
    },
    {
        "cat": "upgrade",
        "q": ("What is the supported order to upgrade VCF 9.1 components — which "
              "component is upgraded first?"),
        "must_cite": True, "not_idk": True, "include": ["vcf operations"],
    },
    {
        "cat": "upgrade",
        "q": ("Can I upgrade directly from VCF 5.x to VCF 9.1, and is it in-place "
              "or a migration?"),
        "must_cite": True, "include": ["5.2"],
    },
    {
        "cat": "upgrade",
        "q": ("How does SDDC Manager handle lifecycle management and applying "
              "updates/patches across the fleet?"),
        "must_cite": True, "not_idk": True,
    },
    {
        "cat": "deploy",
        "q": "What are the prerequisites for deploying a new VCF 9.1 management domain?",
        # No not_idk here ON PURPOSE: greenfield procedural prereqs exist in the
        # corpus but the literal query embeds nearer the UPGRADE "deploy mgmt
        # services" pages, so retrieval under-serves them. The scenario rule in
        # SYSTEM_PROMPT makes the model correctly SAY the context is upgrade-
        # focused rather than pass off upgrade steps as greenfield — honest
        # partial-IDK is the right behaviour until retrieval-side query
        # decomposition closes the gap. Tightening retrieval here is open work.
        "must_cite": True,
    },
    {
        "cat": "deploy",
        "q": ("What is the minimum number of ESXi hosts for a VCF management "
              "domain and what shared storage does it require?"),
        "must_cite": True, "not_idk": True, "include": ["vsan"],
    },
    {
        "cat": "deploy",
        "q": ("What is the VCF Installer and what does the initial bring-up of a "
              "new VCF fleet involve?"),
        "must_cite": True, "not_idk": True, "include": ["vcf installer"],
    },
    {
        "cat": "product",
        "q": "What is the difference between VCF Operations and VCF Automation?",
        "must_cite": True, "not_idk": True,
        "include": ["vcf operations", "vcf automation"],
    },
    {
        "cat": "product",
        "q": "What is a VCF fleet versus a VCF instance?",
        "must_cite": True, "not_idk": True, "include": ["fleet", "instance"],
    },
    {
        "cat": "product",
        "q": "What is new in VCF 9.1 compared to earlier versions?",
        "must_cite": True, "not_idk": True,
    },
]

_CITE_RE = re.compile(r"\[[^\]]*p\.[0-9]+[^\]]*\]")
# Catches honest "it's not in the context" hedges. Word-anchored so it does NOT
# fire on incidental phrasings like "does not integrate" / "not included in the
# fan-out" — an early version matched "not in" inside "not integrate".
_IDK_RE = re.compile(
    r"context (?:provided )?does not|"
    r"does not (?:contain|specify|provide|include|mention|detail)|"
    r"not (?:specified|provided|mentioned|detailed|present|available) in|"
    r"cannot find|no information (?:on|about|in|is)|don't have (?:enough|that|the)",
    re.I,
)


def ask(url: str, question: str) -> tuple[list[dict], str]:
    body = json.dumps({"question": question, "history": []}).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    hits: list[dict] = []
    answer: list[str] = []
    event = None
    with urllib.request.urlopen(req, timeout=240) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                if not payload:
                    continue
                data = json.loads(payload)
                if event == "sources":
                    hits = data.get("hits", [])
                elif event == "done" and data.get("answer"):
                    answer = [data["answer"]]
    return hits, "".join(answer)


def under_optional_heading(text: str, component: str) -> bool:
    """True if `component`'s first mention sits below an 'Optional' heading."""
    cur_optional = False
    cl = component.lower()
    for line in text.split("\n"):
        s = line.strip()
        is_head = s.startswith("#") or (
            s.startswith("**") and s.endswith("**") and len(s) < 60
        )
        if is_head:
            cur_optional = "optional" in s.lower()
        if cl in line.lower():
            return cur_optional
    return False


def check(case: dict, hits: list[dict], answer: str) -> list[str]:
    """Return a list of failure messages ([] == all checks passed)."""
    fails: list[str] = []
    a = answer.lower()
    if case.get("must_cite") and not _CITE_RE.search(answer):
        fails.append("no inline citation")
    if case.get("not_idk") and _IDK_RE.search(answer):
        fails.append("said IDK on an answerable question")
    for term in case.get("include", []):
        if term.lower() not in a:
            fails.append(f"missing expected term: '{term}'")
    comp = case.get("not_optional")
    if comp and under_optional_heading(answer, comp):
        fails.append(f"REGRESSION: '{comp}' classified under an Optional heading")
    if not answer.strip():
        fails.append("empty answer")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description="vcfbot accuracy regression harness")
    ap.add_argument("--url", default=DEFAULT_URL, help="chat endpoint")
    ap.add_argument("--category", help="run only this category "
                    "(flagship/upgrade/deploy/product)")
    ap.add_argument("--quick", action="store_true", help="flagship only")
    ap.add_argument("--json", dest="json_out", help="write full results to this file")
    ap.add_argument("--verbose", "-v", action="store_true", help="print full answers")
    args = ap.parse_args()

    cases = CASES
    if args.quick:
        cases = [c for c in cases if c["cat"] == "flagship"]
    elif args.category:
        cases = [c for c in cases if c["cat"] == args.category.lower()]
    if not cases:
        print("no matching cases", file=sys.stderr)
        return 2

    results = []
    passed = 0
    for i, case in enumerate(cases, 1):
        try:
            hits, answer = ask(args.url, case["q"])
        except Exception as exc:  # noqa: BLE001
            print(f"\n[{i}/{len(cases)}] {case['cat']}: REQUEST ERROR — {exc}")
            results.append({"q": case["q"], "error": str(exc)})
            continue
        fails = check(case, hits, answer)
        ok = not fails
        passed += ok
        cites = len(set(_CITE_RE.findall(answer)))
        status = "PASS" if ok else "FAIL"
        print(f"\n[{i}/{len(cases)}] {status}  ({case['cat']})  {case['q']}")
        print(f"    sources={len(hits)}  citations={cites}  chars={len(answer)}")
        if fails:
            for f in fails:
                print(f"    ✗ {f}")
        if args.verbose:
            print("    " + "\n    ".join(answer.strip().split("\n")))
        results.append({
            "cat": case["cat"], "q": case["q"], "ok": ok, "fails": fails,
            "sources": len(hits), "citations": cites, "answer": answer,
        })

    print("\n" + "=" * 70)
    print(f"RESULT: {passed}/{len(cases)} passed")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.json_out}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
