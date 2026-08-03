"""Tier-2 live tests: real LLM proposals for customizable natural-language goals.

Each spec in test_scripts/goal_specs/*.json describes a behavior goal (e.g.
"go left stably, then right stably") and structural expectations for the
proposal the pipeline must produce. The runner calls propose_task_config
directly (no HTTP server needed) with the real two-stage LLM pipeline, then
checks:

  - source_type == "llm"  (the LLM proposal survived, no heuristic fallback)
  - the proposal adds custom reward content beyond the default template
  - the custom content references the variables the goal demands
    (must_reference_groups: every group needs at least one match)

COSTS API CALLS. Use --dry-run to inspect specs, --only NAME for one spec.
Every full proposal is saved under test_scripts/live_results/ (gitignored) so
failures can be inspected and turned into Tier-1 fixtures.

Run:  python test_scripts/test_live_goal_proposals.py [--only NAME] [--llm-id ID]
                                                      [--strategy retry|fallback]
                                                      [--list] [--dry-run]
Exit code 0 = all specs pass, 1 = failures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPECS_DIR = Path(__file__).resolve().parent / "goal_specs"
RESULTS_DIR = Path(__file__).resolve().parent / "live_results"


def load_specs(only: str | None) -> list[dict[str, Any]]:
    specs = []
    for path in sorted(SPECS_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8-sig"))
        spec["_path"] = path
        if only and only.lower() not in str(spec.get("name", path.stem)).lower():
            continue
        specs.append(spec)
    return specs


def references(expressions: list[str], variable: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(variable)}\b")
    return any(pattern.search(expr or "") for expr in expressions)


def collect_custom_content(backend, proposal: dict[str, Any], env_name: str) -> tuple[list[dict], list[str]]:
    """Terms the LLM added or rewrote (vs the default template) + all custom expressions."""
    template = {term["key"]: term for term in backend.reward_template_for_env(env_name)}
    custom_terms = [
        term for term in proposal.get("reward_terms") or []
        if term.get("key") not in template
        or str(term.get("expression") or "") != str(template[term["key"]].get("expression") or "")
    ]
    custom_expressions = [str(term.get("expression") or "") for term in custom_terms]
    custom_expressions += [
        str(signal.get("expression") or "")
        for signal in proposal.get("derived_signals") or []
    ]
    return custom_terms, custom_expressions


def check_spec(backend, spec: dict[str, Any], proposal: dict[str, Any]) -> list[str]:
    expect = spec.get("expect") or {}
    failures: list[str] = []
    env_name = spec.get("env_name") or "CartPole-v1"

    if "source_type" in expect and proposal.get("source_type") != expect["source_type"]:
        headline = (proposal.get("warnings") or ["no warnings"])[0]
        failures.append(
            f"source_type: expected {expect['source_type']!r}, got {proposal.get('source_type')!r} — {headline}"
        )

    custom_terms, custom_expressions = collect_custom_content(backend, proposal, env_name)
    if "min_custom_reward_terms" in expect and len(custom_terms) < int(expect["min_custom_reward_terms"]):
        failures.append(
            f"min_custom_reward_terms: expected >= {expect['min_custom_reward_terms']}, "
            f"got {len(custom_terms)} ({[t.get('key') for t in custom_terms]})"
        )

    for group in expect.get("must_reference_groups") or []:
        if not any(references(custom_expressions, variable) for variable in group):
            failures.append(
                f"must_reference_groups: none of {group} appear in the custom expressions "
                f"({custom_expressions})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="Run only specs whose name contains this substring.")
    parser.add_argument("--llm-id", default=None, help="Planner LLM id (see GET /task_config_llms); default = backend default.")
    parser.add_argument("--strategy", default=None, choices=["retry", "fallback"],
                        help="Override the per-spec proposal strategy.")
    parser.add_argument("--list", action="store_true", help="List spec names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print goals and expectations without calling the LLM.")
    args = parser.parse_args()

    specs = load_specs(args.only)
    if not specs:
        print(f"No specs found in {SPECS_DIR}" + (f" matching --only {args.only!r}" if args.only else ""))
        return 1
    if args.list:
        for spec in specs:
            print(f"{spec.get('name', spec['_path'].stem)}: {spec.get('goal', '')[:100]}")
        return 0
    if args.dry_run:
        for spec in specs:
            print(json.dumps({key: value for key, value in spec.items() if key != "_path"}, indent=2))
        return 0

    print(f"Importing backend (main.py) — first import is slow because of torch/gym ...")
    import main as backend

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    passed = 0
    failed = 0
    for spec in specs:
        name = spec.get("name", spec["_path"].stem)
        env_name = spec.get("env_name") or "CartPole-v1"
        strategy = args.strategy or spec.get("strategy") or "retry"
        started = time.time()
        try:
            proposal = backend.propose_task_config(
                spec["goal"],
                env_name,
                run_id=f"live-goal-{name}",
                llm_id=args.llm_id,
                strategy=strategy,
            )
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {name}: propose_task_config raised {type(exc).__name__}: {exc}")
            continue
        elapsed = round(time.time() - started, 1)

        artifact_path = RESULTS_DIR / f"{stamp}_{name}.json"
        artifact_path.write_text(
            json.dumps({"spec": {k: v for k, v in spec.items() if k != "_path"}, "proposal": proposal},
                       indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        failures = check_spec(backend, spec, proposal)
        custom_terms, _ = collect_custom_content(backend, proposal, env_name)
        summary = (
            f"source={proposal.get('source_type')} provider={proposal.get('provider')} "
            f"model={proposal.get('model')} attempts={proposal.get('proposal_attempts')} "
            f"custom_terms={[t.get('key') for t in custom_terms]} elapsed={elapsed}s"
        )
        if failures:
            failed += 1
            print(f"[FAIL] {name} — {summary}")
            for failure in failures:
                print(f"       - {failure}")
        else:
            passed += 1
            print(f"[PASS] {name} — {summary}")
        for warning in (proposal.get("warnings") or [])[:3]:
            print(f"       warning: {warning}")
        print(f"       saved: {artifact_path}")

    print()
    print(f"{passed} passed, {failed} failed out of {passed + failed} spec(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
