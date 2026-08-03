"""Tier-1 regression suite for the LLM task-config proposal pipeline.

Replays recorded model outputs (fixtures) through the exact production code path
used after an LLM response arrives — parse_llm_response_text ->
normalize_task_config_proposal_shape -> validate_task_config_for_run ->
normalize_reward_terms — with zero network calls and zero API cost.

Each fixture in test_scripts/fixtures/task_config_proposals/ pins the outcome the
pipeline must produce for one recorded response. When a proposal breaks in the
app, convert its trace into a fixture with make_fixture_from_trace.py and the
breakage becomes a permanent regression test.

Run:  python test_scripts/test_task_config_fixtures.py [--only NAME] [--list] [--verbose]
Exit code 0 = all fixtures pass, 1 = failures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "task_config_proposals"


def load_fixtures(only: str | None) -> list[dict[str, Any]]:
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        # utf-8-sig: tolerate BOMs from Windows editors that may touch fixture files
        fixture = json.loads(path.read_text(encoding="utf-8-sig"))
        fixture["_path"] = path
        if only and only.lower() not in str(fixture.get("name", path.stem)).lower():
            continue
        fixtures.append(fixture)
    return fixtures


def replay_fixture(backend, fixture: dict[str, Any]) -> dict[str, Any]:
    """Run one recorded response through the production post-LLM pipeline."""
    result: dict[str, Any] = {
        "parse": None,
        "parse_error": "",
        "pipeline": None,
        "pipeline_error": "",
        "reward_terms": [],
        "desired_tags": [],
        "avoid_tags": [],
    }
    stage = fixture.get("stage") or "reward_config"
    try:
        parsed = backend.parse_llm_response_text(fixture["raw_response_text"], stage_label=stage)
    except backend.LlmProposalError as exc:
        result["parse"] = "error"
        result["parse_error"] = str(exc.__cause__ or exc)
        return result
    result["parse"] = "recovered" if isinstance(parsed, dict) and parsed.get("_parse_recovered") else "ok"
    if not isinstance(parsed, dict):
        result["pipeline"] = "invalid"
        result["pipeline_error"] = f"parsed JSON is {type(parsed).__name__}, expected an object"
        return result

    env_name = fixture.get("env_name") or "CartPole-v1"
    if stage == "behavior_plan":
        try:
            plan = backend.normalize_behavior_tag_plan(parsed, env_name, goal=fixture.get("goal"))
            result["pipeline"] = "valid"
            result["desired_tags"] = [tag["key"] for tag in plan.get("desired_tags", [])]
            result["avoid_tags"] = [tag["key"] for tag in plan.get("avoid_tags", [])]
        except Exception as exc:
            result["pipeline"] = "invalid"
            result["pipeline_error"] = str(exc)
        return result

    # Mirrors the post-LLM handling in propose_task_config (main.py).
    proposal = backend.normalize_task_config_proposal_shape(parsed)
    try:
        backend.validate_task_config_for_run(proposal, env_name, run_id=None)
        proposal["reward_terms"] = backend.normalize_reward_terms(env_name, proposal.get("reward_terms") or [])
        result["pipeline"] = "valid"
        result["reward_terms"] = [term.get("key") for term in proposal["reward_terms"]]
    except Exception as exc:
        result["pipeline"] = "invalid"
        result["pipeline_error"] = str(exc)
    return result


def check_expectations(fixture: dict[str, Any], result: dict[str, Any]) -> list[str]:
    expect = fixture.get("expect") or {}
    failures: list[str] = []
    if not expect:
        failures.append("fixture has an empty 'expect' block; pin at least the parse outcome")
        return failures
    if "parse" in expect and result["parse"] != expect["parse"]:
        detail = result["parse_error"] or "no parse error"
        failures.append(f"parse: expected {expect['parse']!r}, got {result['parse']!r} ({detail})")
    if "parse_error_contains" in expect and expect["parse_error_contains"] not in result["parse_error"]:
        failures.append(
            f"parse_error_contains: {expect['parse_error_contains']!r} not found in {result['parse_error']!r}"
        )
    if "pipeline" in expect and result["pipeline"] != expect["pipeline"]:
        detail = result["pipeline_error"] or "no pipeline error"
        failures.append(f"pipeline: expected {expect['pipeline']!r}, got {result['pipeline']!r} ({detail})")
    if "pipeline_error_contains" in expect and expect["pipeline_error_contains"] not in result["pipeline_error"]:
        failures.append(
            f"pipeline_error_contains: {expect['pipeline_error_contains']!r} not found in {result['pipeline_error']!r}"
        )
    if "min_reward_terms" in expect and len(result["reward_terms"]) < int(expect["min_reward_terms"]):
        failures.append(
            f"min_reward_terms: expected >= {expect['min_reward_terms']}, got {len(result['reward_terms'])} "
            f"({result['reward_terms']})"
        )
    if "min_desired_tags" in expect and len(result["desired_tags"]) < int(expect["min_desired_tags"]):
        failures.append(
            f"min_desired_tags: expected >= {expect['min_desired_tags']}, got {len(result['desired_tags'])} "
            f"({result['desired_tags']})"
        )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=None, help="Run only fixtures whose name contains this substring.")
    parser.add_argument("--list", action="store_true", help="List fixture names and exit.")
    parser.add_argument("--verbose", action="store_true", help="Print full replay results for every fixture.")
    args = parser.parse_args()

    fixtures = load_fixtures(args.only)
    if not fixtures:
        print(f"No fixtures found in {FIXTURES_DIR}" + (f" matching --only {args.only!r}" if args.only else ""))
        return 1
    if args.list:
        for fixture in fixtures:
            print(f"{fixture.get('name', fixture['_path'].stem)}: {fixture.get('description', '')}")
        return 0

    print(f"Importing backend (main.py) — first import is slow because of torch/gym ...")
    import main as backend  # noqa: PLC0415 — heavy import deferred until after --list/--help

    passed = 0
    failed = 0
    for fixture in fixtures:
        name = fixture.get("name", fixture["_path"].stem)
        result = replay_fixture(backend, fixture)
        failures = check_expectations(fixture, result)
        if failures:
            failed += 1
            print(f"[FAIL] {name}")
            for failure in failures:
                print(f"       - {failure}")
        else:
            passed += 1
            print(f"[PASS] {name}")
        if args.verbose:
            printable = {key: value for key, value in result.items() if value not in (None, "", [])}
            print(f"       result: {json.dumps(printable, default=str)}")

    print()
    print(f"{passed} passed, {failed} failed out of {passed + failed} fixture(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
