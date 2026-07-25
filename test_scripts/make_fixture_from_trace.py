"""Turn a recorded llm_traces file into a replayable proposal fixture.

Workflow: when a proposal breaks, find the trace under logs/llm_traces/<run_id>/,
then run:

    python test_scripts/make_fixture_from_trace.py logs/llm_traces/<run_id>/<trace>.json

The fixture lands in test_scripts/fixtures/task_config_proposals/ and is picked up
by test_scripts/test_task_config_fixtures.py. Edit the generated "expect" block to
pin the behavior you want the pipeline to have for this response.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "fixtures" / "task_config_proposals"

TRACE_TYPE_TO_EXPECTED_PARSE = {
    "parse_error": "error",
    "parse_recovered": "recovered",
    "response_received": "ok",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "fixture"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_path", help="Path to a logs/llm_traces/.../*.json trace file.")
    parser.add_argument("--name", default=None, help="Fixture name (default: derived from stage + trace type + model).")
    parser.add_argument("--expect-parse", choices=["ok", "recovered", "error"], default=None,
                        help="Expected parse outcome (default: inferred from the trace_type).")
    parser.add_argument("--expect-pipeline", choices=["valid", "invalid"], default=None,
                        help="Expected normalize/validate outcome (omit to leave unpinned).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Fixture output directory.")
    args = parser.parse_args()

    trace_path = Path(args.trace_path).resolve()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))

    raw_response_text = trace.get("raw_response_text")
    if not raw_response_text:
        print("Trace has no raw_response_text; request-only traces (request_prepared/request_sent) "
              "cannot be replayed. Pick the matching *response* or *parse_error* trace instead.")
        return 1

    stage = trace.get("stage") or "reward_config"
    trace_type = trace.get("trace_type") or "response_received"
    model = trace.get("model") or "unknown-model"
    name = args.name or slugify(f"{stage}_{trace_type}_{model}_{trace_path.stem.split('_')[0]}")

    expect: dict[str, object] = {}
    expected_parse = args.expect_parse or TRACE_TYPE_TO_EXPECTED_PARSE.get(trace_type)
    if expected_parse:
        expect["parse"] = expected_parse
    if args.expect_pipeline:
        expect["pipeline"] = args.expect_pipeline

    try:
        source_trace = str(trace_path.relative_to(REPO_ROOT))
    except ValueError:
        source_trace = str(trace_path)

    fixture = {
        "name": name,
        "description": f"Recorded {trace_type} from {model} ({stage} stage). Edit me: say why this response matters.",
        "env_name": trace.get("env_name") or "CartPole-v1",
        "goal": trace.get("goal") or "",
        "stage": stage,
        "source_trace": source_trace.replace("\\", "/"),
        "source_model": model,
        "source_error": trace.get("error"),
        "raw_response_text": raw_response_text,
        "expect": expect,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    if out_path.exists():
        print(f"Refusing to overwrite existing fixture: {out_path}")
        return 1
    out_path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Expected parse outcome: {expect.get('parse', '(unpinned)')} — review the 'expect' block before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
