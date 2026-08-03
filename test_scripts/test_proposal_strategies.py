"""Offline tests for the proposal query strategies in propose_task_config.

Stubs call_llm_task_config_proposal (no network, no API cost) and verifies the
two strategies:

  - "fallback": one LLM attempt; heuristic fallback on the first failure
  - "retry":    re-query the LLM after a failed call or non-runnable proposal,
                up to the attempt cap, before falling back to the heuristic

Run:  python test_scripts/test_proposal_strategies.py
Exit code 0 = all scenarios pass, 1 = failures.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_NAME = "CartPole-v1"


def make_valid_proposal() -> dict[str, Any]:
    return {
        "goal": "stub goal",
        "task_params": [],
        "derived_signals": [],
        "reward_terms": [
            {"key": "native", "label": "Native Reward", "description": "Native env reward",
             "weight": 1.0, "enabled": True, "expression": "native_reward"},
            {"key": "sway_bonus", "label": "Sway Bonus", "description": "Reward sway",
             "weight": 0.5, "enabled": True, "expression": "-square(pole_angle)"},
        ],
        "success_metric": "stub metric",
        "rationale": "stub rationale",
        "warnings": [],
        "provider": "stub",
        "model": "stub-model",
    }


def make_invalid_proposal() -> dict[str, Any]:
    proposal = make_valid_proposal()
    proposal["reward_terms"][1]["expression"] = "cart_side_change"
    return proposal


class StubLlm:
    """Returns/raises the configured outcome per call; repeats the last outcome."""

    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, goal, env_name, *, run_id=None, llm_id=None):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome()
        return outcome


def main() -> int:
    print("Importing backend (main.py) — first import is slow because of torch/gym ...")
    import main as backend

    original = backend.call_llm_task_config_proposal
    failures: list[str] = []
    try:
        def scenario_runner():
            results = []

            def with_stub(stub, **kwargs):
                backend.call_llm_task_config_proposal = stub
                return backend.propose_task_config("stub goal", ENV_NAME, **kwargs), stub

            def check(name, condition, detail):
                if not condition:
                    failures.append(f"{name}: {detail}")

            def call_error():
                return backend.LlmProposalError(
                    "reward_config returned malformed JSON: stub",
                    stage_label="reward_config",
                    raw_response_text="{broken",
                )

            # 1. retry mode recovers after a failed LLM call
            proposal, stub = with_stub(StubLlm([call_error(), make_valid_proposal]),
                                       run_id="strategy-test-1", strategy="retry")
            results.append("retry_recovers_after_call_failure")
            check(results[-1], proposal.get("source_type") == "llm",
                  f"expected source_type llm, got {proposal.get('source_type')} (warnings={proposal.get('warnings')})")
            check(results[-1], proposal.get("proposal_attempts") == 2,
                  f"expected 2 attempts, got {proposal.get('proposal_attempts')}")
            check(results[-1], stub.calls == 2, f"expected 2 LLM calls, got {stub.calls}")
            check(results[-1], any("Attempt 1 failed during llm_call" in w for w in proposal.get("warnings", [])),
                  f"expected attempt-1 warning, got {proposal.get('warnings')}")
            check(results[-1], any(term.get("key") == "sway_bonus" for term in proposal.get("reward_terms", [])),
                  "expected stub custom term sway_bonus to survive normalization")

            # 2. fallback mode stops at the first failed call
            proposal, stub = with_stub(StubLlm([call_error(), make_valid_proposal]),
                                       run_id="strategy-test-2", strategy="fallback")
            results.append("fallback_stops_after_first_call_failure")
            check(results[-1], proposal.get("source_type") == "heuristic",
                  f"expected heuristic, got {proposal.get('source_type')}")
            check(results[-1], stub.calls == 1, f"expected 1 LLM call, got {stub.calls}")
            check(results[-1], any("heuristic fallback" in w for w in proposal.get("warnings", [])),
                  f"expected fallback warning, got {proposal.get('warnings')}")

            # 3. retry mode recovers after a non-runnable proposal
            proposal, stub = with_stub(StubLlm([make_invalid_proposal, make_valid_proposal]),
                                       run_id="strategy-test-3", strategy="retry")
            results.append("retry_recovers_after_validation_failure")
            check(results[-1], proposal.get("source_type") == "llm",
                  f"expected llm, got {proposal.get('source_type')} (warnings={proposal.get('warnings')})")
            check(results[-1], proposal.get("proposal_attempts") == 2,
                  f"expected 2 attempts, got {proposal.get('proposal_attempts')}")
            check(results[-1], any("Attempt 1 failed during validation" in w for w in proposal.get("warnings", [])),
                  f"expected validation warning, got {proposal.get('warnings')}")

            # 4. retry mode exhausts all attempts, then falls back
            proposal, stub = with_stub(StubLlm([make_invalid_proposal]),
                                       run_id="strategy-test-4", strategy="retry", max_attempts=3)
            results.append("retry_exhausts_to_heuristic")
            check(results[-1], proposal.get("source_type") == "heuristic",
                  f"expected heuristic, got {proposal.get('source_type')}")
            check(results[-1], proposal.get("proposal_attempts") == 3,
                  f"expected 3 attempts, got {proposal.get('proposal_attempts')}")
            check(results[-1], stub.calls == 3, f"expected 3 LLM calls, got {stub.calls}")
            status = backend.task_config_request_status_by_run.get("strategy-test-4", {})
            check(results[-1], status.get("status") == "fallback_validation",
                  f"expected fallback_validation status, got {status.get('status')}")
            check(results[-1], "model_proposal_preview" in proposal,
                  "expected model_proposal_preview so the UI can show the rejected proposal")

            # 5. fallback mode ignores max_proposal_attempts
            proposal, stub = with_stub(StubLlm([call_error()]),
                                       run_id="strategy-test-5", strategy="fallback", max_attempts=5)
            results.append("fallback_ignores_max_attempts")
            check(results[-1], stub.calls == 1, f"expected 1 LLM call, got {stub.calls}")
            check(results[-1], proposal.get("proposal_attempts") == 1,
                  f"expected 1 attempt recorded, got {proposal.get('proposal_attempts')}")

            return results

        scenario_names = scenario_runner()
    finally:
        backend.call_llm_task_config_proposal = original

    failed_names = {failure.split(":", 1)[0] for failure in failures}
    for name in scenario_names:
        print(f"[{'FAIL' if name in failed_names else 'PASS'}] {name}")
    for failure in failures:
        print(f"       - {failure}")
    print()
    print(f"{len(scenario_names) - len(failed_names)} passed, {len(failed_names)} failed "
          f"out of {len(scenario_names)} scenario(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
