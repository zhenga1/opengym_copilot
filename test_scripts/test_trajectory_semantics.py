"""Tier-3 semantic tests: does a proposed reward actually reward the requested behavior?

Validation (Tier 1) proves an expression *runs*; a live proposal (Tier 2) proves
the LLM produced structurally plausible terms. This tier closes the loop: it
builds synthetic CartPole trajectories — one that genuinely alternates sides
with stable holds, one parked on the left, one parked at center — scores each
with the proposed reward via trajectory_reward_eval (the exact production
reward semantics, no gym env, no network), and asserts the ordering the goal
demands:

    mean_reward(alternating) > mean_reward(one_sided)
    mean_reward(alternating) > mean_reward(stationary)

If the LLM's reward config scores a parked cart as high as an alternating one,
training was never going to produce alternation — this catches that in
milliseconds instead of a failed training run.

Proposal sources (first match wins):
  --proposal FILE       any JSON with reward_terms/task_params/derived_signals
  --from-live NAME      latest test_scripts/live_results artifact for a Tier-2 spec
  (default)             test_scripts/sample_proposals/left_then_right_stable.json

Run:  python test_scripts/test_trajectory_semantics.py [--verbose]
Exit code 0 = ordering holds, 1 = violated.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_PROPOSAL = Path(__file__).resolve().parent / "sample_proposals" / "left_then_right_stable.json"
LIVE_RESULTS_DIR = Path(__file__).resolve().parent / "live_results"

ENV_NAME = "CartPole-v1"
DT = 0.02          # CartPole physics timestep
STEPS = 400        # 8 seconds
AMPLITUDE = 0.8    # target |cart_position| at each side
PERIOD_SEC = 4.0   # full left-right cycle


def _trajectory_from_positions(positions: list[float]) -> list[dict]:
    trajectory = []
    prev_x = positions[0]
    for index, x in enumerate(positions):
        velocity = (x - prev_x) / DT if index > 0 else 0.0
        # pole stays near upright; tiny angle leaning into the motion
        pole_angle = 0.01 * math.tanh(velocity)
        trajectory.append({
            "obs": [x, velocity, pole_angle, 0.0],
            "action": 1.0 if velocity >= 0 else 0.0,
            "native_reward": 1.0,
        })
        prev_x = x
    return trajectory


def make_alternating_trajectory() -> list[dict]:
    """Holds near -A, swings to +A, repeats — the behavior the goal asks for."""
    positions = [
        AMPLITUDE * math.tanh(4 * math.sin(2 * math.pi * (index * DT) / PERIOD_SEC))
        for index in range(1, STEPS + 1)
    ]
    return _trajectory_from_positions(positions)


def make_one_sided_trajectory() -> list[dict]:
    """Parked on the left the whole episode — the classic failure mode."""
    return _trajectory_from_positions([-AMPLITUDE] * STEPS)


def make_stationary_trajectory() -> list[dict]:
    """Parked at center — balancing without any side motion."""
    return _trajectory_from_positions([0.0] * STEPS)


def load_proposal(args) -> tuple[str, dict]:
    if args.proposal:
        path = Path(args.proposal)
        return str(path), json.loads(path.read_text(encoding="utf-8-sig"))
    if args.from_live:
        candidates = sorted(LIVE_RESULTS_DIR.glob(f"*_{args.from_live}*.json"))
        if not candidates:
            raise SystemExit(f"No live_results artifact matching {args.from_live!r} in {LIVE_RESULTS_DIR}")
        artifact = json.loads(candidates[-1].read_text(encoding="utf-8-sig"))
        return str(candidates[-1]), artifact.get("proposal") or artifact
    return str(SAMPLE_PROPOSAL), json.loads(SAMPLE_PROPOSAL.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proposal", default=None, help="Path to a proposal/reward-config JSON to score.")
    parser.add_argument("--from-live", default=None, help="Score the latest Tier-2 live artifact for this spec name.")
    parser.add_argument("--verbose", action="store_true", help="Print per-term totals for every trajectory.")
    args = parser.parse_args()

    from trajectory_reward_eval import evaluate_proposal_on_trajectory

    source, proposal = load_proposal(args)
    print(f"Scoring proposal from: {source}")

    scores = {}
    for label, trajectory in (
        ("alternating", make_alternating_trajectory()),
        ("one_sided", make_one_sided_trajectory()),
        ("stationary", make_stationary_trajectory()),
    ):
        result = evaluate_proposal_on_trajectory(ENV_NAME, proposal, trajectory, step_duration_sec=DT)
        scores[label] = result
        print(f"  {label:12s} mean/step = {result['mean_per_step']:+.4f}  (total {result['total']:+.1f} over {result['steps']} steps)")
        if args.verbose:
            for key, value in sorted(result["per_term_totals"].items(), key=lambda item: item[1]):
                print(f"               {key:28s} {value:+10.2f}")

    failures = []
    margin_one_sided = scores["alternating"]["mean_per_step"] - scores["one_sided"]["mean_per_step"]
    margin_stationary = scores["alternating"]["mean_per_step"] - scores["stationary"]["mean_per_step"]
    if margin_one_sided <= 0:
        failures.append(
            f"alternating does NOT out-score one_sided (margin {margin_one_sided:+.4f}/step) — "
            "this reward would happily train a cart that parks on one side"
        )
    if margin_stationary <= 0:
        failures.append(
            f"alternating does NOT out-score stationary (margin {margin_stationary:+.4f}/step) — "
            "this reward would happily train a cart that never moves"
        )

    print()
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
    else:
        print(f"[PASS] alternating beats one_sided by {margin_one_sided:+.4f}/step "
              f"and stationary by {margin_stationary:+.4f}/step")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
