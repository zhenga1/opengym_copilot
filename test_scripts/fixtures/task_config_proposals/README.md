# Task-Config Proposal Fixtures (Tier 1 regression suite)

Recorded LLM responses replayed through the **production** post-response pipeline
(`parse_llm_response_text` → `normalize_task_config_proposal_shape` →
`validate_task_config_for_run` → `normalize_reward_terms`) with no network calls
and no API cost.

## Run

```bash
python test_scripts/test_task_config_fixtures.py            # whole suite
python test_scripts/test_task_config_fixtures.py --only ternary --verbose
python test_scripts/test_task_config_fixtures.py --list
```

Exit code 0 = pass, 1 = failure. First run is slow (~10s) because importing
`main.py` pulls in torch/gym.

## Workflow: a proposal broke in the app — make it a fixture

1. Find the trace under `logs/llm_traces/<run_id>/` (look for `parse_error`,
   `response_received`, or `parse_recovered` files; request-only traces can't
   be replayed).
2. Convert it:

   ```bash
   python test_scripts/make_fixture_from_trace.py logs/llm_traces/<run_id>/<trace>.json --name my_bug_name
   ```

3. Edit the generated fixture: write a real `description` and pin the `expect`
   block to the behavior the pipeline **should** have for this response.
4. Run the suite. Commit the fixture. The breakage is now a permanent regression test.

## Fixture schema

| Field | Meaning |
|---|---|
| `name` | Unique id; also the filename. |
| `description` | Why this response matters — what broke, what's pinned. |
| `env_name` / `goal` | Context used for validation (variable set comes from `env_name`). |
| `stage` | `reward_config` (full proposal pipeline) or `behavior_plan` (tag-plan normalization). |
| `source_trace` | Path to the original trace, for archaeology. `null` for synthetic fixtures. |
| `raw_response_text` | The exact model output text being replayed. |
| `expect` | Assertions, see below. |

### `expect` assertions

| Key | Meaning |
|---|---|
| `parse` | `ok` (strict JSON), `recovered` (balanced-object rescue kicked in), or `error` (LlmProposalError → heuristic fallback in prod). |
| `parse_error_contains` | Substring that must appear in the parse error. |
| `pipeline` | `valid` (survives normalize + validate) or `invalid` (rejected → heuristic fallback in prod). |
| `pipeline_error_contains` | Substring that must appear in the validation error. |
| `min_reward_terms` | Minimum count of normalized reward terms (reward_config stage). |
| `min_desired_tags` | Minimum count of surviving desired tags (behavior_plan stage). |

Pin only what you mean: an expectation like `parse: error` documents *current*
intended behavior — if you later add truncation repair, flipping that fixture to
`recovered` is the test-driven way to build it.
