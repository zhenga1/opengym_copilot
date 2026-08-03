"""Offline tests for the user-defined LLM provider registry.

Redirects the providers file to a temp path, then exercises upsert / catalog
merge / settings resolution / key stripping / delete with no network calls.

Run:  python test_scripts/test_llm_provider_registry.py
Exit code 0 = all checks pass, 1 = failures.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECRET = "sk-test-secret-9137"


def main() -> int:
    print("Importing backend (main.py) — first import is slow because of torch/gym ...")
    import main as backend
    from fastapi import HTTPException

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str) -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {name}")
        if not condition:
            failures.append(f"{name}: {detail}")

    original_file = backend.LLM_PROVIDERS_FILE
    tmp_dir = Path(tempfile.mkdtemp(prefix="llm_providers_test_"))
    backend.LLM_PROVIDERS_FILE = tmp_dir / "llm_providers.json"
    try:
        # upsert creates a provider and it appears in the catalog, available
        stored = backend.upsert_user_llm_provider({
            "label": "My Anthropic",
            "preset": "anthropic",
            "base_url": "https://api.anthropic.com/v1/",
            "model": "claude-sonnet-5",
            "api_key": SECRET,
        })
        check("upsert_returns_slug_id", stored["id"] == "my-anthropic", f"got {stored['id']}")
        catalog = backend._task_config_llm_catalog()
        entry = next((item for item in catalog if item["id"] == "my-anthropic"), None)
        check("catalog_contains_user_provider", entry is not None, f"ids: {[i['id'] for i in catalog]}")
        check("catalog_entry_available", bool(entry and entry["available"]), f"entry: {entry}")
        check("base_url_trailing_slash_stripped", bool(entry and entry["base_url"] == "https://api.anthropic.com/v1"),
              f"got {entry and entry['base_url']}")
        check("catalog_entry_flagged_user_defined", bool(entry and entry.get("user_defined")), f"entry: {entry}")

        # settings resolution by id picks up the stored key/endpoint
        settings = backend._task_config_llm_settings("my-anthropic")
        check("settings_resolve_user_provider",
              settings["base_url"] == "https://api.anthropic.com/v1" and settings["api_key"] == SECRET
              and settings["model"] == "claude-sonnet-5",
              f"settings: { {k: v for k, v in settings.items() if k != 'api_key'} }")

        # public endpoints must never leak the key
        listing = backend.get_task_config_llms()
        check("task_config_llms_strips_key", SECRET not in json.dumps(listing), "secret found in /task_config_llms payload")
        upsert_response = backend.upsert_llm_provider_endpoint(backend.LlmProviderUpsertRequest(
            label="My Anthropic", base_url="https://api.anthropic.com/v1", model="claude-sonnet-5",
        ))
        check("upsert_endpoint_strips_key", SECRET not in json.dumps(upsert_response), "secret found in upsert response")

        # update without api_key keeps the stored key
        settings = backend._task_config_llm_settings("my-anthropic")
        check("update_without_key_keeps_key", settings["api_key"] == SECRET, "stored key lost on keyless update")

        # reserved builtin ids are rejected
        try:
            backend.upsert_user_llm_provider({"id": "openai", "label": "openai", "base_url": "https://x", "model": "m", "api_key": "k"})
            check("builtin_id_rejected", False, "no error raised")
        except ValueError:
            check("builtin_id_rejected", True, "")

        # persistence round-trip: file on disk holds the provider
        on_disk = json.loads(backend.LLM_PROVIDERS_FILE.read_text(encoding="utf-8"))
        check("persisted_to_disk", any(item["id"] == "my-anthropic" for item in on_disk["providers"]),
              f"file: {on_disk}")

        # delete removes it from catalog; deleting again 404s at the endpoint
        check("delete_returns_true", backend.delete_user_llm_provider("my-anthropic"), "delete returned False")
        catalog = backend._task_config_llm_catalog()
        check("deleted_from_catalog", all(item["id"] != "my-anthropic" for item in catalog), "still in catalog")
        try:
            backend.delete_llm_provider_endpoint("my-anthropic")
            check("delete_endpoint_404s_when_missing", False, "no HTTPException raised")
        except HTTPException as exc:
            check("delete_endpoint_404s_when_missing", exc.status_code == 404, f"status {exc.status_code}")

        # builtin providers cannot be deleted via the endpoint
        try:
            backend.delete_llm_provider_endpoint("glm")
            check("builtin_delete_rejected", False, "no HTTPException raised")
        except HTTPException as exc:
            check("builtin_delete_rejected", exc.status_code == 400, f"status {exc.status_code}")
    finally:
        backend.LLM_PROVIDERS_FILE = original_file

    print()
    if failures:
        for failure in failures:
            print(f"  - {failure}")
    print(f"{'FAILED' if failures else 'OK'}: {len(failures)} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
