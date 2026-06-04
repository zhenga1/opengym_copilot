from __future__ import annotations

import json
import logging
import os
import threading
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("OPEN_GYM_DATA_DIR", str(BASE_DIR))).resolve()
LOGS_DIR = (DATA_DIR / "logs").resolve()
RUN_EVENTS_LOG_PATH = (LOGS_DIR / "run_events.jsonl").resolve()
REWARD_SPECS_LOG_PATH = (LOGS_DIR / "reward_specs.jsonl").resolve()
TASK_VARIABLE_NAMES_LOG_PATH = (LOGS_DIR / "task_variable_names.jsonl").resolve()
LLM_TRACES_LOG_PATH = (LOGS_DIR / "llm_traces.jsonl").resolve()
LLM_TRACES_DIR = (LOGS_DIR / "llm_traces").resolve()
BACKEND_LOG_PATH = (LOGS_DIR / "backend.log").resolve()

_WRITE_LOCK = threading.Lock()
_CONFIGURED = False


def ensure_log_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LLM_TRACES_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_backend_logging(level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    ensure_log_dirs()
    logger = logging.getLogger("opengym")
    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(BACKEND_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    _CONFIGURED = True
    return logger


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_log_dirs()
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _safe_path_component(value: str | None, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return cleaned.strip("._") or fallback


def _write_pretty_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_log_dirs()
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            # two spaces for indentation to balance readability with file size
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def _try_parse_json_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _wrap_long_text(value: Any, width: int = 100) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if "\n" in value:
        return value
    if len(value) < width + 20:
        return value
    return textwrap.fill(value, width=width, break_long_words=False, break_on_hyphens=False)


def _prettify_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pretty = json.loads(json.dumps(payload, ensure_ascii=False))
    request_payload = pretty.get("request_payload")
    if isinstance(request_payload, dict):
        messages = request_payload.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                parsed = _try_parse_json_text(content)
                if parsed is not content:
                    message["content_json"] = parsed
                else:
                    message["content_wrapped"] = _wrap_long_text(content)
    raw_response_text = pretty.get("raw_response_text")
    parsed_raw_response = _try_parse_json_text(raw_response_text)
    if parsed_raw_response is not raw_response_text:
        pretty["raw_response_json"] = parsed_raw_response
    else:
        pretty["raw_response_wrapped"] = _wrap_long_text(raw_response_text)
    goal = pretty.get("goal")
    pretty["goal_wrapped"] = _wrap_long_text(goal, width=88)
    return pretty


def _llm_trace_pretty_path(
    *,
    run_id: str | None,
    stage: str,
    trace_type: str,
    llm_id: str | None,
    ts: str,
    attempt: int | None,
) -> Path:
    # _safe_path_component this just cleans to make sure every char is alphanumeric or - or _ or .
    run_key = _safe_path_component(run_id, "no_run_id")
    stage_key = _safe_path_component(stage, "stage")
    trace_key = _safe_path_component(trace_type, "trace")
    llm_key = _safe_path_component(llm_id, "llm")
    ts_key = _safe_path_component(ts.replace(":", "-"), "ts")
    attempt_suffix = f"__attempt_{attempt}" if attempt is not None else ""
    filename = f"{ts_key}__{stage_key}__{trace_key}__{llm_key}{attempt_suffix}.json"
    return LLM_TRACES_DIR / run_key / filename


def log_run_event(
    event_type: str,
    *,
    run_id: str | None = None,
    env_name: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ts": utc_now_iso(),
        "event_type": str(event_type),
        "run_id": run_id,
        "env_name": env_name,
        "details": details or {},
    }
    _append_jsonl(RUN_EVENTS_LOG_PATH, payload)
    return payload


def log_reward_spec_snapshot(
    *,
    run_id: str | None,
    env_name: str,
    terms: list[dict[str, Any]],
    task_config: dict[str, Any] | None = None,
    source: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ts": utc_now_iso(),
        "run_id": run_id,
        "env_name": env_name,
        "source": source,
        "terms": terms,
        "task_config": task_config or {},
        "details": details or {},
    }
    _append_jsonl(REWARD_SPECS_LOG_PATH, payload)
    return payload


def log_task_variable_names(
    *,
    run_id: str | None,
    env_name: str | None,
    variable_names: list[str],
    source: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "ts": utc_now_iso(),
        "run_id": run_id,
        "env_name": env_name,
        "source": str(source),
        "variable_count": len(variable_names),
        "variable_names": list(variable_names),
        "details": details or {},
    }
    _append_jsonl(TASK_VARIABLE_NAMES_LOG_PATH, payload)
    return payload


def log_llm_trace(
    *,
    run_id: str | None,
    env_name: str | None,
    stage: str,
    trace_type: str,
    llm_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    goal: str | None = None,
    attempt: int | None = None,
    request_payload: dict[str, Any] | None = None,
    raw_response_text: str | None = None,
    parsed_response: dict[str, Any] | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = utc_now_iso()
    payload = {
        "ts": ts,
        "run_id": run_id,
        "env_name": env_name,
        "stage": str(stage),
        "trace_type": str(trace_type),
        "llm_id": llm_id,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "goal": goal,
        "attempt": attempt,
        "request_payload": request_payload or {},
        "raw_response_text": raw_response_text,
        "parsed_response": parsed_response or {},
        "error": error,
        "details": details or {},
    }
    # Trace out pretty path = make a valid path from the inputs
    pretty_path = _llm_trace_pretty_path(
        run_id=run_id,
        stage=stage,
        trace_type=trace_type,
        llm_id=llm_id,
        ts=ts,
        attempt=attempt,
    )
    payload["pretty_path"] = str(pretty_path)
    # write the raw payload into the LLM_TRACES_LOG_PATH
    _append_jsonl(LLM_TRACES_LOG_PATH, payload)
    # write the prettified payload into the pretty path
    # Prettified = payload made more human readable
    _write_pretty_json(pretty_path, _prettify_llm_payload(payload))
    return payload
