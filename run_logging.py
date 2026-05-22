from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("OPEN_GYM_DATA_DIR", str(BASE_DIR))).resolve()
LOGS_DIR = (DATA_DIR / "logs").resolve()
RUN_EVENTS_LOG_PATH = (LOGS_DIR / "run_events.jsonl").resolve()
REWARD_SPECS_LOG_PATH = (LOGS_DIR / "reward_specs.jsonl").resolve()
BACKEND_LOG_PATH = (LOGS_DIR / "backend.log").resolve()

_WRITE_LOCK = threading.Lock()
_CONFIGURED = False


def ensure_log_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


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
