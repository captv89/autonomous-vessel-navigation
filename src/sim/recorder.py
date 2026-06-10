"""
Episode recording and playback.

Every simulation step is written as one JSON line, giving complete
visibility into what each agent saw and decided. The format is shared by
both agents so the replay viewer and the evaluation report treat classical
and RL episodes identically.

File layout (JSON Lines):
    {"type": "header", "scenario": {...}, "agent": {...}, "config": {...}}
    {"type": "step", "t": ..., "vessel": {...}, "obstacles": [...],
     "decision": {...}, "events": [...], "reward": {...}?}
    {"type": "summary", "outcome": ..., "metrics": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, IO, Iterator, List, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy types so json.dumps never chokes."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class EpisodeRecorder:
    """Writes one episode to a JSONL file (or collects in memory if path=None)."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.records: List[Dict[str, Any]] = []
        self._fh: Optional[IO] = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "w")

    def _write(self, record: Dict[str, Any]) -> None:
        record = _jsonable(record)
        self.records.append(record)
        if self._fh:
            self._fh.write(json.dumps(record) + "\n")

    def header(self, scenario: Dict[str, Any], agent: Dict[str, Any],
               config: Dict[str, Any]) -> None:
        self._write({"type": "header", "scenario": scenario,
                     "agent": agent, "config": config})

    def step(self, t: float, vessel: Dict[str, float],
             obstacles: List[Dict[str, float]], decision: Dict[str, Any],
             events: List[str], reward: Optional[Dict[str, float]] = None) -> None:
        record: Dict[str, Any] = {"type": "step", "t": round(t, 3),
                                  "vessel": vessel, "obstacles": obstacles,
                                  "decision": decision, "events": events}
        if reward is not None:
            record["reward"] = reward
        self._write(record)

    def summary(self, outcome: str, metrics: Dict[str, Any]) -> None:
        self._write({"type": "summary", "outcome": outcome, "metrics": metrics})

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_episode(path: str | Path) -> Dict[str, Any]:
    """Load a recorded episode into {header, steps, summary}."""
    header: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["type"] == "header":
                header = record
            elif record["type"] == "step":
                steps.append(record)
            elif record["type"] == "summary":
                summary = record
    if header is None:
        raise ValueError(f"{path}: missing header record")
    return {"header": header, "steps": steps, "summary": summary}
