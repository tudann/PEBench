from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .dataset import Case, canonical_json, sha256_bytes


def load_candidate_rows(path: str | Path) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"candidate JSONL line {line_number}: {exc}") from None
        if not isinstance(row, dict):
            raise ValueError(f"candidate JSONL line {line_number}: expected object")
        if set(row) != {"mode", "capid", "method", "pe_prompt"}:
            raise ValueError(f"candidate JSONL line {line_number}: required keys are mode, capid, method, pe_prompt")
        mode, capid, method, prompt = row["mode"], str(row["capid"]).strip(), row["method"], row["pe_prompt"]
        if mode not in {"t2v", "i2v", "r2v"} or not capid:
            raise ValueError(f"candidate JSONL line {line_number}: invalid mode or capid")
        if not isinstance(method, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", method.strip()):
            raise ValueError(f"candidate JSONL line {line_number}: method must use only letters, digits, dot, underscore or hyphen")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 200_000:
            raise ValueError(f"candidate JSONL line {line_number}: pe_prompt must be non-empty text <= 200000 chars")
        key = (mode, capid, method.strip())
        if key in seen:
            raise ValueError(f"candidate JSONL line {line_number}: duplicate (mode, capid, method): {key}")
        seen.add(key)
        result.append({"mode": mode, "capid": capid, "method": method.strip(),
                       "pe_prompt": prompt, "candidate_hash": sha256_bytes(prompt.encode("utf-8"))})
    return result


def validate_candidates(rows: list[dict[str, Any]], cases: list[Case]) -> None:
    case_keys = {case.key for case in cases}
    for row in rows:
        key = f"{row['mode']}:{row['capid']}"
        if key not in case_keys:
            raise ValueError(f"candidate references unknown case: {key}")


def template_rows(cases: list[Case], method: str = "method-name") -> str:
    return "".join(canonical_json({"mode": c.mode, "capid": c.capid,
                                   "method": method, "pe_prompt": ""}) + "\n" for c in cases)
