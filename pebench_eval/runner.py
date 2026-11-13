from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from pathlib import Path
from typing import Any

from .candidates import load_candidate_rows, validate_candidates
from .dataset import (Case, canonical_json, case_fingerprint, dataset_fingerprint, load_cases,
                      sha256_bytes, validate_assets)
from .evidence import evidence_cache_key, validate_evidence
from .prompts import validate_dimension_result
from .report import render_markdown, summarize
from .schema import DIMENSIONS, RUBRIC_VERSION
from .storage import case_filename, read_json, write_json


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _safe(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean[:100] or "unnamed"


def select_cases(cases: list[Case], modes: set[str] | None, limit: int | None) -> list[Case]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    selected = [c for c in cases if not modes or c.mode in modes]
    return selected if limit is None else selected[:limit]


def _load_or_extract_evidence(run_dir: Path, case: Case, sample: dict[str, Any], manifest: dict[str, dict[str, Any]], provider: Any) -> dict[str, Any]:
    evidence_provider = provider.evidence_provider(case) if hasattr(provider, "evidence_provider") else provider
    provider_identity = getattr(evidence_provider, "identity", f"{evidence_provider.name}:{evidence_provider.version}")
    key = evidence_cache_key(case, manifest, provider_identity, evidence_provider.version)
    path = run_dir / "evidence" / case_filename(case.key)
    if path.is_file():
        cached = read_json(path)
        if cached.get("cache_key") == key:
            value = cached["evidence"]
            validate_evidence(value, case, manifest)
            return value
    value = provider.extract_evidence(case, sample, manifest)
    validate_evidence(value, case, manifest)
    write_json(path, {"case_key": case.key, "cache_key": key, "created_at": _now(),
                      "provider": evidence_provider.name, "provider_version": evidence_provider.version,
                      "raw_model_response": getattr(evidence_provider, "last_raw_response", ""),
                      "usage": getattr(evidence_provider, "last_usage", None), "evidence": value})
    return value


def _load_judgment(path: Path, *, case_key: str, method: str, dimension: str,
                   candidate_hash: str, evidence_key: str, judge_key: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = read_json(path)
        if (value.get("case_key") == case_key and value.get("method") == method and value.get("dimension") == dimension
                and value.get("candidate_hash") == candidate_hash and value.get("evidence_cache_key") == evidence_key
                and value.get("judge_cache_key") == judge_key):
            return value
    except (OSError, json.JSONDecodeError):
        return None
    return None


def run_pipeline(*, data_root: str | Path, output_root: str | Path, provider: Any,
                 candidate_file: str | Path | None = None, modes: set[str] | None = None,
                 limit: int | None = None, run_id: str | None = None, resume: bool = True,
                 inspect_content: bool = True, case_filter_keys: set[str] | None = None) -> Path:
    root = Path(data_root).resolve()
    all_cases = load_cases(root)
    eligible_cases = [case for case in all_cases if case_filter_keys is None or case.key in case_filter_keys]
    selected_cases = select_cases(eligible_cases, modes, limit)
    if not selected_cases:
        raise ValueError("no cases selected")
    candidate_rows = load_candidate_rows(candidate_file) if candidate_file else []
    validate_candidates(candidate_rows, all_cases)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        if modes and row["mode"] not in modes:
            continue
        by_case.setdefault(f"{row['mode']}:{row['capid']}", []).append(row)
    if candidate_file is not None:
        candidate_case_keys = set(by_case)
        selected_cases = [case for case in selected_cases if case.key in candidate_case_keys]
        if not selected_cases:
            raise ValueError("candidate file contains no cases in the selected scope")
    if not selected_cases:
        raise ValueError("no cases selected")
    selected_keys = {case.key for case in selected_cases}
    by_case = {key: rows for key, rows in by_case.items() if key in selected_keys}
    candidate_identity = [
        {"mode": row["mode"], "capid": row["capid"], "method": row["method"],
         "candidate_hash": row["candidate_hash"]}
        for key in sorted(by_case) for row in by_case[key]
    ]
    candidate_fingerprint = (sha256_bytes(canonical_json(candidate_identity).encode("utf-8"))
                             if candidate_file is not None else None)
    run_id = run_id or (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore and hyphen")
    output_root_path = Path(output_root).expanduser().resolve()
    data_disk_root = Path("/pfs/pfs-spMCeu").resolve()
    if not data_disk_root.is_dir() or not output_root_path.is_relative_to(data_disk_root):
        raise ValueError(f"output_root must be on the configured data disk under {data_disk_root}")
    run_dir = output_root_path / run_id
    if run_dir.parent != output_root_path:
        raise ValueError("run directory must stay inside output_root")
    run_json = run_dir / "run.json"
    previous = read_json(run_json) if run_json.is_file() else None
    saved_manifest = read_json(run_dir / "cases.json") if previous and (run_dir / "cases.json").is_file() else None
    if previous and (not saved_manifest or [c.get("case_key") for c in saved_manifest.get("cases", [])] != [c.key for c in selected_cases]):
        raise ValueError("existing run has a different case scope; use a new run_id")
    manifest = saved_manifest.get("asset_manifest", {}) if saved_manifest else {}
    required_assets = {asset.relative_path for case in selected_cases for asset in case.assets}
    manifest_current = bool(saved_manifest) and required_assets.issubset(manifest) and all(
        (root / info["path"]).is_file()
        and (root / info["path"]).stat().st_size == info.get("size")
        and (root / info["path"]).stat().st_mtime_ns == info.get("mtime_ns")
        for info in manifest.values())
    if not manifest_current:
        manifest = validate_assets(selected_cases, inspect_content=inspect_content)
    dataset_fp = dataset_fingerprint(root, manifest)
    if previous and (previous.get("dataset_fingerprint") != dataset_fp
                     or previous.get("selected_modes") != sorted(modes or {"t2v", "i2v", "r2v"})
                     or previous.get("limit") != limit):
        raise ValueError("existing run has a different dataset fingerprint or case scope; use a new run_id")
    old_candidate_fingerprint = previous.get("candidate_fingerprint") if previous else None
    if old_candidate_fingerprint is not None and old_candidate_fingerprint != candidate_fingerprint:
        raise ValueError("candidate content changed for this run_id; use a new run_id to avoid mixing PE versions")
    meta = {
        "run_id": run_id, "status": "running", "started_at": previous.get("started_at", _now()) if previous else _now(),
        "data_root": str(root), "output_root": str(Path(output_root).resolve()),
        "dataset_fingerprint": dataset_fp, "case_count": len(selected_cases),
        "all_case_count": len(all_cases), "selected_modes": sorted(modes or {"t2v", "i2v", "r2v"}),
        "limit": limit, "asset_validation": "content" if inspect_content else "path-and-hash",
        "provider": provider.metadata() if hasattr(provider, "metadata") else
                    {"name": provider.name, "version": provider.version, "mock": bool(getattr(provider, "is_mock", False))},
        "rubric_version": RUBRIC_VERSION, "candidate_file": str(candidate_file) if candidate_file else None,
        "candidate_count": sum(len(v) for v in by_case.values()),
        "candidate_fingerprint": candidate_fingerprint,
    }
    write_json(run_json, meta)
    write_json(run_dir / "cases.json", {"cases": [c.to_dict() | {"case_fingerprint": case_fingerprint(c, manifest)} for c in selected_cases],
                                         "asset_manifest": manifest})
    for rows in by_case.values():
        for candidate in rows:
            write_json(run_dir / "candidates" / _safe(candidate["method"]) /
                       case_filename(f"{candidate['mode']}:{candidate['capid']}"), candidate)
    for case in selected_cases:
        sample = case.to_dict()
        try:
            evidence_provider = provider.evidence_provider(case) if hasattr(provider, "evidence_provider") else provider
            evidence_identity = getattr(evidence_provider, "identity", f"{evidence_provider.name}:{evidence_provider.version}")
            evidence_key = evidence_cache_key(case, manifest, evidence_identity, evidence_provider.version)
            evidence = _load_or_extract_evidence(run_dir, case, sample, manifest, provider)
            (run_dir / "evidence" / (case_filename(case.key) + ".error.json")).unlink(missing_ok=True)
        except Exception as exc:
            evidence_path = run_dir / "evidence" / (case_filename(case.key) + ".error.json")
            write_json(evidence_path, {"case_key": case.key, "stage": "evidence",
                                       "error_type": type(exc).__name__, "error": str(exc), "created_at": _now()})
            continue
        judge_provider = provider.judge_provider() if hasattr(provider, "judge_provider") else provider
        import hashlib
        judge_identity = getattr(judge_provider, "identity", f"{judge_provider.name}:{judge_provider.version}")
        for candidate in by_case.get(case.key, []):
            method = candidate["method"]
            method_dir = _safe(method)
            for dimension in DIMENSIONS:
                judge_key = hashlib.sha256(f"{judge_identity}:{RUBRIC_VERSION}:{dimension}".encode()).hexdigest()
                target = run_dir / "judgments" / case_filename(case.key) / method_dir / f"{dimension}.json"
                error_path = target.with_suffix(".error.json")
                if resume and _load_judgment(target, case_key=case.key, method=method, dimension=dimension,
                                             candidate_hash=candidate["candidate_hash"], evidence_key=evidence_key, judge_key=judge_key):
                    continue
                try:
                    value = provider.judge(sample, candidate, evidence, dimension)
                    value = validate_dimension_result(value, dimension)
                    raw_response = getattr(judge_provider, "last_raw_response", "")
                    value.update({"case_key": case.key, "method": method, "dimension": dimension,
                                  "candidate_hash": candidate["candidate_hash"], "evidence_cache_key": evidence_key,
                                  "judge_cache_key": judge_key, "raw_model_response": raw_response,
                                  "usage": getattr(judge_provider, "last_usage", None), "created_at": _now()})
                    write_json(target, value)
                    error_path.unlink(missing_ok=True)
                except Exception as exc:
                    write_json(error_path, {"case_key": case.key, "method": method, "dimension": dimension,
                                            "stage": "judge", "error_type": type(exc).__name__,
                                            "error": str(exc), "created_at": _now()})
    summary = summarize(run_dir)
    error_count = len(list(run_dir.rglob("*.error.json")))
    method_count = len({row["method"] for rows in by_case.values() for row in rows})
    expected_candidates = method_count * len(selected_cases)
    meta["candidate_coverage"] = {"actual": meta["candidate_count"], "expected_for_methods": expected_candidates}
    if not meta["candidate_count"]:
        meta["status"] = "prepared_with_errors" if error_count else "prepared"
    elif error_count:
        meta["status"] = "completed_with_errors"
    elif meta["candidate_count"] < expected_candidates:
        meta["status"] = "completed_partial"
    else:
        meta["status"] = "completed"
    meta["error_count"] = error_count
    meta["completed_at"] = _now()
    meta["summary_path"] = str(run_dir / "summary.json")
    write_json(run_json, meta)
    (run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return run_dir
