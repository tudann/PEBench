"""Run paired text evaluations without loading generated videos or changing old runs."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

from .candidates import load_candidate_rows, validate_candidates
from .dataset import canonical_json, dataset_fingerprint, load_cases, sha256_bytes
from .schema import DIMENSIONS
from .storage import case_filename, read_json, write_json
from .text_rubric import VERSION, SCORING, overall_score, preference, protocol_fingerprint, score_assessment, validate_pair, validate_plan


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _hash(value):
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _response(value):
    value = deepcopy(value)
    raw = value.pop("_pebench_raw_model_response", "")
    usage = value.pop("_pebench_usage", None)
    attempts = value.pop("_pebench_attempts", [])
    return value, raw, usage, attempts


def _cached_request(path, cache_key, request, validator, reuse=True):
    if reuse and path.is_file():
        cached = read_json(path)
        if cached.get("cache_key") == cache_key:
            return validator(cached["result"])
    pending = path.with_suffix(".pending.json")
    draft = None
    if reuse and pending.is_file():
        saved = read_json(pending)
        if saved.get("cache_key") == cache_key: draft = saved["draft"]
    try:
        value, raw, usage, attempts = _response(request(draft))
        value = validator(value)
    except Exception as exc:
        if getattr(exc, "partial_response", None) is not None:
            write_json(pending, {"cache_key": cache_key, "draft": exc.partial_response,
                                 "repair_targets": exc.repair_targets, "attempts": getattr(exc, "attempts", []),
                                 "created_at": _now()})
        raise
    write_json(path, {"cache_key": cache_key, "created_at": _now(), "result": value,
                      "raw_model_response": raw, "usage": usage, "attempts": attempts})
    pending.unlink(missing_ok=True)
    return value


def _sf_batches(path, key, sample, candidates, plan, judge, resume, draft=None):
    """Bound SF output size while preserving per-requirement checkpoints and evidence."""
    combined = {"dimension": "SF", "summary": "SF requirements assessed in bounded batches", "candidates": {l: {"requirements": []} for l in ("A", "B")}}
    for start in range(0, len(plan["requirements"]), 8):
        subset = deepcopy(plan)
        subset["requirements"] = plan["requirements"][start:start + 8]
        subset["_all_requirement_ids"] = [r["id"] for r in plan["requirements"]]
        ids = {r["id"] for r in subset["requirements"]}
        part = path.parent / (path.stem + '.parts') / f'{start // 8:03d}.json'
        part_key = _hash([key, "SF-batch-8", subset["requirements"]])
        pending = part.with_suffix('.pending.json')
        if resume and draft is not None and not part.is_file() and not pending.is_file():
            seed = deepcopy(draft)
            for label in ("A", "B"):
                seed["candidates"][label]["requirements"] = [r for r in seed["candidates"][label]["requirements"] if r.get("id") in ids]
            write_json(pending, {"cache_key": part_key, "draft": seed, "created_at": _now()})
        value = _cached_request(part, part_key,
                    lambda seed: judge.judge_text_pair(sample, candidates, subset, "SF", draft=seed) if seed is not None else judge.judge_text_pair(sample, candidates, subset, "SF"),
                    lambda v: validate_pair(v, "SF", candidates, subset), reuse=resume)
        for label in ("A", "B"): combined["candidates"][label]["requirements"].extend(value["candidates"][label]["requirements"])
    return validate_pair(combined, "SF", candidates, plan)


def _round(run_dir, sample, candidates, plan, judge, key, index, resume):
    # Method names are stored for audit, never included in model payloads.
    result = []
    for reverse in (False, True):
        ordered = list(reversed(candidates)) if reverse else candidates
        dimensions = {c["method"]: {} for c in candidates}
        assessments = {c["method"]: {} for c in candidates}
        for dim in DIMENSIONS:
            path = run_dir / "pairwise" / case_filename(sample["case_key"]) / f"round-{index}" / f"{dim}.{'reverse' if reverse else 'forward'}.json"
            cache_key = _hash([key, index, reverse, dim])
            def request(draft):
                if dim == "SF" and len(plan["requirements"]) > 8:
                    return _sf_batches(path, cache_key, sample, ordered, plan, judge, resume, draft)
                return judge.judge_text_pair(sample, ordered, plan, dim, draft=draft) if draft is not None else judge.judge_text_pair(sample, ordered, plan, dim)
            value = _cached_request(path, cache_key, request,
                                    lambda v: validate_pair(v, dim, ordered, plan), reuse=resume)
            for label, c in zip(("A", "B"), ordered):
                answer = value["candidates"][label]
                dimensions[c["method"]][dim] = score_assessment(answer, dim, plan)
                assessments[c["method"]][dim] = answer
        totals = {m: overall_score(d) for m, d in dimensions.items()}
        methods = [c["method"] for c in candidates]
        result.append({"dimensions": dimensions, "assessments": assessments, "totals": totals,
                       "preference": preference(totals[methods[0]]["score"], totals[methods[1]]["score"])})
    return result



def _order_consistent(observations):
    a, b = observations
    if a["preference"] != b["preference"]:
        return False
    for method, dims in a["dimensions"].items():
        for dim, score in dims.items():
            other = b["dimensions"][method][dim]
            if (abs(score["dimension_score"] - other["dimension_score"]) > SCORING["order_score_tolerance"]
                    or score["critical_violation"] != other["critical_violation"]):
                return False
    return True


def _evaluate_case(run_dir, case, candidates, judge, identity, signature, resume):
    name = case_filename(case.key)
    target = run_dir / "pairs" / name
    error = target.with_suffix(".error.json")
    try:
        if resume and target.is_file():
            old = read_json(target)
            if old.get("signature") == signature and all(
                    (run_dir / "judgments" / name / c["method"] / f"{d}.json").is_file()
                    and read_json(run_dir / "judgments" / name / c["method"] / f"{d}.json").get("signature") == signature
                    for c in candidates for d in DIMENSIONS):
                error.unlink(missing_ok=True)
                return
        sample = case.to_dict()
        plan_key = _hash([VERSION, protocol_fingerprint(), identity, sample])
        plan = _cached_request(run_dir / "plans" / name, plan_key,
                               lambda draft: judge.plan_text(sample, draft=draft) if draft is not None else judge.plan_text(sample), lambda v: validate_plan(v, sample))
        if not candidates:
            error.unlink(missing_ok=True)
            return
        key = _hash([signature, plan])
        observations = _round(run_dir, sample, candidates, plan, judge, key, 0, resume)
        initial = [x["preference"] for x in observations]
        reviewed = not _order_consistent(observations)
        if reviewed:
            observations += _round(run_dir, sample, candidates, plan, judge, key, 1, resume)
        methods = [c["method"] for c in candidates]
        totals = {}
        for c in candidates:
            m = c["method"]
            dims = {}
            for dim in DIMENSIONS:
                dd = [x["dimensions"][m][dim] for x in observations]
                dims[dim] = {"dimension_score": round(sum(x["dimension_score"] for x in dd) / len(dd), 6),
                             "critical_violation": any(x["critical_violation"] for x in dd)}
                answers = [x["assessments"][m][dim] for x in observations]
                criteria = []
                for code in DIMENSIONS[dim][1]:
                    app = plan["applicability"][code]["applicable"]
                    row = {"criterion": code, "applicability": "applicable" if app else "not_applicable"}
                    if dim == "SF":
                        ids = {r["id"] for r in plan["requirements"] if r["criterion"] == code}
                        rr = [r for a in answers for r in a["requirements"] if r["id"] in ids]
                        counts = {s: sum(r["status"] == s for r in rr) for s in SCORING["requirement_values"]}
                        row.update(status_counts=counts, severity=None if not app else "major" if counts["omitted"] or counts["contradicted"] else "minor" if counts["weakened"] else "none")
                    elif dim == "IC":
                        rr = [r for a in answers for r in a["criteria"] if r["criterion"] == code]
                        severity = max((r["severity"] for r in rr), key=lambda s: {None: -1, "none": 0, "minor": 1, "major": 2}[s])
                        row.update(severity=severity)
                    else:
                        rr = [r for a in answers for r in a["criteria"] if r["criterion"] == code]
                        row.update(rating=round(sum(r["rating"] for r in rr) / len(rr), 6) if app else None,
                                   enhancement_types=[r["enhancement_type"] for r in rr])
                    criteria.append(row)
                write_json(run_dir / "judgments" / name / m / f"{dim}.json",
                           {"case_key": case.key, "method": m, "dimension": dim, "rubric_version": VERSION,
                            "signature": signature, "mock": bool(getattr(judge, "is_mock", False)),
                            **dims[dim], "criteria": criteria, "order_assessments": answers,
                            "order_count": len(observations), "created_at": _now()})
            totals[m] = overall_score(dims)
        final = preference(totals[methods[0]]["score"], totals[methods[1]]["score"])
        latest = [x["preference"] for x in observations[-2:]]
        stable = _order_consistent(observations[-2:]) and latest[0] == latest[1] == final
        write_json(target, {"case_key": case.key, "signature": signature, "methods": methods,
                            "mock": bool(getattr(judge, "is_mock", False)), "rubric_version": VERSION,
                            "totals": totals, "preference": final if stable else "unresolved",
                            "status": "stable_after_review" if stable and reviewed else "stable" if stable else "unresolved",
                            "reviewed": reviewed, "initial_preferences": initial, "latest_preferences": latest,
                            "order_totals": [x["totals"] for x in observations], "created_at": _now()})
        error.unlink(missing_ok=True)
    except Exception as exc:
        write_json(error, {"case_key": case.key, "stage": "text-v3", "error_type": type(exc).__name__,
                           "error": str(exc), "raw_model_response": getattr(exc, "raw_model_response", ""),
                           "usage": getattr(exc, "usage", None), "attempts": getattr(exc, "attempts", []),
                           "partial_response": getattr(exc, "partial_response", None), "created_at": _now()})


def run_text_pipeline(*, data_root, output_root, provider, candidate_file=None, modes=None,
                      limit=None, run_id=None, resume=True, inspect_content=True, case_filter_keys=None, max_workers=1, checkpoint_source=None):
    from .runner import select_cases
    from .text_report import summarize_text, render_text_markdown
    import re
    if type(max_workers) is not int or not 1 <= max_workers <= 128:
        raise ValueError("max_workers must be an integer in [1,128]")
    root = Path(data_root).resolve()
    all_cases = load_cases(root)
    cases = select_cases([c for c in all_cases if case_filter_keys is None or c.key in case_filter_keys], modes, limit)
    if not cases or any(c.mode != "t2v" or c.assets for c in cases):
        raise ValueError("text-v3 currently requires T2V text-only cases; select --mode t2v")
    rows = load_candidate_rows(candidate_file) if candidate_file else []
    validate_candidates(rows, all_cases)
    by_case = {c.key: sorted([r for r in rows if f"{r['mode']}:{r['capid']}" == c.key], key=lambda r: r["method"]) for c in cases}
    if candidate_file and any(len(v) != 2 for v in by_case.values()):
        raise ValueError("text-v3 requires exactly two PE methods for every selected case")
    if candidate_file and len({tuple(r["method"] for r in v) for v in by_case.values()}) != 1:
        raise ValueError("text-v3 requires the same two method identities across selected cases")
    judge = provider.judge_provider() if hasattr(provider, "judge_provider") else provider
    if not hasattr(judge, "plan_text") or not hasattr(judge, "judge_text_pair"):
        raise ValueError("provider does not support text-v3")
    identity = getattr(judge, "identity", f"{judge.name}:{judge.version}")
    run_id = run_id or "text-v3-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    output = Path(output_root).expanduser().resolve()
    disk = Path("/pfs/pfs-spMCeu").resolve()
    if not disk.is_dir() or not output.is_relative_to(disk):
        raise ValueError("output_root must be on the configured data disk under /pfs/pfs-spMCeu")
    run_dir = output / run_id
    path = run_dir / "run.json"
    previous = read_json(path) if path.is_file() else None
    dataset_fp = dataset_fingerprint(root, {})
    candidate_fp = _hash([r for key in sorted(by_case) for r in by_case[key]]) if candidate_file else None
    scope = [c.key for c in cases]
    if previous:
        if (previous.get("rubric_version") != VERSION or previous.get("protocol") != "text-v3"
                or previous.get("dataset_fingerprint") != dataset_fp or previous.get("case_keys") != scope
                or previous.get("judge_identity") != identity or previous.get("scoring_policy") != SCORING
                or previous.get("protocol_fingerprint") != protocol_fingerprint()):
            raise ValueError("existing run has a different protocol, dataset, scope, provider or scoring policy; use a new run_id")
        if previous.get("candidate_fingerprint") is not None and previous["candidate_fingerprint"] != candidate_fp:
            raise ValueError("candidate content changed; use a new run_id")
    meta = {"run_id": run_id, "protocol": "text-v3", "rubric_version": VERSION, "scoring_policy": SCORING,
            "status": "running", "started_at": previous["started_at"] if previous else _now(),
            "dataset_fingerprint": dataset_fp, "candidate_fingerprint": candidate_fp,
            "case_count": len(cases), "case_keys": scope, "candidate_count": sum(map(len, by_case.values())),
            "data_root": str(root), "output_root": str(output), "candidate_file": str(candidate_file) if candidate_file else None,
            "judge_identity": identity, "provider": provider.metadata(),
            "mock": bool(getattr(judge, "is_mock", False)), "generated_video_evaluation": False, "protocol_fingerprint": protocol_fingerprint()}
    signature = _hash([VERSION, protocol_fingerprint(), SCORING, dataset_fp, candidate_fp, identity])
    meta["signature"] = signature
    if checkpoint_source and not previous:
        from .text_checkpoint import import_checkpoints
        sources = checkpoint_source if isinstance(checkpoint_source, (list, tuple)) else [checkpoint_source]
        imports = [import_checkpoints(Path(source), run_dir, meta, cases, by_case, identity, signature) for source in sources]
        meta["checkpoint_import"] = imports[0] if len(imports) == 1 else {"sources": imports}
        write_json(run_dir / "checkpoint_import.json", meta["checkpoint_import"])
    if previous and previous.get("checkpoint_import"):
        meta["checkpoint_import"] = previous["checkpoint_import"]
    write_json(path, meta)
    write_json(run_dir / "cases.json", {"cases": [c.to_dict() for c in cases], "asset_manifest": {}})
    for candidates in by_case.values():
        for row in candidates:
            write_json(run_dir / "candidates" / row["method"] / case_filename(f"{row['mode']}:{row['capid']}"), row)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_evaluate_case, run_dir, c, by_case[c.key], judge, identity, signature, resume) for c in cases]
        for count, f in enumerate(as_completed(futures), 1):
            f.result()
            complete = sum(1 for p in (run_dir / "pairs").glob("*.json") if not p.name.endswith(".error.json"))
            errors = len(list((run_dir / "pairs").glob("*.error.json")))
            progress = {"processed": count, "total": len(cases), "complete": complete, "errors": errors, "updated_at": _now()}
            write_json(run_dir / "progress.json", progress)
            print(f"[text-v3] {count}/{len(cases)} complete={complete} errors={errors}", flush=True)
    summary = summarize_text(run_dir)
    meta.update(status=("completed_with_errors" if summary["error_files"] else "prepared" if not candidate_file else
                        "completed_with_unresolved_pairs" if summary["comparison"]["unresolved"] else "completed"),
                error_count=summary["error_files"], completed_at=_now(), summary_path=str(run_dir / "summary.json"))
    write_json(path, meta)
    (run_dir / "summary.md").write_text(render_text_markdown(summary), encoding="utf-8")
    return run_dir
