from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import DIMENSIONS, RUBRIC_VERSION
from .storage import read_json, write_json


def _current_judgments(run_dir: Path):
    root = run_dir / "judgments"
    if not root.is_dir():
        return []
    run = read_json(run_dir / "run.json")
    provider = run.get("provider", {})
    if "judge" in provider and isinstance(provider["judge"], dict):
        j = provider["judge"]
        judge_identity = j.get("identity") or (f"{j.get('name')}:{j.get('model')}:{j.get('version')}" if j.get("model") else f"{j.get('name')}:{j.get('version')}")
    else:
        judge_identity = f"{provider.get('name')}:{provider.get('version')}"
    current = []
    for file in sorted(root.glob("*/*/*.json")):
        try:
            value = read_json(file)
            case_key, method, dimension = value.get("case_key"), value.get("method"), value.get("dimension")
            if not case_key or not method or dimension not in DIMENSIONS:
                continue
            candidate_path = run_dir / "candidates" / method / file.parents[1].name
            evidence_path = run_dir / "evidence" / file.parents[1].name
            evidence_error_path = evidence_path.with_name(evidence_path.name + ".error.json")
            if not candidate_path.is_file() or not evidence_path.is_file() or evidence_error_path.exists():
                continue
            candidate = read_json(candidate_path)
            evidence = read_json(evidence_path)
            expected_judge_key = hashlib.sha256(f"{judge_identity}:{RUBRIC_VERSION}:{dimension}".encode()).hexdigest()
            if (candidate.get("candidate_hash") != value.get("candidate_hash")
                    or evidence.get("cache_key") != value.get("evidence_cache_key")
                    or value.get("judge_cache_key") != expected_judge_key):
                continue
            current.append((file, value))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return current


def _aggregate_dimension(values: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    real_values = [v for v in values if not v.get("mock")]
    scores = [v["dimension_score"] for v in real_values
              if isinstance(v.get("dimension_score"), (int, float)) and not isinstance(v.get("dimension_score"), bool)]
    criterion_stats = {}
    for code in DIMENSIONS[dimension][1]:
        applicable = passed = 0
        severity_counts = defaultdict(int)
        for value in real_values:
            row = next((x for x in value.get("criteria", []) if x.get("criterion") == code), None)
            if not row or row.get("applicability") != "applicable":
                continue
            applicable += 1
            severity_counts[row.get("severity")] += 1
            if row.get("severity") == "none":
                passed += 1
        criterion_stats[code] = {"applicable": applicable, "passed": passed,
                                 "pass_rate": round(passed / applicable, 6) if applicable else None,
                                 "severity_counts": dict(severity_counts)}
    return {"count": len(real_values), "mock_count": len(values) - len(real_values),
            "real_score_count": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
            "criteria": criterion_stats}


def summarize(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    meta = read_json(run / "run.json")
    case_doc = read_json(run / "cases.json") if (run / "cases.json").is_file() else {"cases": []}
    case_info = {row["case_key"]: row for row in case_doc.get("cases", [])}
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = _current_judgments(run)
    mock_count = sum(1 for _, value in files if value.get("mock"))
    for _, value in files:
        by_method[value["method"]].append(value)
    candidate_files = list((run / "candidates").glob("*/*.json")) if (run / "candidates").exists() else []
    method_names = {read_json(path).get("method", path.parent.name) for path in candidate_files}
    methods: dict[str, Any] = {}
    for method in sorted(method_names | set(by_method)):
        rows = by_method.get(method, [])
        groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for value in rows:
            groups["overall"][value["dimension"]].append(value)
            info = case_info.get(value.get("case_key"), {})
            for field, value_key in (("mode", "mode"), ("task_category", "task_category"),
                                     ("task_type", "task_type"), ("duration", "duration")):
                group_value = info.get(value_key)
                if group_value is not None:
                    groups[f"{field}:{group_value}"][value["dimension"]].append(value)
        group_out = {}
        for group_key, dimensions in sorted(groups.items()):
            group_out[group_key] = {dimension: _aggregate_dimension(dimensions.get(dimension, []), dimension)
                                    for dimension in DIMENSIONS}
        methods[method] = {"dimensions": group_out.get("overall", {
            dimension: _aggregate_dimension([], dimension) for dimension in DIMENSIONS}),
            "groups": group_out}
    errors = list(run.rglob("*.error.json"))
    judgment_count = len(files)
    expected = len(method_names) * int(meta.get("case_count", 0))
    observed_case_methods = len(candidate_files)
    if judgment_count and mock_count == judgment_count:
        status = "mock_only"
    elif mock_count:
        status = "mixed_mock_and_real"
    elif judgment_count:
        status = "real"
    else:
        status = "no_real_judgments"
    result = {"run_id": meta["run_id"], "dataset_fingerprint": meta["dataset_fingerprint"],
              "summary_status": status, "candidate_cases": observed_case_methods,
              "candidate_case_slots": expected, "judgment_rows": judgment_count,
              "mock_judgment_rows": mock_count, "error_files": len(errors), "methods": methods}
    write_json(run / "summary.json", result)
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# PEBench evaluation summary: `{summary['run_id']}`", "",
             f"Dataset fingerprint: `{summary['dataset_fingerprint']}`", "",
             f"Status: **{summary['summary_status']}**", "",
             f"Candidate case records: {summary['candidate_cases']} / {summary['candidate_case_slots']}; "
             f"judgment rows: {summary['judgment_rows']}; failed stage files: {summary['error_files']}", ""]
    if summary["summary_status"] == "mock_only":
        lines += ["> Mock results are protocol smoke tests. They are excluded from scores and pass rates.", ""]
    for method, payload in summary["methods"].items():
        lines += [f"## Method: {method}", "", "| Dimension | Real cases | Mock cases | Real scores | Mean score |", "|---|---:|---:|---:|---:|"]
        for dimension, item in payload["dimensions"].items():
            lines.append(f"| {dimension} | {item['count']} | {item['mock_count']} | {item['real_score_count']} | {item['mean_score'] if item['mean_score'] is not None else 'n/a'} |")
        lines.append("")
        group_rows = [(key, value) for key, value in payload.get("groups", {}).items() if key != "overall"]
        for group_key, dimensions in group_rows:
            lines += [f"### {group_key}", "", "| Dimension | Cases | Mean score |", "|---|---:|---:|"]
            for dimension, item in dimensions.items():
                if item["count"] or item["mock_count"]:
                    score = "n/a" if item["mean_score"] is None else item["mean_score"]
                    lines.append(f"| {dimension} | {item['count']} | {score} |")
            lines.append("")
            lines += ["| Criterion | Applicable | Pass rate |", "|---|---:|---:|"]
            for dimension, item in dimensions.items():
                for code, stat in item["criteria"].items():
                    if not stat["applicable"]:
                        continue
                    rate = "n/a" if stat["pass_rate"] is None else f"{stat['pass_rate']:.4f}"
                    lines.append(f"| {code} ({dimension}) | {stat['applicable']} | {rate} |")
            lines.append("")
        lines += ["| Criterion | Applicable | Pass rate |", "|---|---:|---:|"]
        for dimension, item in payload["dimensions"].items():
            for code, stat in item["criteria"].items():
                rate = "n/a" if stat["pass_rate"] is None else f"{stat['pass_rate']:.4f}"
                lines.append(f"| {code} ({dimension}) | {stat['applicable']} | {rate} |")
        lines.append("")
    return "\n".join(lines)
