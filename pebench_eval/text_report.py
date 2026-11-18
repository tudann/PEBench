"""Versioned text-v3 reports; historical model-reported scores are never converted."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from .schema import DIMENSIONS
from .storage import read_json, write_json
from .text_rubric import SCORING, VERSION, preference


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def order_averaged_totals(pair, method):
    orders = pair.get("order_totals") or []
    if not orders:
        return pair["totals"][method]
    values = [order[method] for order in orders]
    return {"score": round(sum(v["score"] for v in values) / len(values), 6),
            "uncapped_score": round(sum(v["uncapped_score"] for v in values) / len(values), 6),
            "critical_violation": any(v["critical_violation"] for v in values),
            "critical_violation_order_count": sum(v["critical_violation"] for v in values),
            "order_count": len(values)}


def summarize_text(run_dir):
    run = Path(run_dir)
    meta = read_json(run / "run.json")
    if meta.get("rubric_version") != VERSION:
        raise ValueError("text-v3 summary requires matching version")
    signature = meta["signature"]
    pairs, judgments = [], []
    for path in sorted((run / "pairs").glob("*.json")):
        if path.name.endswith(".error.json") or path.with_suffix(".error.json").exists(): continue
        pair = read_json(path)
        if pair.get("signature") != signature: continue
        rows = []
        for method in pair["methods"]:
            for dim in DIMENSIONS:
                p = run / "judgments" / path.name / method / f"{dim}.json"
                if p.is_file():
                    v = read_json(p)
                    if v.get("signature") == signature and v.get("rubric_version") == VERSION:
                        rows.append(v)
        if len(rows) != 6: continue
        pairs.append(pair)
        judgments.extend(rows)
    real_pairs = [p for p in pairs if not p["mock"]]
    real_rows = [v for v in judgments if not v["mock"]]
    candidates = [read_json(p) for p in (run / "candidates").glob("*/*.json")]
    methods = sorted({v["method"] for v in candidates})
    case_info = {c["case_key"]: c for c in read_json(run / "cases.json")["cases"]}

    def dimensions(rows):
        result = {}
        for dim, (_, codes) in DIMENSIONS.items():
            dd = [v for v in rows if v["dimension"] == dim]
            stats = {}
            for code in codes:
                rr = [r for v in dd for r in v["criteria"] if r["criterion"] == code and r["applicability"] == "applicable"]
                if dim == "DC":
                    stats[code] = {"applicable": len(rr), "mean_rating": _mean([r["rating"] for r in rr]),
                                   "enhancement_types": dict(Counter(k for r in rr for k in r["enhancement_types"]))}
                else:
                    passed = sum(r["severity"] == "none" for r in rr)
                    stats[code] = {"applicable": len(rr), "passed": passed,
                                   "pass_rate": passed / len(rr) if rr else None,
                                   "severity_counts": dict(Counter(r["severity"] for r in rr))}
            result[dim] = {"count": len(dd), "real_score_count": len(dd),
                           "mean_score": _mean([v["dimension_score"] for v in dd]), "criteria": stats}
        return result

    method_out = {}
    for method in methods:
        rows = [v for v in real_rows if v["method"] == method]
        pp = [p for p in real_pairs if method in p["methods"]]
        groups = defaultdict(list)
        for v in rows:
            info = case_info[v["case_key"]]
            for label, value in (("ability", (info.get("source_metadata") or {}).get("ability") or info.get("task_type")),
                                 ("scene", (info.get("source_metadata") or {}).get("scene") or info.get("task_category"))):
                if value is not None: groups[f"{label}:{value}"].append(v)
        method_out[method] = {"dimensions": dimensions(rows),
                              "overall": {"complete_cases": len(pp), "mean_score": _mean([order_averaged_totals(p, method)["score"] for p in pp]),
                                          "critical_violation_cases": sum(order_averaged_totals(p, method)["critical_violation"] for p in pp),
                                          "critical_violation_order_rate": _mean([
                                              order_averaged_totals(p, method)["critical_violation_order_count"] /
                                              order_averaged_totals(p, method)["order_count"] for p in pp])},
                              "groups": {g: {"dimensions": dimensions(vv),
                                             "mean_overall_score": _mean([order_averaged_totals(p, method)["score"] for p in pp if p["case_key"] in {v["case_key"] for v in vv}])}
                                         for g, vv in sorted(groups.items())}}
    comparison = {"complete_real_pairs": len(real_pairs), "mock_pairs": len(pairs) - len(real_pairs),
                  "pending_cases": meta["case_count"] - len(pairs), "reviewed": sum(p["reviewed"] for p in real_pairs),
                  "unresolved": sum(p.get("status") == "unresolved" or p.get("preference") == "unresolved" for p in real_pairs),
                  "all_complete_pairs_comparable": True, "aggregation": "mean of order-level final scores",
                  "by_pair": {}}
    case_comparisons = []
    buckets = defaultdict(list)
    for p in real_pairs: buckets[tuple(p["methods"])].append(p)
    for method_pair, pp in sorted(buckets.items()):
        point_rows = []
        for pair in pp:
            totals = {m: order_averaged_totals(pair, m) for m in method_pair}
            result = preference(totals[method_pair[0]]["score"], totals[method_pair[1]]["score"])
            point_rows.append((pair, totals, result))
            case_comparisons.append({"case_key": pair["case_key"], "methods": list(method_pair),
                                     "order_count": totals[method_pair[0]]["order_count"],
                                     "scores": {m: totals[m]["score"] for m in method_pair},
                                     "score_delta_first_minus_second": round(totals[method_pair[0]]["score"] - totals[method_pair[1]]["score"], 6),
                                     "preference": result,
                                     "stability_status": pair.get("status", "unresolved")})
        counts = Counter(result for _, _, result in point_rows)
        stable_counts = Counter(result for pair, _, result in point_rows if pair.get("status") != "unresolved")
        n = len(point_rows)
        stable_n = sum(stable_counts.values())
        comparison["by_pair"][" vs ".join(method_pair)] = {
            "methods": list(method_pair), "first_wins": counts["first"], "second_wins": counts["second"],
            "ties": counts["tie"], "unresolved": sum(pair.get("status") == "unresolved" for pair, _, _ in point_rows),
            "comparable_pairs": n,
            "first_preference_score": round(100 * (counts["first"] + 0.5 * counts["tie"]) / n, 4) if n else None,
            "stable_only_first_wins": stable_counts["first"], "stable_only_second_wins": stable_counts["second"],
            "stable_only_ties": stable_counts["tie"], "stable_only_pairs": stable_n,
            "stable_only_first_preference_score": round(100 * (stable_counts["first"] + 0.5 * stable_counts["tie"]) / stable_n, 4) if stable_n else None}
    result = {"run_id": meta["run_id"], "protocol": "text-v3", "rubric_version": VERSION,
              "scoring_policy": meta["scoring_policy"], "dataset_fingerprint": meta["dataset_fingerprint"],
              "summary_status": "real" if real_pairs else "mock_only" if pairs else "no_real_judgments",
              "candidate_cases": len(candidates), "candidate_case_slots": 2 * meta["case_count"],
              "judgment_rows": len(judgments), "mock_judgment_rows": len(judgments) - len(real_rows),
              "error_files": len(list((run / "pairs").glob("*.error.json"))), "methods": method_out,
              "comparison": comparison}
    write_json(run / "summary.json", result)
    write_json(run / "paired_comparison_by_case.json", {"run_id": meta["run_id"],
               "aggregation": comparison["aggregation"], "cases": case_comparisons})
    return result


def render_text_markdown(summary):
    lines = [f"# PE text evaluation: `{summary['run_id']}`", "",
             f"Protocol: `{summary['rubric_version']}`; status: **{summary['summary_status']}**", "",
             "Text only. SF/IC use fixed violation values; DC uses 0–4 positive ratings. "
             f"Overall = {SCORING["dimension_weights"]["SF"]:.2f} SF + {SCORING["dimension_weights"]["IC"]:.2f} IC + {SCORING["dimension_weights"]["DC"]:.2f} DC, capped at 69 for critical violations. "
             "Per-case totals are the mean of order-level final scores. This is a custom evaluation protocol, not the paper's original score conversion.", "",
             f"Judgments: {summary['judgment_rows']}; errors: {summary['error_files']}; "
             f"pending pairs: {summary['comparison']['pending_cases']}; order-unstable pairs: {summary['comparison']['unresolved']}.", "",
             "All complete pairs have a numeric comparison using the mean of their saved order-level final scores. "
             "Order-unstable pairs remain flagged and are included in the point-estimate comparison; stable-only results are shown separately.", "",
             "| Method | Complete cases | SF | IC | DC | Mean order-level overall | Cases with any critical flag | Mean critical-flagged order rate |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, v in summary["methods"].items():
        d, o = v["dimensions"], v["overall"]
        lines.append(f"| {method} | {o['complete_cases']} | {d['SF']['mean_score']} | {d['IC']['mean_score']} | {d['DC']['mean_score']} | {o['mean_score']} | {o['critical_violation_cases']} | {o['critical_violation_order_rate']} |")
    lines += ["", "## Automatic paired comparison", "",
              "Each pair's point estimate averages its two orderings, plus the two review orderings when triggered. "
              "Preference score = 100 × (wins + 0.5 × ties) / all complete pairs; it is not absolute PE quality or a certainty measure.", "",
              "| First / second method | First wins | Ties | Second wins | Compared | Order-unstable | First preference score |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, p in summary["comparison"]["by_pair"].items():
        lines.append(f"| {name} | {p['first_wins']} | {p['ties']} | {p['second_wins']} | {p['comparable_pairs']} | {p['unresolved']} | {p['first_preference_score']} |")
        lines += ["", "Stability-filtered view of the same order-averaged scores:", "",
                  "| First / second method | First wins | Ties | Second wins | Stable pairs | First preference score |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines.append(f"| {name} | {p['stable_only_first_wins']} | {p['stable_only_ties']} | {p['stable_only_second_wins']} | {p['stable_only_pairs']} | {p['stable_only_first_preference_score']} |")
    lines += ["", "Per-case score differences and stability flags: `paired_comparison_by_case.json`."]
    for method, v in summary["methods"].items():
        lines += ["", f"## {method}: criterion detail", "", "| DC criterion | Applicable cases | Mean rating (0–4) |", "|---|---:|---:|"]
        for code, stat in v["dimensions"]["DC"]["criteria"].items():
            lines.append(f"| {code} | {stat['applicable']} | {stat['mean_rating']} |")
        lines += ["", "| SF/IC criterion | Applicable cases | No-issue rate |", "|---|---:|---:|"]
        for dim in ("SF", "IC"):
            for code, stat in v["dimensions"][dim]["criteria"].items():
                lines.append(f"| {code} | {stat['applicable']} | {stat['pass_rate']} |")
        lines += ["", "| Group | Cases | SF | IC | DC | Overall |", "|---|---:|---:|---:|---:|---:|"]
        for name, g in v["groups"].items():
            d = g["dimensions"]
            lines.append(f"| {name} | {d['SF']['count']} | {d['SF']['mean_score']} | {d['IC']['mean_score']} | {d['DC']['mean_score']} | {g['mean_overall_score']} |")
    return "\n".join(lines) + "\n"
