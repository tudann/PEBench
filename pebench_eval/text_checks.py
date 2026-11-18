"""Controlled text mutations test model behavior without videos or human labels."""
from __future__ import annotations

import json
import re
from pathlib import Path
from .storage import read_json, write_json
from .text_runner import run_text_pipeline

SOURCE = ("Flat 2D vector animation in one continuous take. The camera is static in a wide eye-level view. "
          "One woman in a red coat stands to the left of a blue wooden box. "
          "She opens the lid with her right hand, then closes it. "
          "No dialogue, no music, no sound effects, no on-screen text.")
CHECKS = [
    ("identical", SOURCE, "equal"),
    ("paraphrase", "Minimalist flat two-dimensional vector animation, one uninterrupted take. "
     "A motionless camera shows a wide view from eye level. A single woman wearing a red coat "
     "is on the left of a blue box made of wood. Using her right hand, she opens its lid and then shuts it. "
     "There is no speech, music, sound effects or visible text.", "paraphrase"),
    ("compatible_detail", SOURCE + " Her opening and closing motion follows the box's rear hinge; "
     "her gaze remains on her right hand. The wide framing keeps the woman and the box visible together.", "compatible"),
    ("padding", SOURCE + " Stunning, incredible, amazing, exquisite, breathtaking, top quality, superb quality.", "padding"),
    ("wrong_count", SOURCE.replace("One woman", "Two women"), "violation"),
    ("forbidden_music", SOURCE + " A cheerful piano soundtrack plays throughout.", "violation"),
    ("wrong_order", SOURCE.replace("opens the lid with her right hand, then closes it", "closes the lid with her right hand, then opens it"), "violation"),
]


def run_checks(*, output_root, run_id, provider, max_workers=2, limit=None):
    if getattr(provider, "is_mock", False):
        raise ValueError("rubric-check requires a real model; mock tests cannot validate semantic behavior")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", run_id) or run_id in {".", ".."}:
        raise ValueError("invalid run_id")
    output = Path(output_root).resolve()
    if not output.is_relative_to(Path('/pfs/pfs-spMCeu').resolve()):
        raise ValueError("rubric-check outputs must stay on the configured data disk")
    chosen = CHECKS if limit is None else CHECKS[:limit]
    data = output / (run_id + '-inputs')
    data.mkdir(parents=True, exist_ok=True)
    records, candidates = [], []
    for name, variant, _ in chosen:
        records.append({"capid": name, "prompt": SOURCE, "duration": None,
                        "generation_type": "t2v", "task_category": "Controlled text check", "task_type": "Text invariance"})
        candidates += [{"mode": "t2v", "capid": name, "method": method, "pe_prompt": text}
                       for method, text in [('baseline', SOURCE), ('variant', variant)]]
    write_json(data / 't2v_500.json', records)
    write_json(data / 'i2v_300.json', [])
    write_json(data / 'r2v_300.json', [])
    candidate_file = data / 'candidates.jsonl'
    candidate_file.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in candidates), encoding='utf-8')
    run = run_text_pipeline(data_root=data, output_root=output, provider=provider, candidate_file=candidate_file,
                            modes={'t2v'}, run_id=run_id, max_workers=max_workers)
    checks = []
    for name, _, kind in chosen:
        path = run / 'pairs' / f't2v__{name}.json'
        error = path.with_suffix('.error.json')
        if error.is_file() or not path.is_file():
            checks.append({'check': name, 'passed': False, 'reason': 'evaluation failed',
                           'error': read_json(error) if error.is_file() else 'missing result'})
            continue
        pair = read_json(path)
        dims = {m: {d: read_json(run / 'judgments' / path.name / m / f'{d}.json')['dimension_score']
                    for d in ('SF', 'IC', 'DC')} for m in ('baseline', 'variant')}
        b, v = dims['baseline'], dims['variant']
        stable = pair['status'] != 'unresolved'
        if kind == 'equal': passed = all(b[d] == v[d] for d in b) and pair['preference'] == 'tie'
        elif kind == 'paraphrase': passed = all(abs(b[d] - v[d]) <= 5 for d in b)
        elif kind == 'compatible': passed = v['SF'] >= b['SF'] and v['IC'] >= b['IC'] and v['DC'] >= b['DC'] - 2
        elif kind == 'padding': passed = v['DC'] <= b['DC'] + 2
        else: passed = v['SF'] < b['SF'] and pair['totals']['variant']['critical_violation']
        checks.append({'check': name, 'passed': passed and stable, 'semantic_passed': passed, 'stable': stable,
                       'dimension_scores': dims, 'totals': pair['totals'], 'pair_status': pair['status']})
    result = {'purpose': 'controlled_text_calibration_not_benchmark', 'run_dir': str(run),
              'checks': checks, 'passed': sum(c['passed'] for c in checks), 'total': len(checks),
              'all_passed': all(c['passed'] for c in checks),
              'semantic_passed': sum(c.get('semantic_passed', False) for c in checks),
              'order_stable': sum(c.get('stable', False) for c in checks),
              'limitations': 'Controlled mutations validate stated text rules, not rendered video quality or human preference.'}
    write_json(run / 'calibration.json', result)
    lines = ['# Controlled text rubric checks', '', '| Check | Semantic rule | Order stable | Overall passed | Pair status |', '|---|---|---|---|---|']
    lines += [f"| {c['check']} | {c.get('semantic_passed', False)} | {c.get('stable', False)} | {c['passed']} | {c.get('pair_status', c.get('reason'))} |" for c in checks]
    (run / 'calibration.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
