"""Explicit, audited import of compatible text-v3 checkpoints into a new run."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from .schema import DIMENSIONS
from .storage import case_filename, read_json, write_json
from .text_rubric import VERSION, SCORING, RepairNeeded, protocol_fingerprint, validate_pair, validate_plan

# Only the reviewed preceding contract is eligible. New scoring/model/data is never migrated.
PILOT_FORMATS = {"5083bcfa5f031bc98f7b4e284dcd32e47a299fa3c038c8de56fc1b3597ca80db", "046bfbf08f34241942ae127b0616187498e25589b695bad022b5f1e5dcd55267", "be3573edab1c3b00718af8a4875600cd63bf686404ec1d05adfba7a80f2eea63"}
PREVIOUS_FORMAT = "7ab7e64ed7eeb76b066d83966d214df9da59ee32e009b6ead3821149a7c5354b"


def _decode(raw):
    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.splitlines()[1:-1])
    return json.loads(raw)


def import_checkpoints(source, destination, meta, cases, by_case, identity, signature):
    from .text_runner import _hash, _now
    source, destination = source.resolve(), destination.resolve()
    if source == destination or (destination / 'run.json').exists():
        raise ValueError('checkpoint import requires a distinct new run')
    old = read_json(source / 'run.json')
    for field in ('rubric_version', 'protocol', 'dataset_fingerprint', 'judge_identity', 'scoring_policy'):
        if old.get(field) != meta[field]: raise ValueError(f'incompatible checkpoint source: {field}')
    if old.get('protocol_fingerprint') not in {PREVIOUS_FORMAT, protocol_fingerprint(), *PILOT_FORMATS}:
        raise ValueError('unreviewed checkpoint protocol; do not import')
    if old.get('status') == 'running': raise ValueError('stop the checkpoint source run before importing')
    old_signature = _hash([VERSION, old['protocol_fingerprint'], SCORING, old['dataset_fingerprint'], old['candidate_fingerprint'], identity])
    if old.get('signature') != old_signature: raise ValueError('checkpoint source signature mismatch')
    source_samples = {c['case_key']: c for c in read_json(source / 'cases.json')['cases']}
    cases = [c for c in cases if c.key in source_samples]
    if not cases: raise ValueError('checkpoint source has no cases in the selected scope')
    for case in cases:
        if source_samples.get(case.key) != case.to_dict(): raise ValueError(f'checkpoint source sample changed: {case.key}')
        for candidate in by_case[case.key]:
            old_candidate = source / 'candidates' / candidate['method'] / case_filename(case.key)
            if not old_candidate.is_file() or read_json(old_candidate) != candidate:
                raise ValueError(f'checkpoint source candidate changed: {case.key}')
    stats = {'source': str(source), 'source_signature': old_signature, 'source_protocol_fingerprint': old['protocol_fingerprint'],
             'created_at': _now(), 'plans': 0, 'pairwise': 0, 'pending_plans': 0, 'pending_pairs': 0, 'sf_parts': 0, 'pending_sf_parts': 0, 'rejected': []}

    def save(path, key, value, origin, pending=False):
        result = {'cache_key': key, 'created_at': _now(), 'imported_from': str(origin),
                  'source_sha256': _hash(read_json(origin))}
        if pending:
            result['draft'] = deepcopy(value)
            path = path.with_suffix('.pending.json')
        else:
            result['result'] = deepcopy(value)
            raw = read_json(origin)
            result['raw_model_response'] = raw.get('raw_model_response', '')
            result['usage'] = raw.get('usage')
            result['attempts'] = raw.get('attempts', [])
        write_json(path, result)
        if not pending: path.with_suffix('.pending.json').unlink(missing_ok=True)

    for case in cases:
        sample, name = case.to_dict(), case_filename(case.key)
        plan_path = source / 'plans' / name
        error_path = source / 'pairs' / name.replace('.json', '.error.json')
        error_value = read_json(error_path) if error_path.is_file() else {}
        draft = None
        if error_value.get('raw_model_response'):
            try: draft = _decode(error_value['raw_model_response'])
            except (ValueError, TypeError): pass
        plan_key = _hash([VERSION, protocol_fingerprint(), identity, sample])
        pending_plan = plan_path.with_suffix('.pending.json')
        if not plan_path.is_file() and pending_plan.is_file():
            pending_value = read_json(pending_plan)
            if pending_value.get('cache_key') == _hash([VERSION, old['protocol_fingerprint'], identity, sample]):
                draft = pending_value['draft']
                error_path = pending_plan
        if not plan_path.is_file():
            if isinstance(draft, dict) and 'requirements' in draft and 'applicability' in draft:
                try:
                    plan = validate_plan(draft, sample)
                    save(destination / 'plans' / name, plan_key, plan, error_path)
                    stats['plans'] += 1
                except RepairNeeded as exc:
                    save(destination / 'plans' / name, plan_key, exc.draft, error_path, pending=True)
                    stats['pending_plans'] += 1
                except ValueError as exc:
                    stats['rejected'].append({'case_key': case.key, 'stage': 'plan', 'error': str(exc)})
            continue
        cached = read_json(plan_path)
        if cached.get('cache_key') != _hash([VERSION, old['protocol_fingerprint'], identity, sample]):
            stats['rejected'].append({'case_key': case.key, 'stage': 'plan', 'error': 'cache key mismatch'})
            continue
        try: plan = validate_plan(cached['result'], sample)
        except ValueError as exc:
            stats['rejected'].append({'case_key': case.key, 'stage': 'plan', 'error': str(exc)})
            continue
        save(destination / 'plans' / name, plan_key, plan, plan_path)
        stats['plans'] += 1
        # All plan fields must remain identical for previous assessments to be reused.
        if plan != cached['result']: continue
        old_key, key = _hash([old_signature, plan]), _hash([signature, plan])
        missing = None
        for index in (0, 1):
            for reverse in (False, True):
                ordered = list(reversed(by_case[case.key])) if reverse else by_case[case.key]
                for dim in DIMENSIONS:
                    relative = Path('pairwise') / name / f'round-{index}' / f"{dim}.{'reverse' if reverse else 'forward'}.json"
                    checkpoint = source / relative
                    cache_key = _hash([key, index, reverse, dim])
                    if checkpoint.is_file():
                        old_pair = read_json(checkpoint)
                        if old_pair.get('cache_key') != _hash([old_key, index, reverse, dim]): continue
                        try: value = validate_pair(old_pair['result'], dim, ordered, plan)
                        except ValueError as exc:
                            stats['rejected'].append({'case_key': case.key, 'stage': str(relative), 'error': str(exc)})
                            continue
                        save(destination / relative, cache_key, value, checkpoint)
                        stats['pairwise'] += 1
                    elif missing is None:
                        missing = (relative, cache_key, dim, ordered)
                    if dim == 'SF' and len(plan['requirements']) > 8:
                        for start in range(0, len(plan['requirements']), 8):
                            part_relative = relative.parent / (relative.stem + '.parts') / f'{start // 8:03d}.json'
                            part = source / part_relative
                            pending_part = part.with_suffix('.pending.json')
                            subset = deepcopy(plan)
                            subset['requirements'] = plan['requirements'][start:start + 8]
                            old_part_key = _hash([_hash([old_key, index, reverse, dim]), 'SF-batch-8', subset['requirements']])
                            part_key = _hash([cache_key, 'SF-batch-8', subset['requirements']])
                            for origin, is_pending in ((part, False), (pending_part, True)):
                                if not origin.is_file(): continue
                                part_value = read_json(origin)
                                if part_value.get('cache_key') != old_part_key: continue
                                draft_part = part_value.get('draft') if is_pending else part_value.get('result')
                                try:
                                    value = validate_pair(draft_part, dim, ordered, subset)
                                    save(destination / part_relative, part_key, value, origin)
                                    stats['sf_parts'] += 1
                                except RepairNeeded as exc:
                                    save(destination / part_relative, part_key, exc.draft, origin, pending=True)
                                    stats['pending_sf_parts'] += 1
                                except ValueError as exc:
                                    stats['rejected'].append({'case_key': case.key, 'stage': str(part_relative), 'error': str(exc)})
        # Old runner was sequential per case: its final failed response belongs to first missing call.
        if missing:
            pending_pair = (source / missing[0]).with_suffix('.pending.json')
            if pending_pair.is_file():
                pending_value = read_json(pending_pair)
                index = int(missing[0].parts[-2].split('-')[1])
                reverse = '.reverse.' in missing[0].name
                if pending_value.get('cache_key') == _hash([old_key, index, reverse, missing[2]]):
                    draft, error_path = pending_value['draft'], pending_pair
        if missing and isinstance(draft, dict) and draft.get('dimension') == missing[2]:
            relative, cache_key, dim, ordered = missing
            try:
                value = validate_pair(draft, dim, ordered, plan)
                save(destination / relative, cache_key, value, error_path)
                stats['pairwise'] += 1
            except RepairNeeded as exc:
                save(destination / relative, cache_key, exc.draft, error_path, pending=True)
                stats['pending_pairs'] += 1
            except ValueError as exc:
                stats['rejected'].append({'case_key': case.key, 'stage': str(relative), 'error': str(exc)})
    write_json(destination / 'checkpoint_import.json', stats)
    return stats
