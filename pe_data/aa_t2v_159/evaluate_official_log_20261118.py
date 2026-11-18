"""Evaluate official vs log PE on GPU 4 using frozen source-only plans."""
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pebench_eval.config import load_config, provider_from_config
from pebench_eval.dataset import load_cases
from pebench_eval.storage import case_filename, read_json, write_json
from pebench_eval.text_rubric import VERSION, SCORING, protocol_fingerprint, validate_plan
from pebench_eval.text_runner import _hash, _now, run_text_pipeline

BASE = ROOT / 'pe_data/aa_t2v_159'
SOURCE = BASE / 'runs/aa-t2v-159-text-v3-formatfix-complete-20261117'
CFG = load_config(BASE / 'config_text_v3_official_log_gpu4_20261118.yaml')
RUN = Path(CFG['paths']['output_root']) / CFG['run']['run_id']
provider = provider_from_config(CFG['provider'])
judge = provider.judge_provider()
cases = [c for c in load_cases(BASE) if c.mode == 't2v']
assert len(cases) == 159

if not (RUN / 'run.json').exists():
    source_cases = {c['case_key']: c for c in read_json(SOURCE / 'cases.json')['cases']}
    seeded = []
    for c in cases:
        assert c.to_dict() == source_cases[c.key], c.key
        origin = SOURCE / 'plans' / case_filename(c.key)
        old = read_json(origin)
        plan = validate_plan(old['result'], c.to_dict())
        assert plan == old['result'], c.key
        write_json(RUN / 'plans' / case_filename(c.key), {
            'cache_key': _hash([VERSION, protocol_fingerprint(), judge.identity, c.to_dict()]),
            'created_at': _now(), 'result': plan, 'imported_from': str(origin),
            'source_sha256': _hash(old), 'reuse_scope': 'source_only_plan',
            'raw_model_response': old.get('raw_model_response', ''),
            'usage': old.get('usage'), 'attempts': old.get('attempts', [])})
        seeded.append({'case_key': c.key, 'source_file': str(origin), 'source_sha256': _hash(old)})
    write_json(RUN / 'source_plan_import.json', {
        'source_run': str(SOURCE), 'created_at': _now(), 'plans': len(seeded), 'entries': seeded,
        'note': 'Source text and all frozen plan fields verified identical. Only source plans reused; both candidates assessed afresh.'})
    print(f'Frozen source plans reused: {len(seeded)}; new candidate judgments: all', flush=True)

exit_code = 1
for attempt in range(1, 4):
    print(f'Evaluation pass {attempt}; workers=32; judge={judge.model}; GPU=4', flush=True)
    run_text_pipeline(data_root=BASE, output_root=CFG['paths']['output_root'],
        provider=provider, candidate_file=CFG['paths']['candidate_file'], modes={'t2v'},
        run_id=CFG['run']['run_id'], resume=True, inspect_content=False, max_workers=32)
    summary = read_json(RUN / 'summary.json')
    if summary['error_files'] == 0 and summary['comparison']['complete_real_pairs'] == 159:
        exit_code = 0
        break
    print(f"Incomplete pass {attempt}: pairs={summary['comparison']['complete_real_pairs']} errors={summary['error_files']}", flush=True)

summary = read_json(RUN / 'summary.json')
comparison = read_json(RUN / 'paired_comparison_by_case.json')
counts = Counter()
for row in comparison['cases']:
    official = row['scores']['official-pe']
    log = row['scores']['log-pe']
    decision = 'official-pe' if official > log else 'log-pe' if log > official else 'tie'
    row['strict_winner'] = decision
    counts[decision] += 1
strict = {'official_pe_wins': counts['official-pe'], 'log_pe_wins': counts['log-pe'],
          'ties': counts['tie'], 'compared': len(comparison['cases']),
          'rule': 'higher order-averaged final score wins; exact equality is a tie; no stability filtering'}
summary['comparison']['strict_score_order'] = strict
summary['source_plan_import'] = str(RUN / 'source_plan_import.json')
comparison['scoring_policy'] = SCORING
write_json(RUN / 'summary.json', summary)
write_json(RUN / 'paired_comparison_by_case.json', comparison)
with (RUN / 'summary.md').open('a', encoding='utf-8') as f:
    f.write('\n## Strict comparison by final score\n\n')
    f.write(f"Official PE wins: {counts['official-pe']}; log PE wins: {counts['log-pe']}; exact ties: {counts['tie']}. No stability filtering.\n")
    f.write('Source-only plans were reused from the preceding 159-case run after input validation; all candidate judgments are new.\n')
write_json(RUN / 'execution_result.json', {'exit_code': exit_code, 'finished_at': _now(),
    'complete_pairs': summary['comparison']['complete_real_pairs'], 'error_files': summary['error_files'],
    'strict_score_order': strict})
print(json.dumps({'run_dir': str(RUN), 'strict_score_order': strict,
    'overall': {m: v['overall'] for m,v in summary['methods'].items()}}, ensure_ascii=False), flush=True)
sys.exit(exit_code)
