from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .candidates import load_candidate_rows, template_rows
from .config import filtered_cases, load_config, provider_from_config
from .dataset import DATASETS, canonical_json, load_cases, sha256_bytes, validate_assets
from .providers import provider_from_env
from .report import render_markdown, summarize
from .runner import run_pipeline
from .schema import PROMPT_FACTS_VERSION
from .prompt_facts import validate_prompt_facts
from .smoke import run_self_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / 'data'
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'runs'
DATA_DISK_ROOT = Path('/pfs/pfs-spMCeu').resolve()

def require_data_disk_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if not DATA_DISK_ROOT.is_dir() or not path.is_relative_to(DATA_DISK_ROOT):
        raise ValueError(f'output path must be on the configured data disk under {DATA_DISK_ROOT}: {path}')
    return path

def _common(parser):
    parser.add_argument('--config', help='YAML configuration file')
    parser.add_argument('--data-root', help='PEBench data directory')

def _selection(parser):
    parser.add_argument('--mode', action='append', choices=['t2v', 'i2v', 'r2v'], help='restrict cases; repeatable')
    parser.add_argument('--limit', type=int, help='select at most N cases after filtering')

def build_parser():
    p = argparse.ArgumentParser(prog='pebench', description='Standalone PEBench direct prompt evaluation pipeline')
    sub = p.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate', help='validate benchmark JSON and referenced assets')
    _common(validate); _selection(validate)
    validate.add_argument('--no-decode', action='store_true', default=None)
    template = sub.add_parser('make-template', help='write an empty PE candidate JSONL template')
    _common(template); _selection(template)
    template.add_argument('--output'); template.add_argument('--method')
    facts = sub.add_parser('extract-facts', help='extract a shared source-prompt requirements checklist')
    _common(facts); _selection(facts)
    facts.add_argument('--force', action='store_true', help='re-extract even when a matching cached checklist exists')
    prepare = sub.add_parser('prepare', help='freeze selected cases and cache evidence')
    _common(prepare); _selection(prepare)
    prepare.add_argument('--output-root'); prepare.add_argument('--run-id')
    prepare.add_argument('--provider', choices=['mock', 'real']); prepare.add_argument('--no-decode', action='store_true', default=None)
    evaluate = sub.add_parser('evaluate', help='evaluate candidate PE prompts')
    _common(evaluate); _selection(evaluate)
    evaluate.add_argument('--candidate-file'); evaluate.add_argument('--output-root'); evaluate.add_argument('--run-id')
    evaluate.add_argument('--provider', choices=['mock', 'real']); evaluate.add_argument('--no-decode', action='store_true', default=None)
    prepare.add_argument('--protocol', choices=['legacy', 'text-v3'])
    evaluate.add_argument('--protocol', choices=['legacy', 'text-v3'])
    evaluate.add_argument('--checkpoint-source', action='append', help='import compatible text-v3 checkpoints into a new run')
    evaluate.add_argument('--no-resume', action='store_true', default=None)
    smoke = sub.add_parser('self-test', help='run offline protocol/checkpoint tests')
    _common(smoke)
    checks = sub.add_parser('rubric-check', help='validate text-v3 with controlled text mutations, without video or human review')
    _common(checks)
    checks.add_argument('--output-root'); checks.add_argument('--run-id'); checks.add_argument('--limit', type=int)
    summary = sub.add_parser('summarize', help='rebuild summary from a completed run')
    summary.add_argument('--run-dir', required=True)
    return p

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if getattr(args, 'config', None) else {}
    paths, selection, run = cfg.get('paths', {}), dict(cfg.get('selection', {})), cfg.get('run', {})
    data_root = getattr(args, 'data_root', None) or paths.get('data_root') or str(DEFAULT_DATA_ROOT)
    if getattr(args, 'mode', None) is not None: selection['modes'] = args.mode
    if getattr(args, 'limit', None) is not None: selection['limit'] = args.limit
    if args.command == 'rubric-check':
        from .text_checks import run_checks
        provider = provider_from_config(cfg.get('provider', {})) if cfg else provider_from_env('real')
        output_root = require_data_disk_path(args.output_root or paths.get('output_root') or DEFAULT_OUTPUT_ROOT)
        if args.limit is not None and not 1 <= args.limit <= 7:
            raise ValueError('rubric-check --limit must be in [1,7]')
        result = run_checks(output_root=output_root, run_id=args.run_id or 'text-v3-check-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),
                            provider=provider, max_workers=min(run.get('max_workers', 1), 4), limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['all_passed'] else 1
    if args.command in {'validate', 'make-template', 'extract-facts', 'prepare', 'evaluate'}:
        cases = load_cases(data_root)
        selected = filtered_cases(cases, selection)
        inspect = False if getattr(args, 'no_decode', None) else run.get('inspect_content', True)
        if args.command == 'validate':
            manifest = validate_assets(selected, inspect_content=inspect)
            print(json.dumps({'status': 'valid', 'all_cases': len(cases), 'selected_cases': len(selected), 'assets': len(manifest)}, ensure_ascii=False))
            return 0
        if args.command == 'make-template':
            output = args.output or paths.get('template_file')
            if not output: raise ValueError('provide --output or paths.template_file')
            output_path = require_data_disk_path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(template_rows(selected, args.method or run.get('method', 'method-name')), encoding='utf-8')
            print(f'wrote {len(selected)} candidate rows to {output_path}')
            return 0
        candidate = getattr(args, 'candidate_file', None) or paths.get('candidate_file')
        if args.command == 'evaluate' and not candidate: raise ValueError('provide --candidate-file or paths.candidate_file')
        if args.command == 'evaluate' and cfg:
            candidate_keys = {f"{row['mode']}:{row['capid']}" for row in load_candidate_rows(candidate)}
            missing = {c.key for c in selected} - candidate_keys
            if missing: raise ValueError(f'candidate file is missing selected cases: {sorted(missing)}')
        explicit_provider = getattr(args, 'provider', None)
        if explicit_provider:
            provider = provider_from_env(explicit_provider)
        elif cfg:
            provider = provider_from_config(cfg.get('provider', {}))
        else:
            provider = provider_from_env('mock')
        if args.command == 'extract-facts':
            if getattr(provider, 'is_mock', False):
                raise ValueError('extract-facts requires a real provider; mock output is not source evidence')
            extractor = provider.judge_provider() if hasattr(provider, 'judge_provider') else provider
            if not hasattr(extractor, 'extract_prompt_requirements'):
                raise ValueError('configured provider does not support source-prompt fact extraction')
            data_path = Path(data_root).resolve() / DATASETS['t2v']
            rows = json.loads(data_path.read_text(encoding='utf-8'))
            rows_by_id = {str(row.get('capid')): row for row in rows}
            identity = getattr(extractor, 'identity', f'{extractor.name}:{extractor.version}')
            model_name = getattr(extractor, 'model', extractor.name)
            max_workers = run.get('max_workers', 1)
            if type(max_workers) is not int or not 1 <= max_workers <= 128:
                raise ValueError('run.max_workers must be an integer in [1, 128]')
            errors = []
            updated = skipped = 0
            total = len(selected)
            completed = 0
            state_lock = threading.Lock()

            def reuse_cached(row, case, prompt_hash) -> bool:
                if getattr(args, 'force', False) or row.get('source_facts') is None:
                    return False
                old_meta = row.get('source_facts_meta') or {}
                same_source = (old_meta.get('prompt_version') == PROMPT_FACTS_VERSION
                               and old_meta.get('source_prompt_sha256') == prompt_hash)
                same_model = (old_meta.get('provider_identity') == identity
                              or old_meta.get('provider_model') == model_name)
                if not (same_source and same_model):
                    return False
                try:
                    validate_prompt_facts(row['source_facts'], case)
                except ValueError:
                    return False
                return True

            def persist(row, facts, prompt_hash, usage) -> None:
                row['source_facts'] = facts
                row['source_facts_meta'] = {
                    'prompt_version': PROMPT_FACTS_VERSION,
                    'provider_identity': identity,
                    'provider_model': model_name,
                    'source_prompt_sha256': prompt_hash,
                    'extracted_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                    'usage': usage,
                }
                temporary = data_path.with_suffix(data_path.suffix + '.tmp')
                temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
                temporary.replace(data_path)

            pending = []
            for case in selected:
                row = rows_by_id.get(case.capid)
                if row is None:
                    errors.append({'case_key': case.key, 'error': 'case is missing from the source JSON'})
                    completed += 1
                    print(f'[{completed}/{total}] ERROR {case.key}: missing from source JSON')
                    sys.stdout.flush()
                    continue
                prompt_hash = sha256_bytes(case.prompt.encode('utf-8'))
                if reuse_cached(row, case, prompt_hash):
                    skipped += 1
                    completed += 1
                    print(f'[{completed}/{total}] cached {case.key}')
                    sys.stdout.flush()
                    continue
                pending.append((case, row, prompt_hash))

            def extract_one(case):
                facts = extractor.extract_prompt_requirements(case)
                usage = getattr(getattr(extractor, '_request_local', None), 'usage', None)
                return facts, usage

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(extract_one, case): (case, row, prompt_hash) for case, row, prompt_hash in pending}
                for future in as_completed(futures):
                    case, row, prompt_hash = futures[future]
                    try:
                        facts, usage = future.result()
                    except Exception as exc:
                        with state_lock:
                            errors.append({'case_key': case.key, 'error_type': type(exc).__name__, 'error': str(exc)})
                            completed += 1
                            print(f'[{completed}/{total}] ERROR {case.key}: {type(exc).__name__}')
                            sys.stdout.flush()
                        continue
                    with state_lock:
                        persist(row, facts, prompt_hash, usage)
                        updated += 1
                        completed += 1
                        print(f'[{completed}/{total}] extracted {case.key}: {len(facts["requirements"])} requirements')
                        sys.stdout.flush()
            error_path = Path(data_root).resolve() / 'source_fact_extraction_errors.json'
            if errors:
                error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2) + chr(10), encoding='utf-8')
            else:
                error_path.unlink(missing_ok=True)
            print(json.dumps({'selected': total, 'extracted': updated, 'cached': skipped,
                              'errors': len(errors), 'data_file': str(data_path)}, ensure_ascii=False))
            return 1 if errors else 0
        output_root = require_data_disk_path(args.output_root or paths.get('output_root') or DEFAULT_OUTPUT_ROOT)
        run_dir = run_pipeline(data_root=data_root, output_root=output_root, provider=provider,
            candidate_file=candidate if args.command == 'evaluate' else None,
            case_filter_keys={c.key for c in selected}, run_id=args.run_id or run.get('run_id'),
            resume=False if getattr(args, 'no_resume', None) else run.get('resume', True), inspect_content=inspect,
            max_workers=run.get('max_workers', 1), protocol=getattr(args, 'protocol', None) or run.get('protocol', 'legacy'),
            checkpoint_source=getattr(args, 'checkpoint_source', None) or run.get('checkpoint_source'))
        print(run_dir)
        metadata = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
        return 1 if metadata.get('protocol') == 'text-v3' and metadata.get('error_count') else 0
    if args.command == 'self-test':
        print(json.dumps(run_self_test(data_root), ensure_ascii=False, indent=2)); return 0
    if args.command == 'summarize':
        run_dir = require_data_disk_path(args.run_dir)
        result = summarize(run_dir)
        Path(run_dir, 'summary.md').write_text(render_markdown(result), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    return 2

if __name__ == '__main__': sys.exit(main())
