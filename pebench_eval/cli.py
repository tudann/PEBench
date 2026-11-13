from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .candidates import load_candidate_rows, template_rows
from .config import filtered_cases, load_config, provider_from_config
from .dataset import load_cases, validate_assets
from .providers import provider_from_env
from .report import render_markdown, summarize
from .runner import run_pipeline
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
    prepare = sub.add_parser('prepare', help='freeze selected cases and cache evidence')
    _common(prepare); _selection(prepare)
    prepare.add_argument('--output-root'); prepare.add_argument('--run-id')
    prepare.add_argument('--provider', choices=['mock', 'real']); prepare.add_argument('--no-decode', action='store_true', default=None)
    evaluate = sub.add_parser('evaluate', help='evaluate candidate PE prompts')
    _common(evaluate); _selection(evaluate)
    evaluate.add_argument('--candidate-file'); evaluate.add_argument('--output-root'); evaluate.add_argument('--run-id')
    evaluate.add_argument('--provider', choices=['mock', 'real']); evaluate.add_argument('--no-decode', action='store_true', default=None)
    evaluate.add_argument('--no-resume', action='store_true', default=None)
    smoke = sub.add_parser('self-test', help='run offline protocol/checkpoint tests')
    _common(smoke)
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
    if args.command in {'validate', 'make-template', 'prepare', 'evaluate'}:
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
        output_root = require_data_disk_path(args.output_root or paths.get('output_root') or DEFAULT_OUTPUT_ROOT)
        run_dir = run_pipeline(data_root=data_root, output_root=output_root, provider=provider,
            candidate_file=candidate if args.command == 'evaluate' else None,
            case_filter_keys={c.key for c in selected}, run_id=args.run_id or run.get('run_id'),
            resume=False if getattr(args, 'no_resume', None) else run.get('resume', True), inspect_content=inspect)
        print(run_dir)
        return 0
    if args.command == 'self-test':
        print(json.dumps(run_self_test(data_root), ensure_ascii=False, indent=2)); return 0
    if args.command == 'summarize':
        run_dir = require_data_disk_path(args.run_dir)
        result = summarize(run_dir)
        Path(run_dir, 'summary.md').write_text(render_markdown(result), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    return 2

if __name__ == '__main__': sys.exit(main())
