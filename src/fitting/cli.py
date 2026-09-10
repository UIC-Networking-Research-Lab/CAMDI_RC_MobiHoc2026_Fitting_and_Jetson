"""Explicit input/output interface for fitting the published accuracy models."""

import argparse
import contextlib
import hashlib
import importlib
import sys
import json
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent


def _module(task):
    if task == 'resnet':
        from . import resnet
        return resnet
    from . import flan_t5
    return flan_t5


@contextlib.contextmanager
def _compatibility_output(module, output_dir):
    """Keep the ResNet compatibility export inside the explicit output directory."""
    if not hasattr(module, 'ESTIMATOR_PATH'):
        yield
        return
    old_path = module.ESTIMATOR_PATH
    module.ESTIMATOR_PATH = str(output_dir / 'compat' / Path(old_path).name)
    try:
        yield
    finally:
        module.ESTIMATOR_PATH = old_path


def fit_csv(task, csv_paths, output_dir, model_types=None, policies=None):
    """Run task-specific fitting with its existing numerical defaults."""
    module = _module(task)
    output_dir = Path(output_dir)
    csv_paths = [str(Path(path)) for path in csv_paths]
    kwargs = {'csv_path': csv_paths if task == 'resnet' else csv_paths[0],
              'output_dir': str(output_dir), 'required_policies': policies}
    if task != 'resnet' and len(csv_paths) != 1:
        raise ValueError('Flan-T5 fitting accepts exactly one CSV.')
    if model_types is not None:
        kwargs['model_types'] = model_types
    if task == 'resnet':
        kwargs['only_required_policies'] = policies is not None
    with _compatibility_output(module, output_dir):
        return module.train_estimators_from_csv(**kwargs)


def generate_mapping(task, csv_paths, output_dir, base_csv):
    """Serialize measured eta/outlier pairs, without fitting or interpolation."""
    import pandas as pd
    frame = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = _module(task)._save_llmint8_eta_mapping(frame, str(Path(output_dir) / base_csv))
    if path is None:
        raise ValueError('No supported LLM.int8 measurement rows in the supplied CSV.')
    return path


def fit_bundled(output_dir):
    """Regenerate all six regressors and both mappings from recorded CSV inputs."""
    output_dir = Path(output_dir)
    recipe = json.loads((PACKAGE / 'recipes.json').read_text(encoding='utf-8'))
    for filename, expected in recipe['data_sha256'].items():
        path = PACKAGE / 'data' / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Calibration CSV checksum mismatch: ' + filename)
    records = []
    for item in recipe['models']:
        # Separate directories retain each task's existing stale-artifact cleanup.
        destination = output_dir / item['task'] / item['policy']
        estimators, summary = fit_csv(
            item['task'], [PACKAGE / 'data' / name for name in item['csv']],
            destination, model_types=recipe['model_type'], policies=[item['policy']],
        )
        if len(summary) != 1 or int(summary.iloc[0]['num_rows']) != item['rows']:
            raise ValueError('Unexpected fitting row count for ' + item['model'])
        records.append({'model': item['model'], 'path': str(destination / item['model']),
                        'rows': item['rows']})
    for item in recipe['mappings']:
        path = generate_mapping(item['task'], [PACKAGE / 'data' / name for name in item['csv']],
                                output_dir / 'mappings', item['base_csv'])
        records.append({'mapping': item['filename'], 'path': path})
    (output_dir / 'generated.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    return records


def _empty_output(path):
    path = Path(path).resolve()
    repository = PACKAGE.parents[1]
    if path == repository or path in repository.parents or (repository / 'models') == path or (repository / 'models') in path.parents:
        raise ValueError('Choose a new output directory outside the bundled models.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('Output directory must be new or empty: ' + str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_accuracy(argv):
    """Measure accuracy only; retain the collectors' options and collect mode."""
    parser = argparse.ArgumentParser(prog='fit.py collect', description='Collect accuracy calibration CSVs.')
    parser.add_argument('task', choices=['resnet', 'flan-t5'])
    if not argv or argv[0] in ('-h', '--help'):
        parser.parse_args(argv)
    route = parser.parse_args(argv[:1])
    options = argv[1:]
    if any(item == '--mode' or item.startswith('--mode=') for item in options):
        parser.error('collect selects --mode collect; use fit.py TASK for fitting')
    module = importlib.import_module('fitting.calibrate_' + route.task.replace('-', '_'))
    saved_argv = sys.argv
    try:
        sys.argv = ['fit.py collect ' + route.task] + options + ['--mode', 'collect']
        return module.main()
    finally:
        sys.argv = saved_argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'collect':
        return collect_accuracy(argv[1:])
    parser = argparse.ArgumentParser(prog='fit.py', description=__doc__,
                                     epilog='Collect new measurements: fit.py collect TASK --help')
    parser.add_argument('task', choices=['bundled', 'resnet', 'flan-t5', 'collect'])
    parser.add_argument('--csv', nargs='+', help='Measurement CSVs, in concatenation order.')
    parser.add_argument('--output-dir', required=True, help='New or empty directory for generated files.')
    parser.add_argument('--model-types', help='Comma-separated types; retains task defaults when omitted.')
    parser.add_argument('--policies', help='Comma-separated compressor policies; ResNet filters, Flan-T5 checks and orders.')
    parser.add_argument('--mapping-only', action='store_true', help='Generate eta mapping from measured outlier/byte ratios.')
    args = parser.parse_args(argv)
    if args.task != 'bundled' and not args.csv:
        parser.error('--csv is required for task-specific fitting.')
    if args.task == 'bundled' and (args.csv or args.model_types or args.policies or args.mapping_only):
        parser.error('bundled uses the fixed recipes; do not pass fitting overrides.')
    output = _empty_output(args.output_dir)
    if args.task == 'bundled':
        records = fit_bundled(output)
        print('Generated {} models/mappings in {}'.format(len(records), output))
        return records
    if args.mapping_only:
        return generate_mapping(args.task, args.csv, output, Path(args.csv[0]).name)
    policies = [value.strip() for value in args.policies.split(',') if value.strip()] if args.policies else None
    return fit_csv(args.task, args.csv, output, args.model_types, policies)


if __name__ == '__main__':
    main()
