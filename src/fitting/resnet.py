"""Fit per-compressor accuracy models and measured eta-to-codec mappings."""
import glob
import json
import logging
import os
from collections import OrderedDict
import pandas as pd
from . import estimator as accuracy_estimator_flex
from .estimator import AccuracyEstimator

ESTIMATOR_MODEL_DIR = 'acc_estimatior_fitting_models'


ESTIMATOR_MODEL_TYPE = 'linear_monotonic'


ESTIMATOR_PATH = os.path.join(ESTIMATOR_MODEL_DIR, 'jetson_resnet_3tp_poly3_flex.pkl')


RAW_ACCURACY_CSV = os.path.join(ESTIMATOR_MODEL_DIR, 'raw_accuracy_resnet56_quantization.csv')


def _discover_supported_model_types():
    model_builders = getattr(accuracy_estimator_flex, '_MODEL_BUILDERS', None)
    if isinstance(model_builders, dict) and model_builders:
        return list(model_builders.keys())
    return ['poly2', 'poly3', 'gbm', 'rf', 'mlp', 'mlp_small']


SUPPORTED_MODEL_TYPES = _discover_supported_model_types()


LLMINT8_POLICY_SPECS = OrderedDict([
    ('fp16_int2', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int2',
    }),
    ('fp16_int4', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int4',
    }),
    ('INT8INT4', {
        'outlier_precision': 'int8',
        'regular_precision': 'int4',
    }),
    ('INT8INT2', {
        'outlier_precision': 'int8',
        'regular_precision': 'int2',
    }),
])


DEFAULT_ESTIMATOR_MODEL_TYPES = 'poly3,linear_monotonic'


def _resolve_model_types(model_types_arg):
    if model_types_arg.strip().lower() == 'all':
        return list(SUPPORTED_MODEL_TYPES)

    model_types = [item.strip() for item in model_types_arg.split(',') if item.strip()]
    invalid = [item for item in model_types if item not in SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError('Unsupported model types: {}. Choose from {}'.format(invalid, SUPPORTED_MODEL_TYPES))
    if not model_types:
        raise ValueError('At least one model type must be provided.')
    return model_types


def _sanitize_tag(text):
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else '_')
    return ''.join(chars).strip('_')


def _ordered_policy_groups(df, required_policies=None):
    present = {}
    for group_name, group_df in df.groupby('compressor_policy', sort=False, dropna=False):
        present[str(group_name)] = group_df.copy()

    ordered = []
    seen = set()
    for policy in required_policies or []:
        if policy in present:
            ordered.append((policy, present[policy]))
            seen.add(policy)

    for policy, group_df in present.items():
        if policy not in seen:
            ordered.append((policy, group_df))
    return ordered


def _summary_output_path(output_dir, compressor_policy):
    return os.path.join(
        output_dir,
        'resnet_training_summary_{}.csv'.format(_sanitize_tag(compressor_policy)),
    )


def _combined_summary_output_path(output_dir):
    return os.path.join(output_dir, 'resnet_training_summary.csv')


def _cleanup_removed_llm_artifacts(output_dir, model_types, active_policies):
    legacy_policies = [
        'llmint8_fp16_int8',
        'llmint8_fp16_int4',
        'llmint8_fp16_int2',
        'llmint8_fp32_int8',
        'llmint8_INT8INT4',
        'llmint8_INT8INT2',
        'llmint8_int8int4',
        'llmint8_int8int2',
    ]
    active_set = set(active_policies or [])
    active_sanitized_set = {_sanitize_tag(policy) for policy in active_set}
    for compressor_policy in legacy_policies:
        if compressor_policy in active_set or _sanitize_tag(compressor_policy) in active_sanitized_set:
            continue

        summary_path = _summary_output_path(output_dir, compressor_policy)
        if os.path.exists(summary_path):
            os.remove(summary_path)
            logging.info('Removed stale summary file %s', summary_path)

        for model_type in model_types:
            model_path = _model_output_path(output_dir, model_type, compressor_policy=compressor_policy)
            if os.path.exists(model_path):
                os.remove(model_path)
                logging.info('Removed stale model file %s', model_path)


def _load_existing_accuracy_df(csv_path):
    if isinstance(csv_path, (list, tuple)):
        files = [str(path) for path in csv_path if str(path).strip()]
        if not files:
            return None
        frames = [pd.read_csv(path) for path in files]
        return pd.concat(frames, ignore_index=True) if frames else None

    if not os.path.exists(csv_path):
        matched_files = sorted(glob.glob(str(csv_path)))
        if matched_files:
            frames = [pd.read_csv(path) for path in matched_files]
            return pd.concat(frames, ignore_index=True) if frames else None
        return None

    if os.path.isdir(csv_path):
        files = sorted(glob.glob(os.path.join(csv_path, '*.csv')))
        if not files:
            return None
        frames = [pd.read_csv(path) for path in files]
        return pd.concat(frames, ignore_index=True) if frames else None

    return pd.read_csv(csv_path)


def _llmint8_mapping_output_path(base_csv_path):
    directory = os.path.dirname(base_csv_path)
    stem, _ = os.path.splitext(os.path.basename(base_csv_path))
    return os.path.join(directory, '{}_llmint8_eta_mapping.json'.format(stem))


def _save_llmint8_eta_mapping(df, base_csv_path):
    if df is None or df.empty:
        return None

    llmint8_df = df.copy()
    if 'compressor' in llmint8_df.columns:
        llmint8_df = llmint8_df[llmint8_df['compressor'].astype(str) == 'llmint8'].copy()
    elif 'compressor_policy' in llmint8_df.columns:
        llmint8_df = llmint8_df[
            llmint8_df['compressor_policy'].astype(str).str.startswith('llmint8_')
        ].copy()

    if llmint8_df.empty or 'llm_policy' not in llmint8_df.columns:
        return None

    policies = []
    for llm_policy, group_df in llmint8_df.groupby('llm_policy', sort=False, dropna=False):
        llm_policy = str(llm_policy).strip()
        if not llm_policy:
            continue

        policy_spec = LLMINT8_POLICY_SPECS.get(llm_policy)
        if policy_spec is None:
            continue

        entries = []
        seen = set()
        for _, row in group_df.iterrows():
            outlier_values = [float(row['outlier0']), float(row['outlier1']), float(row['outlier2'])]
            feature_k_values = [float(row['k0']), float(row['k1']), float(row['k2'])]
            key = (
                tuple(round(value, 8) for value in outlier_values),
                tuple(round(value, 8) for value in feature_k_values),
            )
            if key in seen:
                continue
            seen.add(key)
            entries.append({
                'outlier_values': outlier_values,
                'feature_k_values': feature_k_values,
            })

        entries.sort(key=lambda item: (
            tuple(float(value) for value in item['outlier_values']),
            tuple(float(value) for value in item['feature_k_values']),
        ))
        policies.append({
            'llm_policy': llm_policy,
            'outlier_precision': policy_spec['outlier_precision'],
            'regular_precision': policy_spec['regular_precision'],
            'llmint8_eta_to_codec_mapping': {
                '_comment_reason': (
                    'LLMint8 optimization is carried out in feature-k / eta space, '
                    'but real ResNet execution requires feasible outlier configurations. '
                    'This mapping converts solver eta to the nearest executable llmint8 configuration.'
                ),
                '_comment_usage': (
                    'Used only when codec_name is llmint8 during true local inference. '
                    'Ignored for topk.'
                ),
                'entries': entries,
            },
        })

    if not policies:
        return None

    payload = {
        'compressor': 'llmint8',
        'num_links': 3,
        'policies': policies,
    }
    mapping_path = _llmint8_mapping_output_path(base_csv_path)
    with open(mapping_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return mapping_path


def _model_output_path(output_dir, model_type, compressor_policy=None):
    if not compressor_policy or compressor_policy == 'topk':
        return os.path.join(output_dir, 'jetson_resnet_3tp_{}_flex.pkl'.format(model_type))

    return os.path.join(
        output_dir,
        'jetson_resnet_3tp_{}_{}_flex.pkl'.format(
            _sanitize_tag(compressor_policy),
            model_type,
        ),
    )


def train_estimators_from_csv(csv_path=RAW_ACCURACY_CSV,
                              output_dir=ESTIMATOR_MODEL_DIR,
                              model_types=DEFAULT_ESTIMATOR_MODEL_TYPES,
                              required_policies=None,
                              only_required_policies=False,
                              active_policies=None):
    resolved_model_types = _resolve_model_types(model_types)
    df = _load_existing_accuracy_df(csv_path)
    if df is None:
        raise FileNotFoundError('No raw accuracy CSV files found at: {}'.format(csv_path))
    if df.empty:
        raise ValueError('Raw accuracy CSV is empty: {}'.format(csv_path))
    if 'compressor_policy' not in df.columns:
        df['compressor_policy'] = df.get('compressor', 'topk')
    if 'llm_policy' not in df.columns:
        df['llm_policy'] = ''
    if required_policies is not None:
        missing = [policy for policy in required_policies if policy not in set(df['compressor_policy'].astype(str))]
        if missing:
            raise ValueError(
                'Raw accuracy data is incomplete. Missing compressor_policy rows: {}'.format(missing)
            )
    if required_policies is not None and only_required_policies:
        df = df[df['compressor_policy'].astype(str).isin(required_policies)].copy()

    rows = []
    summary_rows_by_policy = OrderedDict()
    estimators = {}
    os.makedirs(output_dir, exist_ok=True)
    combined_summary_path = _combined_summary_output_path(output_dir)
    if os.path.exists(combined_summary_path):
        os.remove(combined_summary_path)
        logging.info('Removed legacy combined summary file %s', combined_summary_path)
    _cleanup_removed_llm_artifacts(
        output_dir=output_dir,
        model_types=resolved_model_types,
        active_policies=active_policies if active_policies is not None else required_policies,
    )

    grouped_frames = _ordered_policy_groups(df, required_policies=required_policies)

    for compressor_policy, group_df in grouped_frames:
        compressor_name = str(group_df['compressor'].iloc[0]) if 'compressor' in group_df.columns else compressor_policy
        llm_policy = str(group_df['llm_policy'].iloc[0]) if 'llm_policy' in group_df.columns else ''
        X = group_df[['k0', 'k1', 'k2']].astype(float).values
        y = group_df['avg_accuracy'].astype(float).values

        for model_type in resolved_model_types:
            logging.info(
                'Training AccuracyEstimator model_type=%s compressor_policy=%s samples=%d',
                model_type,
                compressor_policy,
                len(group_df),
            )
            estimator = AccuracyEstimator(model_type)
            estimator.feature_names = ['k0', 'k1', 'k2']
            estimator.train(X, y)

            model_path = _model_output_path(output_dir, model_type, compressor_policy=compressor_policy)
            estimator.save(model_path)
            estimators[(compressor_policy, model_type)] = estimator

            metrics = dict(estimator.metrics)
            rows.append({
                'compressor': compressor_name,
                'compressor_policy': compressor_policy,
                'llm_policy': llm_policy,
                'model_type': model_type,
                'model_path': model_path,
                'num_rows': len(group_df),
                'n_features': estimator.n_features,
                'feature_names': ','.join(estimator.feature_names or []),
                **metrics,
            })
            summary_rows_by_policy.setdefault(compressor_policy, []).append(rows[-1])

            if compressor_policy == 'topk' and model_type == ESTIMATOR_MODEL_TYPE:
                estimator.save(ESTIMATOR_PATH)
                logging.info('Saved compatibility copy to %s', ESTIMATOR_PATH)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and 'test_rmse' in summary_df.columns:
        summary_df = summary_df.sort_values(
            ['compressor_policy', 'test_rmse', 'test_r2'],
            ascending=[True, True, False],
        )
    for compressor_policy, policy_rows in summary_rows_by_policy.items():
        policy_df = pd.DataFrame(policy_rows)
        if not policy_df.empty and 'test_rmse' in policy_df.columns:
            policy_df = policy_df.sort_values(['test_rmse', 'test_r2'], ascending=[True, False])
        summary_path = _summary_output_path(output_dir, compressor_policy)
        policy_df.to_csv(summary_path, index=False)
        logging.info('Saved training summary to %s', summary_path)
    return estimators, summary_df
