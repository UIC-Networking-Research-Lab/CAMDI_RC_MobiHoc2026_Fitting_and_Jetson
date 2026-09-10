"""Fit per-compressor accuracy models and measured eta-to-codec mappings."""
import glob
import json
import logging
import os
from collections import OrderedDict
import pandas as pd
from . import estimator as accuracy_estimator_flex
from .estimator import AccuracyEstimator

SUPPORTED_MODEL_TYPES = ['linear_monotonic', 'poly3']


DEFAULT_MODEL_TYPES = 'poly3,linear_monotonic'


ESTIMATOR_MODEL_DIR = 'acc_estimatior_fitting_models'


RAW_ACCURACY_CSV = os.path.join(ESTIMATOR_MODEL_DIR, 'raw_accuracy_flan_t5_sst2_3tp_quantization.csv')


LLMINT8_POLICY_SPECS = OrderedDict([
    ('fp16_int8', {
        'outlier_precision': 'fp16',
        'regular_precision': 'int8',
    }),
])


def _sanitize_tag(text):
    chars = []
    for ch in str(text).strip().lower():
        chars.append(ch if ch.isalnum() else '_')
    return ''.join(chars).strip('_')


def _parse_model_types(model_types_arg):
    names = [item.strip() for item in str(model_types_arg).split(',') if item.strip()]
    if not names:
        raise ValueError('At least one model type must be provided.')
    invalid = [item for item in names if item not in SUPPORTED_MODEL_TYPES]
    if invalid:
        raise ValueError('Unsupported model types: {}. Choose from {}'.format(invalid, SUPPORTED_MODEL_TYPES))
    return list(OrderedDict.fromkeys(names))


def _ordered_policy_groups(df, required_policies=None):
    present = OrderedDict()
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
        'flan_t5_sst2_training_summary_{}.csv'.format(_sanitize_tag(compressor_policy)),
    )


def _combined_summary_output_path(output_dir):
    return os.path.join(output_dir, 'flan_t5_sst2_training_summary.csv')


def _model_output_path(output_dir, model_type, compressor_policy):
    return os.path.join(
        output_dir,
        'flan_t5_sst2_3tp_{}_{}_flex.pkl'.format(
            _sanitize_tag(compressor_policy),
            model_type,
        ),
    )


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
                    'but real Flan-T5 execution requires feasible outlier configurations. '
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


def train_estimators_from_csv(csv_path=RAW_ACCURACY_CSV,
                              output_dir=ESTIMATOR_MODEL_DIR,
                              model_types=DEFAULT_MODEL_TYPES,
                              required_policies=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError('Raw accuracy CSV not found: {}'.format(csv_path))

    resolved_model_types = _parse_model_types(model_types)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError('Raw accuracy CSV is empty: {}'.format(csv_path))
    if 'compressor_policy' not in df.columns:
        raise ValueError('Missing required column compressor_policy in {}'.format(csv_path))

    if required_policies is None:
        required_policies = list(OrderedDict.fromkeys(df['compressor_policy'].astype(str).tolist()))
    else:
        present_policies = set(df['compressor_policy'].astype(str).tolist())
        missing_policies = [policy for policy in required_policies if policy not in present_policies]
        if missing_policies:
            raise ValueError('Raw accuracy CSV is missing required policies: {}'.format(missing_policies))

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    estimators = {}

    for compressor_policy, group_df in _ordered_policy_groups(df, required_policies=required_policies):
        X = group_df[['k0', 'k1', 'k2']].astype(float).values
        y = group_df['avg_accuracy'].astype(float).values
        compressor_name = str(group_df['compressor'].iloc[0])
        llm_policy = str(group_df['llm_policy'].iloc[0]) if 'llm_policy' in group_df.columns else ''

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

            model_path = _model_output_path(output_dir, model_type, compressor_policy)
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

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty and 'test_rmse' in summary_df.columns:
        summary_df = summary_df.sort_values(
            ['compressor_policy', 'test_rmse', 'test_r2'],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    combined_summary_path = _combined_summary_output_path(output_dir)
    summary_df.to_csv(combined_summary_path, index=False)
    logging.info('Saved combined training summary to %s', combined_summary_path)

    for compressor_policy, group_df in _ordered_policy_groups(summary_df, required_policies=required_policies):
        summary_path = _summary_output_path(output_dir, compressor_policy)
        group_df.to_csv(summary_path, index=False)
        logging.info('Saved training summary to %s', summary_path)

    return estimators, summary_df
