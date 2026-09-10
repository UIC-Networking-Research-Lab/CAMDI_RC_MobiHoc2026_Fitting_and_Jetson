# Platform-independent accuracy-function fitting

This component generates accuracy-function fitting models for **Communication-Aware Model Distributed Inference via Latent Representation Compression (MobiHoc 2026)**.

**The fitting code is general-purpose and is not limited to Jetson.** Fitting from measured CSVs runs on a regular CPU machine using NumPy, pandas, SciPy, and scikit-learn, with no dependency on L4T, CUDA, pretrained neural weights, or dataset downloads. The resulting accuracy functions can be used by online and offline inference optimizers.

The bundled ResNet and Flan-T5 workflows estimate accuracy from three activation-transfer ratios, `k0`, `k1`, and `k2`. The core `AccuracyEstimator` also accepts feature/accuracy arrays for other models, inferring the feature count from its input. The supplied task-specific CLI commands and recipes use the three-transfer-point schema. These models are fitted from measured accuracy; they do not train ResNet or Flan-T5 weights.

For the supplied three-input polynomial models, `estimator.py` standardizes the inputs, expands them into 19 polynomial features of degree at most three, and fits `Ridge(alpha=0.1)`. An 80/20 split with `random_state=42` supplies the reported validation metrics; the scaler and regressor are then fitted again on all rows for deployment. Predictions are clipped to `[0, 1]`. The pickle stores the regressor, scaler, polynomial transform, feature names, and validation metrics. The optional `linear_monotonic` model uses nonnegative linear regression; the other supported estimator types retain their declared parameters.

`resnet.py` and `flan_t5.py` group observations by compression policy, fit each requested estimator, and write training summaries. They also construct LLM.int8 mappings from measured transfer ratios and outlier settings. `calibrate_resnet.py` and `calibrate_flan_t5.py` collect raw accuracy by running four model partitions locally and compressing activations at the three transfer points. `cli.py` provides the `fit.py` commands.

## Collect accuracy measurements

Run from the repository root on a CPU or CUDA machine, including ordinary workstations. Fresh accuracy collection uses PyTorch and requires the model checkpoint and evaluation dataset; these sweeps measure classification accuracy rather than four-node network timing. Use `--help` for each task's options. `fit.py collect` selects collection mode; use the fitting commands below to train regressors from the resulting CSV.

```bash
# Raw accuracy for all three compressors (single machine; full sweeps can be expensive).
python fit.py collect resnet --device cpu --checkpoint_path models/resnet56-4bfd9763.th --data_root data/cifar10 --download_data --compressor_name all --csv_path outputs/calibration/resnet.csv
python fit.py collect flan-t5 --device cpu --model_name google/flan-t5-base --dataset_path data/sst2_validation.jsonl --compressor_name all --csv_path outputs/calibration/flan_t5.csv
```

Each CSV records compression settings, measured payload ratios and classification accuracy. For TopK/quantization, `k0..k2` use the configured feature ratios rounded to four decimals; for LLM.int8 they use measured compressed/original byte ratios, capped at 1 and rounded to four decimals. All codecs also record measured `tp*_payload_ratio`. LLM.int8 collection writes `<CSV stem>_llmint8_eta_mapping.json`, associating each outlier-fraction vector and precision policy with the measured `k0..k2`. Fitting consumes the raw accuracy CSV; runtime consumes the fitted `.pkl` and mapping JSON. ResNet online inference uses `fp16_int4`, Flan-T5 `fp16_int8`. INT4 has stochastic rounding, so fresh measurements can differ; the fitting package includes the measured CSV used to reconstruct the bundled estimators. The fitting commands and recorded input recipes are below.

The supplied fitting inputs were measured with these settings. Each grid is the Cartesian product over all three transfer points; sample counts are **per configuration**. ResNet uses all 10,000 CIFAR-10 test images; its recorded sample/batch counts correspond to batches of 500 for TopK and 100 for LLM.int8/quantization. Flan-T5 uses the ordered SST-2 validation samples and the default sentiment prompt with `positive`/`negative` verbalizers (token IDs 1465/2841).

| CSV in `data/` | Grid at each transfer point | Samples / batches | Collection options |
| --- | --- | --- | --- |
| `jetson_resnet_3tp_raw_accuracy.csv` (`topk`, 216 rows) | k = 0.125, 0.25, 0.5, 0.75, 0.9, 1 | 10,000 / 20 | `--compressor_name topk --batch_size 500 --max_batches 20` |
| Same combined CSV (`llmint8_fp16_int4`, 27 rows) and `jetson_resnet_3tp_raw_accuracy_llmint8_fp16_int4.csv` | Outliers = 0, 0.5, 1 | 10,000 / 100 | `--compressor_name llmint8 --llm_policies fp16_int4 --llm_outlier_levels 0,0.5,1 --batch_size 100 --max_batches 100` |
| `raw_accuracy_resnet56_llmint8_{fp16_int2,fp16_int4,int8int2,int8int4}.csv` (125 rows each) | Outliers = 0, 0.25, 0.5, 0.75, 1 | 10,000 / 100 | `--compressor_name llmint8 --llm_policies all --llm_outlier_levels 0,0.25,0.5,0.75,1 --batch_size 100 --max_batches 100` |
| `raw_accuracy_resnet56_quantization.csv` (125 rows) | k = 0.0625, 0.125, 0.25, 0.5, 1 | 10,000 / 100 | `--compressor_name quantization --quant_k_levels 0.0625,0.125,0.25,0.5,1 --batch_size 100 --max_batches 100` |
| `raw_accuracy_flan_t5_sst2_3tp_topk.csv` (125 rows) | k = 0.125, 0.25, 0.5, 0.75, 1 | 100 | `--compressor_name topk --k_levels 0.125,0.25,0.5,0.75,1 --max_samples 100` |
| `raw_accuracy_flan_t5_sst2_3tp_llmint8_fp16_int8.csv` (64 rows) | Outliers = 0.01, 0.25, 0.5, 1 | 100 | `--compressor_name llmint8 --llm_policies fp16_int8 --llm_outlier_levels 0.01,0.25,0.5,1 --max_samples 100` |
| `raw_accuracy_flan_t5_sst2_3tp_quantization.csv` (125 rows) | k = 0.0625, 0.125, 0.25, 0.5, 1 | 872 | `--compressor_name quantization --quant_k_levels 0.0625,0.125,0.25,0.5,1 --max_samples 872` |

Use the options in place of `--compressor_name all` in the commands above, and use a new `--csv_path` for each sweep. A collection writes the combined CSV plus one CSV per policy; the recorded 27-row LLM.int8 observations also occur in the 243-row combined ResNet CSV. The fitting recipe retains both copies when forming the 179-row LLM.int8 fit. The CSVs record the grids and sample counts but do not record a random seed or complete collection environment; these commands reproduce the measurement procedure, not the exact recorded stochastic outcomes or timing.

For a short **real-data** collection check, add `--max_batches 1 --batch_size 2 --quant_k_levels 0.25,1 --compressor_name quantization` for ResNet, or `--max_samples 2 --quant_k_levels 0.25,1 --compressor_name quantization` for Flan-T5. These commands require the checkpoint and dataset.

## Generate the supplied models

From the repository root:

```bash
python fit.py bundled --output-dir outputs/fitted
```

This CPU-only command uses the ten measurement CSVs in `data/` and produces six accuracy models, two mappings, training summaries, and `generated.json`. Models are written under `<output>/<task>/<policy>/`; mappings under `<output>/mappings/`. `recipes.json` records the ordered inputs, row counts, and CSV SHA-256 values.

| Model in `models/accuracy_estimators/` | Training measurements | Rows |
| --- | --- | ---: |
| `jetson_resnet_3tp_poly3_flex.pkl` | ResNet56 / CIFAR-10, Top-K; `jetson_resnet_3tp_raw_accuracy.csv`, filtered to `topk` | 216 |
| `jetson_resnet_3tp_quantization_poly3_flex.pkl` | ResNet56 / CIFAR-10, quantization; `raw_accuracy_resnet56_quantization.csv` | 125 |
| `jetson_resnet_3tp_llmint8_fp16_int4_poly3_flex.pkl` | ResNet56 / CIFAR-10, FP16 outliers and INT4 regular values; ordered recipe below | 179 |
| `flan_t5_sst2_3tp_topk_poly3_flex.pkl` | Flan-T5 / SST-2, Top-K; `raw_accuracy_flan_t5_sst2_3tp_topk.csv` | 125 |
| `flan_t5_sst2_3tp_quantization_poly3_flex.pkl` | Flan-T5 / SST-2, quantization; `raw_accuracy_flan_t5_sst2_3tp_quantization.csv` | 125 |
| `flan_t5_sst2_3tp_llmint8_fp16_int8_poly3_flex.pkl` | Flan-T5 / SST-2, FP16 outliers and INT8 regular values; `raw_accuracy_flan_t5_sst2_3tp_llmint8_fp16_int8.csv` | 64 |

Each ResNet measurement evaluates 10,000 CIFAR-10 test images. Flan-T5 Top-K and LLM.int8 measurements use 100 SST-2 validation examples; quantization uses all 872. ResNet Top-K sweeps `[0.125,0.25,0.5,0.75,0.9,1]` at each transfer point, while Flan-T5 Top-K omits `0.9`. Quantization uses `[0.0625,0.125,0.25,0.5,1]`. The 125-row ResNet LLM.int8 sweep uses outlier fractions `[0,0.25,0.5,0.75,1]`; Flan-T5 uses `[0.01,0.25,0.5,1]`. These describe the included observations; regeneration from CSV preserves their actual order and values.

The ResNet LLM.int8 fit concatenates, in order, `jetson_resnet_3tp_raw_accuracy.csv`, `jetson_resnet_3tp_raw_accuracy_llmint8_fp16_int4.csv`, and `raw_accuracy_resnet56_llmint8_fp16_int4.csv`, then selects `llmint8_fp16_int4`. This produces `27 + 27 + 125 = 179` rows. The repeated observations affect their weight in the regression and are deliberately retained.

The ResNet mapping, `raw_accuracy_resnet56_llmint8_eta_mapping.json`, records 125 configurations for each of INT8/INT2, INT8/INT4, FP16/INT2, and FP16/INT4. The Flan-T5 mapping, `raw_accuracy_flan_t5_sst2_3tp_llmint8_eta_mapping.json`, records 64 FP16/INT8 configurations. Their generation pairs `outlier0..2` with measured `k0..2`, removes duplicate pairs, sorts the entries, and writes JSON. At runtime the mapping converts the optimizer's requested ratio into an executable compression setting; the mapping itself is not a learned model.

## Fit newly collected measurements

The [collection commands](#collect-accuracy-measurements) run the four partitions locally, apply compression at each transfer point, compare predictions with labels, and record accuracy and activation-byte counts. LLM.int8 fitting inputs are measured compressed/original byte ratios. Top-K and quantization use the configured `feature_k_values`; actual payload ratios are recorded separately. The fitting ratios are capped at one and rounded to four decimal places. CSV row order, selected samples, compressor settings, and repeated observations all affect the fitted model.

```bash
python fit.py resnet --csv outputs/calibration/resnet.csv --output-dir outputs/resnet-fit --model-types poly3
python fit.py flan-t5 --csv outputs/calibration/flan_t5.csv --output-dir outputs/flan-fit --model-types poly3
python fit.py resnet --csv outputs/calibration/resnet.csv --output-dir outputs/resnet-mapping --mapping-only
```

Required fit columns are `k0,k1,k2,avg_accuracy`; also retain `compressor,compressor_policy,llm_policy` from collection. Mapping generation additionally requires `outlier0,outlier1,outlier2`. ResNet accepts multiple `--csv` paths in the specified order. Flan-T5 accepts one CSV. `--policies` filters the ResNet fit; for Flan-T5 it checks required policies and determines processing order, while retaining other groups.

The output directory must be new or empty. Without `--model-types`, both tasks retain the `poly3,linear_monotonic` defaults. A ResNet Top-K linear fit also emits a compatibility pickle named `jetson_resnet_3tp_poly3_flex.pkl`; the CLI places that export in `compat/` so its misleading historical name cannot overwrite the actual polynomial model. Its estimator type remains `linear_monotonic`.
