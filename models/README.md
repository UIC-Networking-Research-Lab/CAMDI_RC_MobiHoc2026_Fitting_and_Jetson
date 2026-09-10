# Models

This directory contains the neural-network checkpoint and the accuracy models
used by the online optimizer.

| File | Purpose |
| --- | --- |
| `resnet56-4bfd9763.th` | ResNet-56 weights trained on CIFAR-10, used for image classification. |
| `accuracy_estimators/jetson_resnet_3tp*.pkl` | Three ResNet accuracy regressors: Top-K, quantization, and LLM.int8 FP16/INT4. The filename without a compressor suffix is the Top-K model. |
| `accuracy_estimators/flan_t5_sst2_3tp*.pkl` | Three Flan-T5/SST-2 accuracy regressors: Top-K, quantization, and LLM.int8 FP16/INT8. |
| `accuracy_estimators/*_eta_mapping.json` | Measured mappings from transfer ratios to LLM.int8 outlier settings, one for each task. |

The six PKLs predict accuracy from three compression ratios. They are fitted
with degree-three polynomial regression from the recorded calibration CSVs.
The fitting workflow is platform-independent and can be used in CPU or GPU
environments beyond Jetson. Its accuracy-function models support online and
offline inference experiments.
The mappings are generated from measured ratios and outlier settings.

Rebuild all six PKLs and both mappings from the repository root:

```bash
python fit.py bundled --output-dir outputs/fitted
```

See the [fitting instructions](../src/fitting/README.md) for each
model's inputs. Flan-T5 neural weights are downloaded to the Hugging Face cache
by `python prepare.py assets --download`.
