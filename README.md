# Model Distributed Inference Experiments

Code accompanying the paper **Communication-Aware Model Distributed Inference via Latent Representation Compression (MobiHoc 2026)**.

**For the offline experiment code, please refer to our main GitHub repository.**

This repository contains the paper's **accuracy-function model fitting**, **Jetson
online experiments**, and **performance trace collection** code. Fitting is a
general-purpose, platform-independent workflow; the online experiments and trace
collectors implement the Jetson evaluation setup.

| Workflow | Entry point | Purpose |
| --- | --- | --- |
| **Inference** | `run.py` | Run **Jetson online tests**, using accuracy models and compression mappings produced by Fitting. |
| **Trace** | `trace.py` | Measure the Jetson network environment and save performance traces, primarily for **offline tests** in the separate `Inference_Optimizer` project. |
| **Fitting** | `fit.py` | Generate accuracy-function fitting models from measured data for online and offline inference experiments; usable on platforms beyond Jetson. |

```text
src/
  jetson_inference/   Jetson online inference and shared runtime components
  traces/             Performance trace generation
  fitting/            Platform-independent accuracy fitting and measurement CSVs
models/               Neural weights, fitted accuracy models, and mappings
data/                 Evaluation datasets
```

**Jetson environment (online experiments and trace collection).** The recorded
platform is **NVIDIA Jetson Orin Nano**, **L4T r36.4.0**, and **Python 3.12**.
[requirements.txt](requirements.txt) records its Python dependencies, including
PyTorch 2.7.0 and torchvision 0.22.0. Use Jetson-compatible PyTorch/torchvision
builds and the same environment on all nodes. Run these commands from the
repository root when preparing the Jetson experiments:

```bash
python -m pip install -r requirements.txt
python prepare.py assets
# Download/cache CIFAR-10 and Flan-T5 weights and tokenizer on each node.
python prepare.py assets --download
```

**Fitting is not restricted to Jetson.** The workflow can be used in CPU or
GPU environments on workstations, servers, or Jetson devices. See the
[Fitting section](#3-fitting-platform-independent-accuracy-function-models) for
the separate fitting setup and commands.

The [models](models/README.md) and [data](data/README.md) READMEs describe the
required files and how to obtain them. Install `datasets` for SST-2 export or
GLUE downloads, and `sentencepiece` if the tokenizer backend requires it.
Append `--help` to any entry point or task command for its options.

## 1. Inference: Jetson online tests

**Inference is the on-device online test implementation.** It runs ResNet-56 on
CIFAR-10, Flan-T5-base on SST-2, or both tasks asynchronously across four Jetson
nodes. Each node executes one model partition. Activations travel through
A → B → C → D over HTTP, and D returns predictions to the controller on A.
The online runner measures the current network and updates compression and
scheduling decisions during execution.

**Online inference uses the outputs of Fitting.** In `fitting_model` mode, the
optimizer loads the fitted `.pkl` accuracy models to estimate the effect of
compression at the three transfer points. LLM.int8 profiles also use the mapping
JSON files to convert requested compression ratios into outlier settings. These
files are bundled in `models/accuracy_estimators/`, so the supplied configuration
can run without repeating the fitting process. The Fitting section below explains
how to regenerate them.

Default node addresses are `192.168.1.20` through `192.168.1.23`, with ports
`52000` through `52003`. For another topology, set `JETSON_NODE_A_IP` through
`JETSON_NODE_D_IP` and, if needed, `JETSON_NODE_A_PORT` through
`JETSON_NODE_D_PORT` consistently on all four nodes. Start B, C, and D first,
then A:

```bash
# On B, C, and D: substitute the corresponding node letter.
python run.py resnet --node B --device cuda
# On A:
python run.py resnet --node A --device cuda --accuracy_estimator_modes fitting_model --max_total_batches 10
```

Use `run.py flan-t5` for Flan-T5/SST-2 or `run.py multi` for the two-task
experiment, following the same worker-first startup order. Use `--device cpu`
for CPU execution. ResNet and Flan-T5 support Top-K, quantization, and LLM.int8;
their LLM.int8 profiles use FP16/INT4 and FP16/INT8, respectively.

Online results are written under `outputs/`. Plot saved results with
`python plot.py single --sweep_dir outputs` or
`python plot.py multi --sweep_dir outputs`.

## 2. Trace: network-environment traces for offline tests

**Trace primarily generates measured network-environment inputs for offline
tests.** It records the conditions observed during a four-node Jetson execution:
per-link transfer rates, packet sizes, transfer times, and end-to-end delays.
It also records activation sizes, node computation times, and codec overhead.
These measurements provide the network and execution profiles for offline
experiments in the separate **`Inference_Optimizer`** project.

Trace files are saved measurements for that offline workflow. The Jetson online
inference workflow above measures its network during the actual run.

Configure the trace collectors' `NODE_IPS` and `NODE_PORTS` for the four nodes,
as described in the [trace instructions](src/traces/README.md). These collectors
use their own address settings. Start B, C, and D first, then A, using consistent
collection options on all nodes:

```bash
# Run on B, C, and D with the appropriate --node, then on A with --node A.
python trace.py resnet --node B --device cuda --checkpoint_path models/resnet56-4bfd9763.th --data_root data/cifar10 --batch_size 100 --max_batches 100 --tx_limit_mbps 15 --output_dir outputs/traces
python trace.py flan-t5 --node B --device cuda --model_name google/flan-t5-base --dataset_path data/sst2_validation.jsonl --max_samples 100 --tx_limit_mbps 15 --output_dir outputs/traces
```

Choose one task per collection. Node A writes a timestamped directory under
`outputs/traces/`:

| Output | Contents |
| --- | --- |
| `no_compression_trace_rows.csv` | Raw per-batch/sample measurements. |
| `*_scenario_trace_<codec>.json` | Ordered network and execution measurements for offline experiments. |
| `*_trace_summary_<codec>.json` | Collection settings and aggregate statistics. |

Actual network payloads use **identity transport (no compression)**; Top-K,
quantization, and LLM.int8 overheads are measured separately on local activations.
Times are in seconds, sizes in bytes, and channel rates in **bytes/second**.
The `--tx_limit_mbps` option uses **megabits/second**. Fresh measurements vary with
the devices and network conditions.

## 3. Fitting: platform-independent accuracy-function models

**Fitting is general-purpose code for generating accuracy-function fitting
models, and is not limited to Jetson.** It learns the relationship between
compression settings and measured accuracy. The resulting function can be used
by online or offline inference optimizers to evaluate compression choices.
The fitting workflow is platform-independent and can be used in CPU or GPU
environments; it is not restricted to Jetson.

The bundled task workflows use three compression features `(k0, k1, k2)` and
classification accuracy as the target. They cover ResNet-56/CIFAR-10 and
Flan-T5/SST-2 with Top-K, quantization, and LLM.int8. The core estimator in
`src/fitting/estimator.py` also accepts other feature/accuracy arrays; its feature
count is inferred from the input. Reusing it for another model requires that
model's measured compression features and accuracy values.

**Tested CSV-fitting environment:** Python 3.12, NumPy 2.2.4, SciPy 1.17.1,
pandas 3.0.1, and scikit-learn 1.8.0. Install the dependencies with:

```bash
python -m pip install numpy==2.2.4 scipy==1.17.1 pandas==3.0.1 scikit-learn==1.8.0
```

To regenerate the six supplied accuracy models and two mappings from the ten
recorded measurement CSVs in `src/fitting/data/`:

```bash
python fit.py bundled --output-dir outputs/fitted
```

The command writes models under `outputs/fitted/<task>/<policy>/` and mappings
under `outputs/fitted/mappings/`. `src/fitting/recipes.json` specifies each model's
input files and row order. To use a regenerated fit in online inference, update
the corresponding files in `models/accuracy_estimators/`, which the online
configuration loads by default.

The supplied `.pkl` files estimate accuracy from compression ratios. The two
LLM.int8 mapping JSONs associate measured transfer ratios with executable outlier
settings. Fitting these functions does not train the ResNet or Flan-T5 neural
weights.

To measure a new accuracy curve, collect a CSV first, then pass it to the fitting
command. The included collectors use PyTorch and the corresponding model/data
assets on a CPU or CUDA machine. For example:

```bash
python fit.py collect resnet --device cuda --checkpoint_path models/resnet56-4bfd9763.th --data_root data/cifar10 --compressor_name all --csv_path outputs/calibration/resnet.csv
python fit.py resnet --csv outputs/calibration/resnet.csv --output-dir outputs/resnet-fit --model-types poly3
```

Accuracy collection runs the four partitions locally on one machine while
sweeping compression settings, with `--device cpu` or `--device cuda`. The
CSV-fitting step uses the bundled NumPy/scikit-learn regression implementation.
Flan-T5 uses `fit.py collect flan-t5` followed by `fit.py flan-t5`. Fitting
requires a new or empty output directory. The [fitting instructions](src/fitting/README.md) give both tasks'
collection commands, measured grids, model filenames, and fitting recipes.

## License

The project code is licensed under the [MIT License](LICENSE).

ResNet architecture attribution is retained in
`src/jetson_inference/common/resnet_arch.py`. Model and dataset terms follow their
upstream releases.
