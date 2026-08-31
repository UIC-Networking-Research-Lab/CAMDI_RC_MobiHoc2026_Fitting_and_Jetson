# Jetson Online Test

Self-contained four-node Jetson test project for:

- ResNet56/CIFAR-10 single-task online inference
- Flan-T5/SST-2 single-task online inference
- asynchronous ResNet56 + Flan-T5 multi-task online inference

All project-specific Python modules, the ResNet56 checkpoint, accuracy-estimator files, LLM.int8 mappings, and a local SST-2 validation file are included here. The code does not import anything from the parent repository.

## 1. Install and prepare

Copy the complete folder to nodes A/B/C/D and use compatible Python environments on all four nodes. The absolute path may differ between nodes.

```bash
cd jetson_online_test
python3 -m pip install -r requirements.txt
python3 prepare_assets.py --download
```

On Jetson, PyTorch often needs the NVIDIA wheel matching the installed JetPack release. If the PyPI `torch`/`torchvision` wheels are unsuitable, install NVIDIA's matching builds first, then install the remaining requirements.

`prepare_assets.py --download` validates the bundled files, downloads CIFAR-10 into `data/cifar10`, and caches `google/flan-t5-base`. Run it once on every node. Subsequent runs use the local caches.

## 2. Configure the four nodes

Defaults are A=`192.168.1.20`, B=`192.168.1.21`, C=`192.168.1.22`, D=`192.168.1.23`, using ports 52000-52003. Override them on **every node** before launching:

```bash
export JETSON_NODE_A_IP=192.168.1.20
export JETSON_NODE_B_IP=192.168.1.21
export JETSON_NODE_C_IP=192.168.1.22
export JETSON_NODE_D_IP=192.168.1.23
```

Optional variables are `JETSON_NODE_A_PORT` through `JETSON_NODE_D_PORT`, plus `FLAN_T5_MODEL` for a local Hugging Face model directory or another compatible model ID. Ensure the selected ports are reachable between nodes.

## 3. Run an experiment

Start workers B, C, and D first on their respective Jetsons, then start controller A. Use exactly the same experiment script and common arguments on all nodes.

### ResNet56 single task

```bash
# Node B/C/D: replace B with the local node ID
python3 jetson_resnet_pipeline_singletask_policy.py --node B --device cuda

# Node A, started last
python3 jetson_resnet_pipeline_singletask_policy.py --node A --device cuda
```

### Flan-T5 single task

```bash
# Node B/C/D
python3 jetson_flan_t5_pipeline_singletask_policy.py --node B --device cuda

# Node A
python3 jetson_flan_t5_pipeline_singletask_policy.py --node A --device cuda
```

### ResNet56 + Flan-T5 multi task

```bash
# Node B/C/D
python3 jetson_multi_task_pipeline_async.py --node B --device cuda

# Node A
python3 jetson_multi_task_pipeline_async.py --node A --device cuda
```

Use `python3 SCRIPT.py --help` for the full parameter list. Useful smoke-test limits include `--max_total_batches`, `--max_dynamic_timeslot_count`, `--compressor_profiles topk`, and `--accuracy_estimator_modes fitting_model`.

Single-task channel prediction defaults to warmup-only training. Add `--channel_predictor_mode online` on all four nodes to retrain online. Multi-task prediction defaults to online mode.

Results are written under `outputs/resnet_single_task`, `outputs/flan_t5_single_task`, or `outputs/multi_task_async` inside this project.

## Project contents

```text
jetson_online_test/
  jetson_resnet_pipeline_singletask_policy.py
  jetson_flan_t5_pipeline_singletask_policy.py
  jetson_multi_task_pipeline_async.py
  channel_estimator.py
  compressors.py
  http_transport.py
  http_transport_dynamic.py
  llmint8_eta_mapping.py
  resnet20.py
  single_task_offline_plots.py
  multi_task_offline_plots.py
  prepare_assets.py
  data/sst2_validation.jsonl
  models/resnet56-4bfd9763.th
  models/accuracy_estimators/
```

## Before publishing

The new folder is technically independent, but repository licensing is a separate concern. Before making it public, add the license you intend for your own code and verify redistribution terms/provenance for the bundled ResNet checkpoint, estimator artifacts, and SST-2 data. Never run this transport on an untrusted network: messages use Python pickle serialization.
