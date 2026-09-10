# Performance trace generation

This package measures four-node Jetson execution and generates network-environment traces primarily for offline tests in the separate `Inference_Optimizer` project. The files record measured link rates, transfer times, activation sizes, computation times, and codec overhead. The Jetson online runner measures its channel during execution. Accuracy measurement and model fitting are documented in [fitting](../fitting/README.md).

| Module | Purpose |
| --- | --- |
| `performance_resnet.py` | Collect ResNet56 / CIFAR-10 performance traces. |
| `performance_flan_t5.py` | Collect Flan-T5 / SST-2 performance traces; reuse the partition and data-loading helpers in `fitting.calibrate_flan_t5`. |
| `performance_resnet_support.py` | ResNet partitions, identity payloads and timing helpers. |
| `cli.py` | Dispatch `trace.py` commands. |

## Collect a trace

Use four reachable nodes with the same software and assets. Defaults are A/B/C/D = `192.168.1.20` through `192.168.1.23`, ports `52000` through `52003`. Configure `NODE_IPS` / `NODE_PORTS` in `performance_resnet_support.py` or `performance_flan_t5.py` for another topology. These collectors do not read the online runner's `JETSON_NODE_*` environment variables.

Run from the repository root. Start B, C and D with the corresponding `--node` first, then run the same command with `--node A`. Keep codec/timing options consistent across all four nodes. Choose one task:

```bash
python trace.py resnet --node B --device cuda --checkpoint_path models/resnet56-4bfd9763.th --data_root data/cifar10 --max_batches 2 --batch_size 2 --tx_limit_mbps 25 --output_dir outputs/traces
python trace.py flan-t5 --node B --device cuda --model_name google/flan-t5-base --dataset_path data/sst2_validation.jsonl --max_samples 2 --tx_limit_mbps 25 --output_dir outputs/traces
```

Use `--help` for all options or `--device cpu` for CPU execution. Prepare datasets/model caches first with `python prepare.py assets --download`. Real collection requires the four nodes and evaluation assets.

## Generated files

Node A saves a timestamped scenario directory under `--output_dir` (default `scenario_trace_outputs/`).

| File | Contents |
| --- | --- |
| `no_compression_trace_rows.csv` | Raw per-batch/sample computation and transfer measurements. |
| `*_scenario_trace_<codec>.json` | Ordered activation sizes, node times, channel rates, delays and codec overhead for offline replay. |
| `*_trace_summary_<codec>.json` | Collection settings, sample counts, mean/median measurements and summary statistics. |

Each selected codec has its own pair of JSON files; defaults are TopK, quantization and LLM.int8. Actual A → B → C → D payloads use **identity (no compression)**. Codec overhead is measured separately on local activations and does not change the transported payloads.

Times are in seconds; activation/packet sizes in bytes; channel rates in **bytes/second**; `--tx_limit_mbps` in megabits/second. Node residence, compute, codec and end-to-end times remain separate fields. Fresh measurements depend on the hardware and network, so regenerated timing values need not equal a previous trace.
