# Data

- `sst2_validation.jsonl`: 872 SST-2 validation sentences for Flan-T5 sentiment
  classification. Each line contains `id`, `sentence`, and `label`
  (`0` = negative, `1` = positive).
- `cifar10/`: CIFAR-10 test images and labels used by ResNet-56. This cache is
  created on download and is not bundled with the repository.

From the repository root, download CIFAR-10 and cache the Flan-T5 model with:

```bash
python prepare.py assets --download
```

To regenerate the SST-2 JSONL from GLUE/SST-2:

```bash
python -m pip install datasets
python prepare.py dataset --split validation --output outputs/sst2_validation.jsonl
```
