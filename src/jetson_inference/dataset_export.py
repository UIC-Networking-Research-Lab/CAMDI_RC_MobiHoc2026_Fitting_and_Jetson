"""Export the SST-2 validation sentences and labels consumed by online inference."""
import argparse
import json
from pathlib import Path


def export_samples(samples, destination):
    """Write only the numeric sample id, sentence and binary sentiment label."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}; choose a new destination.")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"id": int(sample.sample_id), "sentence": sample.sentence, "label": sample.label}
            for sample in samples]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from jetson_inference.flan_t5.model import load_hf_sst2_samples
    count = export_samples(load_hf_sst2_samples(args.split), args.output)
    print(f"Wrote {count} SST-2 records to {args.output}")


if __name__ == "__main__":
    main()
