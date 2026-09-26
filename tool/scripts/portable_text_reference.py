#!/usr/bin/env python3
"""Independent ORT oracles for the source mask and the upstream ARDY mask."""
import argparse
import json
import pathlib
import time
from ardy_onnx_encoder import OnnxEncoder, MODEL, REVISION, HASHES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    refs = json.loads((args.model / "validation/text_encoder_reference.json").read_text())
    result = dict(model=MODEL, revision=REVISION, hashes=HASHES, modes={})
    for mode in ("causal_export", "bidirectional"):
        start = time.perf_counter()
        encoder = OnnxEncoder(args.model, bidirectional=mode == "bidirectional")
        records = []
        for r in refs["records"]:
            inputs = encoder.inputs(r["prompt"])
            record = encoder.encode(r["prompt"])
            record["tokens"] = {k: v.reshape(-1).tolist() for k, v in inputs.items()}
            records.append(record)
        result["modes"][mode] = records
        print(mode, "reference seconds", time.perf_counter() - start, flush=True)
        del encoder
    args.output.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
