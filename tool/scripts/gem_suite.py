#!/usr/bin/env python3
"""Write portable relative URLs and pinned manifest identities for a GEM suite."""
import argparse
import json
import os
import pathlib

MODELS = {
    "vitpose": "nvidia/GEM-X:vitpose",
    "sam_vision": "nvidia/GEM-X:sam-body",
    "sam_decoder": "nvidia/GEM-X:sam-decoder",
    "denoiser": "nvidia/GEM-X:denoiser",
    "mhr": "facebookresearch/MHR:SOMA-X-lod1",
    "soma": "nvidia/SOMA-X",
    "transfer": "nvidia/SOMA-X:MHR-transfer",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    for name in MODELS:
        parser.add_argument("--" + name.replace("_", "-"), type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = {}
    for name, model in MODELS.items():
        directory = getattr(args, name).resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["model"] != model:
            raise ValueError(f"Expected {model}, found {manifest['model']}")
        # The runtime authenticates the seal, metadata and all tensor bytes.
        result[name] = dict(base=pathlib.Path(os.path.relpath(directory, args.out.resolve().parent)).as_posix(),
                            sha256=manifest["content_sha256"])
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
