#!/usr/bin/env python3
"""Offline, lossless ONNX MatMulNBits -> Burn Q4F32 export. No ONNX runtime needed.

Download the pinned public repository with its Llama license notices first. Run
human-model-pack on the output directory to produce the bounded CDN artifact.
The original export has a causal attention mask; the Burn runtime exposes both
export parity and ARDY's intended bidirectional LLM2Vec attention explicitly.
"""
import argparse
import hashlib
import json
import pathlib
import shutil

import numpy as np
import onnx
from onnx import numpy_helper
from ardy_onnx_encoder import HASHES, MODEL, REVISION, verify


def export(root, output):
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    model = onnx.load(root / "onnx/text_encoder_int4.onnx", load_external_data=False)
    initializers = {t.name: t for t in model.graph.initializer}
    tensors = []

    def read(name):
        t = initializers[name]
        external = {e.key: e.value for e in t.external_data}
        if external:
            location = (root / "onnx" / external["location"]).resolve()
            if not location.is_relative_to(root.resolve()):
                raise ValueError("External tensor escapes the verified source")
            return np.memmap(location, mode="r", dtype=onnx.helper.tensor_dtype_to_np_dtype(t.data_type),
                             offset=int(external.get("offset", "0")), shape=tuple(t.dims))
        return numpy_helper.to_array(t)

    def write(name, shape, dtype, chunks):
        path = f"raw/{name}.bin"
        digest = hashlib.sha256()
        with (output / path).open("wb") as stream:
            for data in chunks:
                stream.write(data)
                digest.update(data)
        tensors.append(dict(name=name, shape=shape, dtype=dtype, file=path, sha256=digest.hexdigest()))

    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs["bits"] == 4 and attrs["block_size"] == 32 and len(node.input) == 3
        n, k = attrs["N"], attrs["K"]
        q = read(node.input[1])
        scales = read(node.input[2])
        assert q.shape == (n, k // 32, 16) and scales.size == n * k // 32
        assert np.isfinite(scales).all()
        name = node.name.removeprefix("/").removesuffix("/MatMul_Q4").replace("/", ".") + ".weight"
        # ONNX unsigned nibbles have implicit zero-point 8. Burn stores signed
        # two's complement nibbles, low nibble first, with appended F32 scales.
        # Negative scales are meaningful checkpoint values, never abs/clamped.
        write(name, [n, k], "q4f32", [(q ^ np.uint8(0x88)).tobytes(), scales.astype("<f4").tobytes()])
    for name in initializers:
        if "layernorm.weight" in name or name == "model.norm.weight":
            array = read(name)
            assert array.shape == (4096,) and array.dtype == np.float16
            write(name, list(array.shape), "f16", [array.astype("<f2").tobytes()])
    vocab = read("model.embed_tokens.weight")
    assert vocab.shape == (128256, 4096) and vocab.dtype == np.float16
    for page, start in enumerate(range(0, len(vocab), 512)):
        values = vocab[start:start + 512]
        write(f"model.embed_tokens.page.{page:04}", list(values.shape), "f16", [values.astype("<f2").tobytes()])
    assets = []
    for source, target in [("tokenizer/tokenizer.json", "tokenizer.json"),
                           ("tokenizer/tokenizer_config.json", "tokenizer_config.json"),
                           ("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"),
                           ("ACCEPTABLE_USE_POLICY.md", "ACCEPTABLE_USE_POLICY.md"), ("LICENSES/MIT-LLM2Vec.txt", "LLM2VEC-LICENSE")]:
        source_path = root / source
        if not source_path.exists():
            if "tokenizer" in source or source in ("LICENSE", "NOTICE"):
                raise FileNotFoundError(source_path)
            continue
        dest = output / target
        shutil.copyfile(source_path, dest)
        assets.append(dict(path=f"metadata/{target}", file=target, sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    provenance = dict(source_hashes=HASHES, quantization="ONNX INT4 block32 implicit zero8 -> signed Q4F, exact scales widened to F32",
                      source_attention="causal", ardy_attention="bidirectional", converter="export_ardy_text.py:v1")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    assets.append(dict(path="metadata/provenance.json", file="provenance.json", sha256=hashlib.sha256((output / "provenance.json").read_bytes()).hexdigest()))
    assert len(tensors) == 224 + 65 + 251, len(tensors)
    result = dict(model=MODEL, model_revision=REVISION, source_revision=REVISION,
                  converter="export_ardy_text.py:v1", license="Llama-3 community license; see metadata/LICENSE and metadata/NOTICE",
                  config=dict(hidden=4096, intermediate=14336, layers=32, heads=32, kv_heads=8,
                              vocab=128256, page_rows=512, max_sequence=64, rms_epsilon=1e-5, rope_theta=500000.0),
                  tensors=tensors, assets=assets)
    (output / "export.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exported {len(tensors)} tensors to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=pathlib.Path)
    ap.add_argument("output", type=pathlib.Path)
    args = ap.parse_args()
    export(args.source, args.output)
