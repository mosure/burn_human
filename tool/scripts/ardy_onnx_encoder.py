#!/usr/bin/env python3
"""Pinned TREE Industries ARDY LLM2Vec INT4 encoder. Built with Meta Llama 3.

Offline numerical oracle for the Rust/Burn encoder. Download the pinned
repository and its license notices separately; no runtime service is needed.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import time

MODEL = "TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX"
REVISION = "7aa52a05d54c2fd9177366aeb3f88e9e7f3c5766"
HASHES = {
    "onnx/text_encoder_int4.onnx": "c44f7ce04611a1a6f44ddf346ee89f4bc37356e45c0348e72ccd80bbf7b84e61",
    "onnx/text_encoder_int4.onnx.data": "765e9ce4f9def78042dbd7e275500a1dd87f3f65f9dc8c518b54d163ae46fca3",
    "tokenizer/tokenizer.json": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
    "tokenizer/tokenizer_config.json": "da0e3a7cce6e4d787e85eb1c24d548420e0d7fe2c7a214e192795c46e40d75bb",
    "validation/text_encoder_reference.json": "f2feeacc3e9ba85740c5bbfeccfd21b5534be7ac578fb9c92f1b7e5933a69b26",
}


def verify(root):
    for name, expected in HASHES.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"{name} differs from pinned {MODEL}@{REVISION}")


class OnnxEncoder:
    def __init__(self, root, provider="CUDAExecutionProvider", profile=False, bidirectional=False):
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.np = np
        self.root = pathlib.Path(root)
        verify(self.root)
        # The CUDA wheel can reuse the CUDA/cuDNN libraries installed by PyTorch.
        if provider == "CUDAExecutionProvider":
            import torch  # noqa: F401
            ort.preload_dlls()
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Requested provider {provider} unavailable")
        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer", local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.enable_profiling = profile
        if profile:
            options.profile_file_prefix = str(self.root / "runtime-profile")
        graph = self.root / "onnx/text_encoder_int4.onnx"
        if bidirectional:
            # The public graph hard-codes a causal mask despite its model card.
            # Change only that mask to match upstream ARDY's LLM2Vec contract.
            import onnx
            model = onnx.load(graph, load_external_data=False)
            masks = [n for n in model.graph.node if n.op_type == "LessOrEqual"]
            if len(masks) != 1 or list(masks[0].input) != ["/Constant_output_0", "/Constant_1_output_0"]:
                raise ValueError("Unexpected pinned attention-mask graph")
            replacement = onnx.helper.make_node("Constant", [], list(masks[0].output),
                value=onnx.numpy_helper.from_array(np.ones((64, 64), dtype=bool)))
            masks[0].CopyFrom(replacement)
            graph = self.root / "onnx/burn_bidirectional_reference.onnx"
            onnx.save_model(model, graph)
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=[provider])
        if self.session.get_providers()[0] != provider:
            raise RuntimeError(f"ONNX Runtime failed to initialize {provider}; refusing silent fallback")
        self.provider = provider

    def inputs(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode()) > 16384:
            raise ValueError("Enter a nonempty prompt of at most 16384 UTF-8 bytes")
        formatted = "<|start_header_id|>user<|end_header_id|>\n\n" + prompt.strip() + "<|eot_id|>"
        raw = self.tokenizer(formatted)["input_ids"]
        # Avoid silently claiming to encode text the fixed-length export dropped.
        if len(raw) > 64:
            raise ValueError(f"Prompt needs {len(raw)} tokens; this ONNX export allows 64 including the chat header. Shorten it.")
        if raw[:5] != [128000, 128006, 882, 128007, 271]:
            raise ValueError("Unexpected ARDY Llama user header")
        encoded = self.tokenizer(formatted, padding="max_length", truncation=True, max_length=64, return_tensors="np")
        mask = encoded["attention_mask"].astype(self.np.int64)
        embed = mask.copy()
        embed[0, self.np.flatnonzero(mask[0])[:5]] = 0
        return {"input_ids": encoded["input_ids"].astype(self.np.int64), "attention_mask": mask, "embed_mask": embed}

    def encode(self, prompt):
        values = self.session.run(["text_embedding"], self.inputs(prompt))[0]
        if values.shape != (1, 4096) or not self.np.isfinite(values).all():
            raise ValueError("Invalid text encoder output")
        return dict(prompt=prompt, encoder=MODEL, revision=REVISION, values=values[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--provider", default="CUDAExecutionProvider", choices=["CUDAExecutionProvider", "CPUExecutionProvider", "DmlExecutionProvider"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    encoder = OnnxEncoder(args.model, args.provider, args.profile)
    load_seconds = time.perf_counter() - start
    reference = json.loads((args.model / "validation/text_encoder_reference.json").read_text())
    records = []
    for i, record in enumerate(reference["records"]):
        inputs = encoder.inputs(record["prompt"])
        for name, values in inputs.items():
            if not encoder.np.array_equal(values.reshape(-1), encoder.np.asarray(record[name]).reshape(-1)):
                raise AssertionError(f"Tokenization mismatch {i}: {name}")
        result = encoder.encode(record["prompt"])
        a = encoder.np.asarray(result["values"], dtype=encoder.np.float64)
        b = encoder.np.asarray(record["embedding"], dtype=encoder.np.float64)
        cosine = float(a @ b / (encoder.np.linalg.norm(a) * encoder.np.linalg.norm(b)))
        if cosine < 0.97:
            raise AssertionError(f"INT4 vs upstream FP16 cosine {cosine} < 0.97")
        (args.out / f"prompt-{i}.json").write_text(json.dumps(result) + "\n")
        records.append(dict(prompt=record["prompt"], cosine=cosine, max_abs=float(encoder.np.max(encoder.np.abs(a - b)))))
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        encoder.encode(reference["records"][0]["prompt"])
        elapsed.append(time.perf_counter() - start)
    import onnxruntime as ort
    evidence = dict(model=MODEL, revision=REVISION, onnxruntime=ort.__version__, provider=encoder.provider,
                    load_seconds=load_seconds, records=records, warm_encode_seconds=elapsed,
                    median_encode_seconds=statistics.median(elapsed), hashes=HASHES,
                    comparison="INT4 CUDA/CPU output vs publisher FP16 references; not bitwise parity")
    if args.profile:
        evidence["profile"] = encoder.session.end_profiling()
    (args.out / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
