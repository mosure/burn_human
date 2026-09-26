"""Small, model-neutral offline raw export writer consumed by human-model-pack."""
import hashlib
import json
import pathlib
import numpy as np


class ExportWriter:
    def __init__(self, output):
        self.output = pathlib.Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "raw").mkdir()
        self.tensors = []
        self.assets = []

    def tensor(self, name, values, dtype="f32"):
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.ascontiguousarray(values, dtype={"f32": "<f4", "f16": "<f2", "i32": "<i4"}[dtype])
        raw = values.tobytes()
        if len(raw) > 64 * 1024 * 1024 - 4096:
            raise ValueError(f"Split oversized tensor {name} before export")
        filename = f"raw/{name}.bin"
        (self.output / filename).write_bytes(raw)
        self.tensors.append(dict(name=name, shape=list(values.shape), dtype=dtype, file=filename,
                                 sha256=hashlib.sha256(raw).hexdigest()))

    def asset(self, name, raw):
        (self.output / name).write_bytes(raw)
        self.assets.append(dict(path=f"metadata/{name}", file=name, sha256=hashlib.sha256(raw).hexdigest()))

    def json(self, name, data):
        self.asset(name, (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode())

    def finish(self, **metadata):
        result = dict(**metadata, tensors=self.tensors, assets=self.assets)
        (self.output / "export.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def verify_file(path, expected):
    with pathlib.Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Checkpoint digest mismatch: {path}")
    return digest
