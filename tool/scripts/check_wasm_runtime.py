#!/usr/bin/env python3
"""Reject repeated LLD constructor wrappers in a named, pre-bindgen WASM file.

Burn's compiler registry uses static constructors. Command-style linkage wraps
every exported function (including allocator/closure interop) with those
constructors. Browser modules must instead initialize the registry once.
Run before wasm-bindgen removes the name section.
"""
import pathlib
import sys


def leb(data, offset):
    value = shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
        shift += 7
        if shift > 35:
            raise ValueError("invalid WASM section length")


def string(data, offset):
    size, offset = leb(data, offset)
    return bytes(data[offset:offset + size]).decode(), offset + size


def function_names(data):
    assert data[:8] == b"\0asm\x01\0\0\0", "expected WASM 1.0"
    offset = 8
    while offset < len(data):
        kind = data[offset]
        size, begin = leb(data, offset + 1)
        offset = begin + size
        if kind != 0:
            continue
        name, pos = string(data, begin)
        if name != "name":
            continue
        while pos < offset:
            subsection = data[pos]
            length, start = leb(data, pos + 1)
            pos = start + length
            if subsection != 1:
                continue
            count, entry = leb(data, start)
            for _ in range(count):
                _, entry = leb(data, entry)
                name, entry = string(data, entry)
                yield name


def main():
    for path in sys.argv[1:]:
        names = list(function_names(pathlib.Path(path).read_bytes()))
        assert "__wasm_call_ctors" in names, "missing named constructors: run before stripping names"
        assert any("initialize_browser_runtime" in n for n in names), "missing guarded browser initializer"
        wrappers = [n for n in names if n.endswith(".command_export")]
        assert not wrappers, f"repeated constructors on exported calls: {wrappers[:6]}"
        print(f"PASS {path}: one-time browser runtime linkage")
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_wasm_runtime.py <named.wasm> [...]")


if __name__ == "__main__":
    main()
