#!/usr/bin/env python3
"""Run synchronized real-checkpoint validators sequentially, retaining provenance.

Use separate output directories and immutable before/after binaries. These are
warm inference measurements, not load benchmarks or isolated GPU guarantees.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['bin-dir', 'catalog', 'ardy-reference', 'soma-reference', 'gem-reference', 'image', 'out']:
        parser.add_argument('--' + arg, type=pathlib.Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    commands = {
        'ardy': ['ardy-validate', args.catalog / 'ardy/core-rp-20fps-h40/v1', args.ardy_reference, 'wgpu'],
        'soma': ['soma-validate', args.catalog / 'soma-x/v1/body', args.soma_reference, args.out / 'soma.json'],
        'gemx': ['gem-validate', args.catalog / 'gemx/v1/suite.json', args.image, args.gem_reference, args.out / 'gemx.json'],
    }
    records = []
    for name, command in commands.items():
        executable = (args.bin_dir / command[0]).resolve()
        command = [str(executable), *map(str, command[1:])]
        record = dict(name=name, label=args.label, command=command,
                      binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                      start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if shutil.which('nvidia-smi'):
            record['gpu_before'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=name,driver_version,memory.used,utilization.gpu', '--format=csv'], text=True)
        print('START', name, record['start_utc'], flush=True)
        start = time.monotonic()
        log_path = args.out / (name + '.log')
        with log_path.open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
        records.append(record)
        (args.out / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
        print('DONE', name, record['exit_code'], record['elapsed_seconds'], flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        if name == 'ardy':
            text = log_path.read_text()
            (args.out / 'ardy.json').write_text(json.dumps(json.loads(text[text.index('{\n'):]), indent=2) + '\n')


if __name__ == '__main__':
    main()
