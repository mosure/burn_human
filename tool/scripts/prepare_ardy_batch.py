#!/usr/bin/env python3
"""Prepare batch inputs from five existing burn_llama TextEmbedding JSON files.

The separate baseline Rust example fills the references; no model outputs are
synthesized here. Embeddings must have their original prompt and provenance.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--baseline-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    embeddings = [json.loads((args.embeddings / f'embedding-{i}.json').read_text()) for i in range(5)]
    cases = []
    for name, actors, frames, history, dense, paths in [
        ('single', 1, 40, 0, True, False),
        ('partial', 2, 44, 4, True, True),
        ('no_history', 2, 80, 0, True, False),
        ('dense_curves', 4, 120, 40, True, True),
        ('sparse_long', 2, 240, 160, False, True),
        ('eight_actors', 8, 40, 0, True, False),
        ('duplicate_seed', 2, 40, 0, True, False),
        *[(f'actors_{n}', n, 40, 0, True, False) for n in [3, 5, 6, 7]],
    ]:
        requests, indices = [], []
        for actor in range(actors):
            index = 0 if name == 'duplicate_seed' else actor % 5
            seed = 42 if name == 'duplicate_seed' else 42 + actor * 101
            waypoints = []
            if paths:
                for frame, x, z, heading in [(0, 0, 0, 3.05), (frames // 2, 0.6, 1.2, -3.05), (frames - 1, 1.5, 2.0, -2.8)]:
                    waypoints.append(dict(frame=frame, position=[x * (-1 if actor % 2 else 1), 0, z], heading=heading, constrain_height=False))
            requests.append(dict(prompt=embeddings[index]['prompt'], seed=seed, frames=frames,
                history_frames=history, diffusion_steps=10, text_guidance=2.0,
                trajectory_guidance=2.0, waypoints=waypoints, dense_trajectory=dense))
            indices.append(index)
        cases.append(dict(name=name, requests=requests, embedding_indices=indices, reference=[]))
    args.out.write_text(json.dumps(dict(baseline_commit=args.baseline_commit, embeddings=embeddings, cases=cases)))


if __name__ == '__main__':
    main()
