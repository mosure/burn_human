"""
Exports deterministic reference outputs from the Python Anny implementation.

This locks golden tensors that the Rust/Burn port must reproduce numerically.
Outputs are written to a safetensors file (plus small JSON metadata) under
`assets/model/`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import roma
from safetensors.torch import save_file as save_safetensors


ROOT = Path(__file__).resolve().parents[2]
ANNY_SRC = ROOT / "anny" / "src"

# Ensure we import the vendored Anny package instead of any installed version.
if str(ANNY_SRC) not in sys.path:
    sys.path.insert(0, str(ANNY_SRC))

# Keep Anny cache inside the repo for reproducibility.
os.environ.setdefault("ANNY_CACHE_DIR", str(ROOT / ".cache" / "anny"))

import anny  # noqa: E402
from anny.models.phenotype import PHENOTYPE_VARIATIONS  # noqa: E402


def _seed_all(seed: int) -> torch.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1)
    return gen


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _to_cpu_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().contiguous()

def _to_f16(t: torch.Tensor) -> torch.Tensor:
    return _to_cpu_tensor(t.to(dtype=torch.float16))


def _identity_pose(model, batch_size: int) -> torch.Tensor:
    eye = torch.eye(4, dtype=model.dtype, device=model.device)
    return eye.view(1, 1, 4, 4).repeat(batch_size, model.bone_count, 1, 1)


def _random_pose(
    model,
    gen: torch.Generator,
    batch_size: int,
    rot_std_deg: float = 25.0,
    trans_std: float = 0.05,
) -> torch.Tensor:
    rotvec = torch.randn(
        (batch_size, model.bone_count, 3),
        generator=gen,
        device=model.device,
        dtype=model.dtype,
    )
    rotvec = rotvec * (rot_std_deg * np.pi / 180.0)
    rotmat = roma.rotvec_to_rotmat(rotvec)
    translations = torch.zeros(
        (batch_size, model.bone_count, 3), device=model.device, dtype=model.dtype
    )
    translations[:, 0, :] = torch.randn(
        (batch_size, 3), generator=gen, device=model.device, dtype=model.dtype
    ) * trans_std
    return roma.Rigid(rotmat, translations).to_homogeneous()


def _phenotype_from_tensor(model, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Build the phenotype kwargs dict expected by the model from a [B, P] tensor.
    """
    return {label: tensor[:, i] for i, label in enumerate(model.phenotype_labels)}


def _sample_phenotype(
    model, gen: torch.Generator, batch_size: int, *, mode: str
) -> Dict[str, torch.Tensor]:
    """
    mode:
      - "mid": constant 0.5 everywhere
      - "zero": constant 0.0 everywhere
      - "random": uniform [0, 1)
    """
    if mode == "mid":
        base = torch.full(
            (batch_size, len(model.phenotype_labels)),
            fill_value=0.5,
            dtype=model.dtype,
            device=model.device,
        )
    elif mode == "zero":
        base = torch.zeros(
            (batch_size, len(model.phenotype_labels)),
            dtype=model.dtype,
            device=model.device,
        )
    elif mode == "random":
        base = torch.rand(
            (batch_size, len(model.phenotype_labels)),
            generator=gen,
            dtype=model.dtype,
            device=model.device,
        )
    else:
        raise ValueError(f"Unknown phenotype sampling mode '{mode}'")
    return _phenotype_from_tensor(model, base)


def _stack_phenotypes(model, phenotype_kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        [phenotype_kwargs[label] for label in model.phenotype_labels], dim=1
    )


def _run_case(
    model,
    name: str,
    pose_parameters: torch.Tensor,
    phenotype_kwargs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        output = model(
            pose_parameters=pose_parameters,
            phenotype_kwargs=phenotype_kwargs,
            return_bone_ends=True,
        )

    return {
        f"{name}__phenotype_inputs": _to_cpu_tensor(
            _stack_phenotypes(model, phenotype_kwargs)
        ),
        f"{name}__pose_parameters": _to_cpu_tensor(pose_parameters),
        f"{name}__blendshape_coeffs": _to_cpu_tensor(output["blendshape_coeffs"]),
        f"{name}__rest_vertices": _to_cpu_tensor(output["rest_vertices"]),
        f"{name}__posed_vertices": _to_cpu_tensor(output["vertices"]),
        f"{name}__rest_bone_poses": _to_cpu_tensor(output["rest_bone_poses"]),
        f"{name}__bone_poses": _to_cpu_tensor(output["bone_poses"]),
        f"{name}__bone_heads": _to_cpu_tensor(output["bone_heads"]),
        f"{name}__bone_tails": _to_cpu_tensor(output["bone_tails"]),
    }


def export_reference(output_path: Path, seed: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".meta.json")

    gen = _seed_all(seed)
    torch.set_default_dtype(torch.float32)

    # Default full-body model, no local changes; matches the core forward pipeline we port.
    model = anny.create_fullbody_model(local_changes=False, extrapolate_phenotypes=False)
    model = model.to(dtype=torch.float32, device="cpu")
    model.eval()

    static_blobs = {
        "faces_quads": _to_cpu_tensor(model.faces),
        "face_texture_coordinate_indices": _to_cpu_tensor(model.face_texture_coordinate_indices),
        "template_vertices": _to_cpu_tensor(model.template_vertices),
        "texture_coordinates": _to_cpu_tensor(model.texture_coordinates),
        "vertex_bone_indices": _to_cpu_tensor(model.vertex_bone_indices),
        "vertex_bone_weights": _to_cpu_tensor(model.vertex_bone_weights),
        "blendshapes": _to_f16(model.blendshapes),
        "blendshape_mask": _to_cpu_tensor(model.stacked_phenotype_blend_shapes_mask),
        "template_bone_heads": _to_cpu_tensor(model.template_bone_heads),
        "bone_heads_blendshapes": _to_cpu_tensor(model.bone_heads_blendshapes),
        "template_bone_tails": _to_cpu_tensor(model.template_bone_tails),
        "bone_tails_blendshapes": _to_cpu_tensor(model.bone_tails_blendshapes),
        "bone_rolls_rotmat": _to_cpu_tensor(model.bone_rolls_rotmat),
    }

    cases: Iterable[Tuple[str, torch.Tensor, Dict[str, torch.Tensor]]] = [
        (
            "neutral_pose_mid_phenotype",
            _identity_pose(model, batch_size=1),
            _sample_phenotype(model, gen, batch_size=1, mode="mid"),
        ),
    ]

    payload: Dict[str, torch.Tensor] = {}
    payload.update(static_blobs)
    case_names = [c[0] for c in cases]
    metadata = {
        "seed": seed,
        "rig": "default",
        "topology": "default",
        "case_names": case_names,
        "dtype": "mixed_f16_f32",
        "bone_labels": model.bone_labels,
        "bone_parents": model.bone_parents,
        "phenotype_labels": model.phenotype_labels,
        "pose_parameterization": model.default_pose_parameterization,
        "phenotype_variations": PHENOTYPE_VARIATIONS,
        "macrodetail_keys": [
            z for detail_type, values in PHENOTYPE_VARIATIONS.items() for z in values
        ],
        "phenotype_anchors": {k: v.cpu().numpy().tolist() for k, v in model.anchors._buffers.items()},
    }

    for name, pose, phen in cases:
        payload.update(_run_case(model, name=name, pose_parameters=pose, phenotype_kwargs=phen))

    save_safetensors(payload, output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote reference outputs to {output_path}")
    print(f"Wrote metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "reference" / "fullbody_default.safetensors",
        help="Destination safetensors path for golden tensors.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    args = parser.parse_args()

    export_reference(output_path=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
