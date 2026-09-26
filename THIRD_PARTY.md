# Human inference provenance

Runtime code is implemented in Rust/Burn. Python/ONNX/Torch utilities are
conversion and reference tools. Model weights are separately downloaded and
are not included in the repository or Cargo packages.

| Component | Pinned source / weights | Notices |
|---|---|---|
| ARDY Core | [source](https://github.com/nv-tlabs/ardy/tree/693f74d13b3d04a0a22ce127ee79c929dd89756b), [checkpoint](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40/tree/abe6c43beb28c867c950acb824b9c4ef3d63fb76) | NVIDIA code Apache-2.0; checkpoint NVIDIA Open Model License; original license/notice retained in `crates/burn_ardy` |
| Llama/LLM2Vec | [TREE Industries export](https://huggingface.co/TREEIndustries/Llama-3-ARDY-Text-Encoder-ONNX/tree/7aa52a05d54c2fd9177366aeb3f88e9e7f3c5766) | Meta Llama 3 Community License; LLM2Vec MIT; retain LICENSE, NOTICE, ACCEPTABLE_USE_POLICY.md, LICENSES/MIT-LLM2Vec.txt |
| SOMA-X | [source](https://github.com/NVlabs/SOMA-X/tree/cc1f3967755f8e36d187d2e26114633dbd651cd5), [assets](https://huggingface.co/nvidia/SOMA-X/tree/104578ed58857f6faa7592fb83d0a2dad43c36fa) | NVIDIA Apache-2.0 source and downloaded asset LICENSE; body-model-specific notices also apply |
| MHR | [source](https://github.com/facebookresearch/MHR); SOMA-X's LOD1 checkpoint SHA-256 `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` | Meta Apache-2.0; license/notice retained in `crates/burn_mhr` |
| GEM-X | [source](https://github.com/NVlabs/GEM-X/tree/32992550dba114c62243fb55e361311972dce8f9), [checkpoint](https://huggingface.co/nvidia/GEM-X/tree/5ccf5ca3746c3620aa4016114f069a5f6ae399cd) | NVIDIA code Apache-2.0; preserve downloaded model LICENSE and upstream model terms. The source README additionally references the NVIDIA Open Model License for associated models. |
| DINOv3 | [source](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751) | Meta DINOv3 License, retained in `crates/burn_gem/LICENSE` |
| SAM 3D Body | [source](https://github.com/facebookresearch/sam-3d-body/tree/b5c765a0d89d789985e186d396315e7590887b94) | Meta SAM License, retained in `crates/burn_gem/LICENSE` and SAM bundle metadata |

**Built with Meta Llama 3.** Meta Llama 3: Copyright © Meta Platforms, Inc.
All Rights Reserved. The local text encoder uses the merged public checkpoint;
its tokenizer and model bytes are authenticated during conversion/loading.
No gated model access is required beyond the terms of the selected public export.

`burn_gem` uses a combined license file because its SAM/DINO ports are covered
by their respective upstream terms. Its `NOTICE` identifies the implementation
files and its `ATTRIBUTIONS.md` retains GEM-X's third-party notices, including
rotation-conversion attribution. These components are not represented as wholly
Apache-licensed. Keep these notices when redistributing code or packed models.

`tests/data/core27-rig.json` contains the Apache-2.0 neutral skeleton from the
pinned ARDY source. Numerical fixtures use downloaded models and are generated
offline; large checkpoints, source test photographs and reference meshes are
kept out of Git. Evidence reports record errors and provenance separately.
