#!/usr/bin/env python3
"""Describe generated clips without claiming a semantic benchmark score.

Inputs: ardy-run clip-N.json and request-N.json pairs. Reports FK diagnostics,
window-boundary continuity and waypoint error. Ground is Y=0; no foot locking
or postprocessing is applied. Plots are optional, standalone artifacts.
"""
import argparse
import hashlib
import json
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation


def joints(clip):
    frames, rig = clip["frames"], clip["rig"]["joints"]
    local = np.asarray([f["local_rotations"] for f in frames])
    positions = np.zeros((len(frames), len(rig), 3))
    rotations = [None] * len(rig)
    for j, joint in enumerate(rig):
        q = Rotation.from_quat(local[:, j])
        parent = joint["parent"]
        if parent is None:
            rotations[j] = q
            positions[:, j] = [f["root_translation"] for f in frames]
        else:
            rotations[j] = rotations[parent] * q
            positions[:, j] = positions[:, parent] + rotations[parent].apply(joint["offset"])
    return positions


def metrics(clip, request):
    p = joints(clip)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite FK positions")
    names = [j["name"] for j in clip["rig"]["joints"]]
    feet = [names.index(n) for n in ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]]
    contact = np.asarray([f["foot_contacts"] for f in clip["frames"]], dtype=bool)
    velocity = np.linalg.norm(np.diff(p[:, feet][:, :, [0, 2]], axis=0), axis=-1) * clip["fps"]
    sliding = velocity[contact[:-1] & contact[1:]]
    root_step = np.linalg.norm(np.diff(p[:, 0], axis=0), axis=-1)
    joint_step = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    relative = p - p[:, :1]
    hand_speed = np.linalg.norm(np.diff(relative[:, names.index("RightHand")], axis=0), axis=-1) * clip["fps"]
    points = request["waypoints"]
    error = [float(np.linalg.norm(p[w["frame"], 0, [0, 2]] - np.asarray(w["position"])[[0, 2]])) for w in points]
    return dict(frames=len(p), fps=clip["fps"], root_distance_m=float(root_step.sum()),
                root_height_range_m=[float(p[:, 0, 1].min()), float(p[:, 0, 1].max())],
                foot_min_height_m=float(p[:, feet, 1].min()),
                foot_contact_samples=int(sliding.size), contact_foot_speed_mean_mps=float(sliding.mean()) if sliding.size else None,
                root_step_max_m=float(root_step.max()), joint_step_max_m=float(joint_step.max()),
                window_boundary_root_step_m=[float(root_step[i - 1]) for i in range(40, len(p), 40)],
                window_boundary_joint_step_max_m=[float(joint_step[i - 1].max()) for i in range(40, len(p), 40)],
                root_relative_right_hand_speed_mean_mps=float(hand_speed.mean()),
                waypoint_error_m=error, waypoint_rmse_m=float(np.sqrt(np.mean(np.square(error)))) if error else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=pathlib.Path, required=True)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--assert-examples", action="store_true", help="Regression gates for the six documented seed-42 examples only")
    args = ap.parse_args()
    records = []
    for path in sorted(args.directory.glob("clip-*.json")):
        label = path.stem.removeprefix("clip-")
        clip = json.loads(path.read_text())
        request = json.loads((args.directory / f"request-{label}.json").read_text())
        records.append(dict(case=label, prompt=request["prompt"], request=request,
                            clip_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), metrics=metrics(clip, request)))
    if not records:
        raise ValueError("No generated clips")
    if (args.directory / "clip-path.json").exists() and (args.directory / "clip-0.json").exists():
        reference_request = json.loads((args.directory / "request-path.json").read_text())
        reference_clip = json.loads((args.directory / "clip-0.json").read_text())
        baseline = metrics(reference_clip, reference_request)["waypoint_rmse_m"]
    else:
        baseline = None
    report = dict(quality_status="diagnostic examples, no human-rated or dataset-level semantic score",
                  method="Y-up FK in metres; predicted contacts; no foot correction; fixed seed 42",
                  unconditioned_waypoint_rmse_m=baseline, records=records)
    if args.assert_examples:
        by_case = {r["case"]: r for r in records}
        expected = ["walk forward calmly", "walk forward confidently with long energetic strides", "sneak forward cautiously", "wave with the right hand while standing", "jog forward quickly"]
        for i, prompt in enumerate(expected):
            record = by_case[str(i)]
            if record["prompt"] != prompt or record["request"]["seed"] != 42 or record["request"]["frames"] != 120:
                raise ValueError("Example gates require the documented prompts/seed/duration")
        distance = lambda key: by_case[key]["metrics"]["root_distance_m"]
        assert distance("4") > distance("0") > distance("2") > distance("3")
        assert distance("3") < 0.3
        assert by_case["path"]["metrics"]["waypoint_rmse_m"] < 0.1
        assert by_case["path"]["metrics"]["waypoint_rmse_m"] < baseline * 0.2
        for record in records:
            m = record["metrics"]
            assert m["foot_min_height_m"] > -0.1
            assert m["contact_foot_speed_mean_mps"] is not None and m["contact_foot_speed_mean_mps"] < 0.15
            assert max(m["window_boundary_root_step_m"]) < 0.25
        report["example_regression_gates"] = "passed; diagnostic examples only"
    (args.directory / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, record in zip(axes.flat, records):
            clip = json.loads((args.directory / f"clip-{record['case']}.json").read_text())
            p = joints(clip)
            ax.plot(p[:, 0, 0], p[:, 0, 2], label="Generated root")
            wp = record["request"]["waypoints"]
            if wp:
                a = np.asarray([w["position"] for w in wp])
                ax.plot(a[:, 0], a[:, 2], "o--", label="Requested")
            ax.set(title=record["prompt"], xlabel="World X (m)", ylabel="World Z (m)", aspect="equal")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.savefig(args.directory / "trajectories.png", dpi=140)


if __name__ == "__main__":
    main()
