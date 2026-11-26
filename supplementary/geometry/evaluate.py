"""Evaluate four geometry conditions from six validation-selected checkpoints per property."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from run import read, save, sha

HERE = Path(__file__).resolve().parent
SEEDS = (42, 44, 46)
CONDITIONS = {"A_prime": ("dft", "dft"), "B_prime": ("dft", "etkdg"),
              "C_prime": ("dft", "mmff"), "D": ("mmff", "mmff")}


def require(value, message):
    if not value:
        raise ValueError(message)


def metrics(prediction, truth):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    require(p.ndim == 1 and len(p) and p.shape == y.shape, "Invalid prediction/label shapes")
    require(np.isfinite(p).all() and np.isfinite(y).all(), "Nonfinite prediction/label")
    error = p - y
    return {"n": len(p), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def ordered_points(ids, points, expected_ids):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype.kind in "iu", "Point-cloud IDs must be integers")
    require(len(ids) == len(points) and len(set(ids.tolist())) == len(ids), "Duplicate/misaligned point-cloud IDs")
    require(len(ids) == len(expected_ids) and set(ids.tolist()) == set(expected_ids), "Point-cloud ID cohort mismatch")
    index = {int(mid): row for row, mid in enumerate(ids)}
    return np.asarray(points)[[index[mid] for mid in expected_ids]]


def validate_points(points, lengths, reference):
    require(points.shape == reference.shape and points.shape[2] == 9, "Point-cloud shape mismatch")
    require(points.dtype == np.float32 and np.isfinite(points).all(), "Invalid point-cloud dtype/values")
    mask = np.arange(points.shape[1])[None, :] < lengths[:, None]
    require(not np.count_nonzero(points[~mask]), "Point-cloud padding is not zero")
    types = points[:, :, 4:]
    require(np.isin(types, (0, 1)).all(), "Element channels are not one-hot")
    np.testing.assert_array_equal(types.sum(axis=2), mask.astype(np.float32))
    np.testing.assert_array_equal(types.sum(axis=1), reference[:, :, 4:].sum(axis=1))
    np.testing.assert_allclose(points[:, :, :3].sum(axis=1) / lengths[:, None], 0, atol=2e-6)
    np.testing.assert_allclose(points[:, :, 3], np.linalg.norm(points[:, :, :3], axis=2), rtol=2e-6, atol=2e-6)


def validate_result(result, expected_ids, expected_targets):
    np.testing.assert_array_equal(result["ids"], expected_ids, err_msg="Evaluation ID order changed")
    np.testing.assert_array_equal(result["targets"], expected_targets, err_msg="Evaluation labels changed")
    require(metrics(result["predictions"], result["targets"]) == result["metrics"], "Metrics do not match predictions")


def csv_bytes(rows):
    require(bool(rows), "Cannot write an empty result table")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_once(path, raw):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == raw, "Existing output differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def verify_receipt(root, receipt, binding):
    require(receipt["binding"] == binding, "Existing evaluation receipt provenance mismatch")
    for name, digest in receipt["files_sha256"].items():
        path = (Path(root) / name).resolve()
        require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Invalid receipt file")
        require(sha(path) == digest, "Existing evaluation output checksum mismatch: " + name)


def condition_result(root, label, seed, binding, expected_ids, expected_targets, producer):
    stem = f"{label}_seed{seed}"
    receipt_path = root / "receipts" / (stem + ".json")
    result_path = root / "results" / (stem + ".json")
    csv_path = root / "predictions" / (stem + ".csv")
    if receipt_path.exists():
        verify_receipt(root, read(receipt_path), binding)
        saved = read(result_path)
        require(saved["binding"] == binding, "Persisted result binding differs")
        validate_result(saved["result"], expected_ids, expected_targets)
        return saved["result"]
    if result_path.exists():
        saved = read(result_path)
        require(saved["binding"] == binding, "Uncommitted result provenance mismatch")
        result = saved["result"]
    else:
        result = producer()
        validate_result(result, expected_ids, expected_targets)
        save(result_path, {"binding": binding, "result": result})
    validate_result(result, expected_ids, expected_targets)
    rows = [{"id": int(mid), "target_original_units": float(y), "prediction_original_units": float(p)}
            for mid, y, p in zip(result["ids"], result["targets"], result["predictions"])]
    write_once(csv_path, csv_bytes(rows))
    files = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in (result_path, csv_path)}
    save(receipt_path, {"status": "complete", "binding": binding, "metrics": result["metrics"],
                        "files_sha256": files})
    return result


def aggregate(rows):
    require(len(rows) == 12, "Four conditions and three seeds are required")
    summary = []
    index = {}
    for label in CONDITIONS:
        selected = [r for r in rows if r["label"] == label]
        require(sorted(r["seed"] for r in selected) == list(SEEDS), "Missing or repeated evaluation seed")
        require(len({r["n"] for r in selected}) == 1, "Different condition cohort sizes")
        first = selected[0]
        record = {k: first[k] for k in ("experiment", "dataset", "target", "unit", "label")}
        record.update(n=first["n"], seeds=3)
        for metric in ("mae", "rmse"):
            values = [r[metric] for r in selected]
            require(np.isfinite(values).all(), "Nonfinite aggregate metric")
            record[metric + "_mean"] = statistics.mean(values)
            record[metric + "_sd"] = statistics.stdev(values)
        summary.append(record)
        index.update({(label, r["seed"]): r for r in selected})
    changes = []
    for other, base in (("B_prime", "A_prime"), ("C_prime", "B_prime"), ("C_prime", "A_prime"),
                        ("D", "C_prime"), ("D", "A_prime")):
        for metric in ("mae", "rmse"):
            denominators = [index[(base, seed)][metric] for seed in SEEDS]
            require(all(v > 0 for v in denominators), "Cannot divide by zero reference error")
            values = [100 * (index[(other, seed)][metric] / index[(base, seed)][metric] - 1) for seed in SEEDS]
            changes.append({"comparison": other + "_vs_" + base, "metric": metric,
                            "mean_change_pct": statistics.mean(values), "sd_change_pct": statistics.stdev(values),
                            **{f"seed_{seed}_pct": value for seed, value in zip(SEEDS, values)}})
    return summary, changes


def evaluate_experiment(work):
    import torch
    from torch.utils.data import DataLoader, Dataset

    plan = read(work / "experiment_plan.json")
    name = plan["experiment"]
    spec = plan["spec"]
    sys.path.insert(0, str(work / "src"))
    import train_fusion as trainer

    require(trainer.source_hashes() == spec["source_hashes"], "Trainer source hashes changed")
    root = work / "assessment"
    run_map = {(r["geometry"], r["seed"]): r for r in plan["runs"]}
    for run in plan["runs"]:
        status = read(Path(run["run_dir"]) / "training_summary.json")
        require(status["status"] == "trained" and not status["diagnostic"], "Complete all six formal fits first")
    prepared, args_map, configs = {}, {}, {}
    device = None
    for geometry in ("dft", "mmff"):
        run = run_map[(geometry, 42)]
        args = trainer.parse_args(run["argv"][3:])
        device = trainer.prepare_device(args) if device is None else device
        require(str(device) == "cuda:0", "All four conditions require the same CUDA backend")
        prepared[geometry] = trainer.prepare_dataset(args)
        dataset, ids, targets, collate, split, split_ids = prepared[geometry][:6]
        scaling = prepared[geometry][-1]
        require({k: len(v) for k, v in split_ids.items()} == spec["split_counts"], "Wrong evaluation split")
        for seed in SEEDS:
            run = run_map[(geometry, seed)]
            args = trainer.parse_args(run["argv"][3:])
            args.mode = "finalize"
            inputs = [args, Path(run["run_dir"]), dataset, ids, targets, scaling,
                      trainer.model_config(args, dataset.dims), split_ids]
            if spec["dataset"] == "QM8":
                inputs.append(prepared[geometry][-2])
            configs[(geometry, seed)] = trainer.verify_config(*inputs)
            args_map[(geometry, seed)] = args
    require(prepared["dft"][5] == prepared["mmff"][5] and prepared["dft"][-1] == prepared["mmff"][-1],
            "DFT/MMFF split or training-only scaler changed")
    for seed in SEEDS:
        left, right = configs[("dft", seed)], configs[("mmff", seed)]
        require(left["initialization_sha256"] == right["initialization_sha256"] and
                "full_model" in left["initialization_sha256"], "DFT/MMFF initialization is not paired")
    checkpoint_hashes = {f"{g}_seed{s}": sha(Path(r["run_dir"]) / "best.pt") for (g, s), r in run_map.items()}
    completion_binding = {"experiment": name, "plan_sha256": sha(work / "experiment_plan.json"),
                          "source_sha256": spec["source_hashes"], "checkpoint_sha256": checkpoint_hashes,
                          "device": "cuda:0", "precision": "float32", "seeds": list(SEEDS)}
    if (root / "completion.json").exists():
        verify_receipt(root, read(root / "completion.json"), completion_binding)
        print(json.dumps({"experiment": name, "status": "verified_already_complete"}))
        return

    dataset, _, targets, collate, split, split_ids = prepared["dft"][:6]
    scaling = prepared["dft"][-1]
    test_ids = split_ids["test"]
    expected_targets = targets[split["test"]]
    raw_index = {int(mid): i for i, mid in enumerate(dataset.arrays["ids"])}
    raw_rows = [raw_index[mid] for mid in test_ids]
    dft_points = dataset.arrays["points"][raw_rows]
    lengths = dataset.arrays["point_lengths"][raw_rows]
    other = prepared["mmff"][0]
    np.testing.assert_array_equal(other.ids, dataset.ids)
    np.testing.assert_array_equal(other.targets, dataset.targets)
    mmff_rows = {int(mid): i for i, mid in enumerate(other.arrays["ids"])}
    mmff_points = other.arrays["points"][[mmff_rows[mid] for mid in test_ids]]
    etkdg_path = Path(spec["data_dir"]) / "etkdg_test.npz"
    require(sha(etkdg_path) == spec["etkdg_test_sha256"], "ETKDG source archive changed")
    with np.load(etkdg_path, allow_pickle=False) as archive:
        etkdg_points = ordered_points(archive["ids"], archive["points"], test_ids)
    for points in (dft_points, etkdg_points, mmff_points):
        validate_points(points, lengths, dft_points)
    np.testing.assert_array_equal(etkdg_points[:, :, 4:], mmff_points[:, :, 4:])

    class CoordinateView(Dataset):
        def __init__(self, points):
            self.points = torch.from_numpy(np.asarray(points))

        def __len__(self):
            return len(test_ids)

        def __getitem__(self, index):
            sample = dataset[int(split["test"][index])]
            require(sample["id"] == test_ids[index], "Dataset test-ID order changed")
            sample["points"] = self.points[index]
            return sample

    rows = []
    for seed in SEEDS:
        finalized = {}
        for geometry in ("dft", "mmff"):
            run = Path(run_map[(geometry, seed)]["run_dir"])
            with trainer.directory_lock(run):
                trainer.finalize(args_map[(geometry, seed)])
            finalization = read(run / "finalization_result.json")
            require(finalization["checkpoint_sha256"] == checkpoint_hashes[f"{geometry}_seed{seed}"] and
                    finalization["config_sha256"] == sha(run / "config.json"), "Finalization provenance mismatch")
            validate_result(finalization["test"], test_ids, expected_targets)
            finalized[geometry] = finalization["test"]
        model = None
        for label, (train_geometry, test_geometry) in CONDITIONS.items():
            run = Path(run_map[(train_geometry, seed)]["run_dir"])
            binding = {**completion_binding, "label": label, "seed": seed,
                       "train_geometry": train_geometry, "test_geometry": test_geometry,
                       "selected_checkpoint_sha256": checkpoint_hashes[f"{train_geometry}_seed{seed}"],
                       "config_sha256": sha(run / "config.json"), "scaling": scaling,
                       "split_sha256": spec["split_sha256"], "cache_sha256": spec["cache_sha256"],
                       "etkdg_sha256": spec["etkdg_test_sha256"]}

            def produce():
                nonlocal model
                if label in ("A_prime", "D"):
                    return finalized[train_geometry]
                if model is None:
                    config = configs[("dft", seed)]
                    best = trainer.load_checkpoint(Path(run_map[("dft", seed)]["run_dir"]) / "best.pt")
                    require(best["config"] == config and best["split_ids"] == split_ids, "DFT best checkpoint binding mismatch")
                    model = trainer.MultimodalModel(**config["model"]).to(device)
                    model.load_state_dict(best["model_state"], strict=True)
                    require(all(p.dtype == torch.float32 for p in model.parameters()), "Model is not FP32")
                    del best
                points = etkdg_points if label == "B_prime" else mmff_points
                loader = DataLoader(CoordinateView(points), batch_size=128, shuffle=False, num_workers=0, collate_fn=collate)
                return trainer.evaluate(model, loader, device, scaling)

            result = condition_result(root, label, seed, binding, test_ids, expected_targets, produce)
            rows.append({"experiment": name, "dataset": spec["dataset"], "target": spec["target"],
                         "unit": spec["unit"], "label": label, "seed": seed,
                         "train_geometry": train_geometry, "test_geometry": test_geometry,
                         **result["metrics"], "checkpoint_sha256": binding["selected_checkpoint_sha256"]})
        del model
        torch.cuda.empty_cache()
    summary, changes = aggregate(rows)
    for filename, table in (("per_seed.csv", rows), ("summary.csv", summary), ("paired_changes.csv", changes)):
        write_once(root / filename, csv_bytes(table))
    require({f"{g}_seed{s}": sha(Path(r["run_dir"]) / "best.pt") for (g, s), r in run_map.items()} == checkpoint_hashes,
            "Checkpoint changed during evaluation")
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.name != "completion.json" and not p.name.endswith(".tmp")}
    save(root / "completion.json", {"status": "complete", "binding": completion_binding,
         "experiment": name, "conditions": 12, "files_sha256": files,
         "notes": ["A and D reuse finalized best-checkpoint predictions; B and C use A's identical checkpoint",
                   "All conditions use cuda:0 FP32; positive paired percentage means worse error"]})
    print(json.dumps({"experiment": name, "status": "complete", "condition_evaluations": 12}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    evaluate_experiment(parser.parse_args().work.resolve())
