"""Prepare a property-specific runtime and train the six paired geometry models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(experiment, data, work):
    properties = read(HERE / "properties.json")
    spec = properties[experiment].copy()
    dataset = spec["dataset"].lower()
    fixed = read(HERE / "splits" / (dataset + "_matched.json"))
    cache_hashes = {}
    split_hashes = {}
    for condition in ("dft", "mmff"):
        folder = data / condition
        meta = read(folder / "metadata.json")
        if read(folder / "fixed_split_ids.json") != fixed:
            raise ValueError("Matched split IDs or order differ from the geometry study")
        split_hashes[condition] = sha(folder / "fixed_split_ids.json")
        if split_hashes[condition] != meta["fixed_split_sha256"]:
            raise ValueError("Split digest differs from its metadata")
        if (meta["dataset"].lower() != dataset or meta["geometry_condition"] != condition
                or meta["target_names"][spec["target_column"]] != spec["target"]
                or meta["target_units"][spec["target"]] != spec["unit"]):
            raise ValueError("Coordinate source, dataset, target column or unit differs")
        cache_hashes[condition] = sha(folder / "cache.npz")
        if cache_hashes[condition] != meta["cache_sha256"]:
            raise ValueError("Cache digest differs from its metadata")
    etkdg_hash = sha(data / "etkdg_test.npz")
    work.mkdir(parents=True, exist_ok=False)
    code = work / "src"
    code.mkdir()
    files = ("Model.py", "independent_feature_gate.py", "train_fusion.py", "data.py")
    for name in files:
        shared = HERE / "templates" / name
        shutil.copyfile(shared if shared.is_file() else HERE / "templates" / dataset / name, code / name)
    save(code / "property.json", {"target": spec["target"], "unit": spec["unit"]})
    spec.update(source_hashes={name: sha(code / name) for name in files}, split_sha256=split_hashes,
                data_dir=str(data), cache_sha256=cache_hashes, etkdg_test_sha256=etkdg_hash)
    runs = []
    for seed in (42, 44, 46):
        for condition in ("dft", "mmff"):
            dest = work / "runs" / condition / f"seed_{seed}"
            argv = [sys.executable, "-u", str(code / "train_fusion.py"), "--mode", "train",
                    "--data", str(data / condition), "--out", str(dest), "--seed", str(seed),
                    "--fusion", "gate", "--device", "cuda:0", "--split-seed", "20260927",
                    "--epochs", "200", "--min-epochs", "50", "--patience", "30", "--max-minutes", "420"]
            runs.append({"seed": seed, "geometry": condition, "run_dir": str(dest), "argv": argv})
    save(work / "experiment_plan.json", {"experiment": experiment, "spec": spec, "runs": runs})


def train(work):
    for run in read(work / "experiment_plan.json")["runs"]:
        folder = Path(run["run_dir"])
        if (folder / "training_summary.json").exists():
            result = read(folder / "training_summary.json")
            if result["status"] != "trained" or result["diagnostic"] or not (folder / "best.pt").is_file():
                raise ValueError(f"Incomplete fit at {folder}")
            continue
        args = run["argv"]
        if (folder / "last.pt").is_file():
            args = args + ["--resume", str(folder / "last.pt")]
        subprocess.run(args, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("prepare")
    init.add_argument("--experiment", choices=list(read(HERE / "properties.json")), required=True)
    init.add_argument("--data", type=Path, required=True, help="Generated data directory")
    init.add_argument("--work", type=Path, required=True, help="New experiment directory")
    commands.add_parser("train").add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.experiment, args.data.resolve(), args.work.resolve())
    else:
        train(args.work.resolve())

