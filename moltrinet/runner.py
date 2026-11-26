"""Train one molecular property and evaluate its best validation checkpoint."""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import QM9Dataset, collate_qm9
from .independent_feature_gate import configure_independent_gate
from .model import MultimodalModel


ROOT = Path(__file__).resolve().parents[1]
PAPER = json.loads((ROOT / "configs/paper.json").read_text())
MODALITIES = ("S", "G", "P", "SG", "SP", "GP", "SGP")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_model(cfg, dims, device):
    params = dict(cfg["architecture"])
    mode = cfg["gate"]
    branches = cfg["modalities"]
    if mode == "entmax15" and branches != "SGP":
        raise ValueError("The sparse experiment uses SGP.")
    fusion = ("gate" if mode == "entmax15" else
              "joint_softmax" if cfg["dense_context"] == "pairwise" else "embeddings")
    model = MultimodalModel(**params, vocab_size=dims["vocab_size"], smiles_length=dims["smiles_length"],
                            d_node=dims["node_dim"], d_edge=dims["edge_dim"], d_point_input=dims["point_dim"],
                            use_smiles="S" in branches, use_graph="G" in branches, use_point_cloud="P" in branches,
                            fusion_type=fusion)
    if mode == "entmax15":
        configure_independent_gate(model, "entmax15")
    return model.to(device)


def partitions(cfg, cache, split, phases):
    split = Path(split or ROOT / "splits" / f"{cfg['dataset']}_seed_{cfg['seed']}.json")
    parts = read(split)
    ids = [i for phase in ("train", "val", "test") for i in parts[phase]]
    if len(set(ids)) != len(ids) or any(not parts[k] for k in ("train", "val", "test")):
        raise ValueError("Split must have three nonempty, disjoint ID lists with unique molecules.")
    cache = Path(cache)
    record = read(cache / "metadata.json")
    if record.get("dataset", cfg["dataset"]).lower() != cfg["dataset"]:
        raise ValueError("Dataset mismatch.")
    schema = PAPER["datasets"][cfg["dataset"]]
    if record["dims"] != schema["dims"] or record["target_names"] != schema["target_names"]:
        raise ValueError("The cache feature dimensions or target column order differ from the dataset schema.")
    with np.load(cache / "cache.npz", allow_pickle=False) as archive:
        lookup = {int(mid): row for row, mid in enumerate(archive["ids"])}
    if set(ids) != set(lookup):
        raise ValueError("Split and input cache describe different cohorts.")
    rows = [lookup[mid] for phase in phases for mid in parts[phase]]
    data = QM9Dataset(cache, target=cfg["target"], indices=rows)
    offset, subsets = 0, {}
    for phase in phases:
        stop = offset + len(parts[phase])
        subsets[phase] = Subset(data, range(offset, stop))
        offset = stop
    return data, subsets, {"cache_sha256": record["cache_sha256"], "split_sha256": digest(split),
                           "target_names": record["target_names"], "dims": record["dims"]}


def arguments(batch, device):
    return tuple(batch[k].to(device) for k in ("smiles_ids", "graph", "points", "point_lengths"))


def standardized(batch, scaling, device):
    return ((batch["y"].reshape(-1) - scaling["mean"]) / scaling["std"]).float().to(device)


@torch.inference_mode()
def metrics(model, loader, scaling, device):
    model.eval()
    losses, absolute, squared, n = [], 0., 0., 0
    for batch in loader:
        pred = model(*arguments(batch, device)).reshape(-1)
        losses.append(float(F.l1_loss(pred, standardized(batch, scaling, device))))
        error = pred.cpu().double() * scaling["std"] + scaling["mean"] - batch["y"].flatten().double()
        absolute += float(error.abs().sum())
        squared += float(error.square().sum())
        n += error.numel()
    return {"n": n, "loss": float(np.mean(losses)), "mae": absolute / n, "rmse": math.sqrt(squared / n)}


def fit(args):
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be at least one.")
    ds = PAPER["datasets"][args.dataset]
    if args.target not in ds["targets"]:
        raise ValueError(f"Choose a target from {list(ds['targets'])}")
    if args.cmcr and (args.modalities != "SGP" or args.gate != "entmax15"):
        raise ValueError("The CMCR comparison uses the entmax15 SGP predictor.")
    if args.gate == "entmax15" and args.dense_context != "embeddings":
        raise ValueError("--dense-context applies to softmax.")
    seed = args.seed
    torch.set_num_threads(args.threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    tr = dict(PAPER["training"])
    tr.update(max_epochs=args.epochs or tr["max_epochs"], torch_num_threads=args.threads)
    cfg = dict(dataset=args.dataset, target=args.target, unit=ds["targets"][args.target],
               gate=args.gate, dense_context=args.dense_context, modalities=args.modalities, cmcr=args.cmcr,
               seed=seed, architecture=PAPER["architecture"], training=tr)
    data, sets, binding = partitions(cfg, args.cache, args.split, ("train", "val"))
    if args.split is None and (ds["cache_sha256"] != binding["cache_sha256"]
                               or ds["split_sha256"][str(seed)] != binding["split_sha256"]):
        raise ValueError("Paper run requires the reference cache and saved split. Supply --split for an explicitly custom cohort.")
    train_y = data.targets[:len(sets["train"])]
    scaling = {"mean": float(train_y.mean()), "std": float(train_y.std(ddof=0))}
    if not scaling["std"] > 0:
        raise ValueError("Training labels have zero variance.")
    loaders = {}
    for index, (phase, subset) in enumerate(sets.items()):
        generator = torch.Generator().manual_seed(seed + index) if args.gate == "entmax15" else None
        loaders[phase] = DataLoader(subset, batch_size=tr["batch_size"], shuffle=phase == "train",
                                    generator=generator, collate_fn=collate_qm9)
    model = build_model(cfg, data.dims, args.device)
    params = list(model.parameters())
    if args.cmcr:
        from supplementary.cmcr import ORIGINAL_CMCR, build_projectors
        projectors = build_projectors(ORIGINAL_CMCR, args.device)
        params += list(projectors.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr["learning_rate"], betas=tuple(tr["betas"]),
                                  weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", **tr["scheduler"])
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    write(run / "config.json", cfg)
    history, best, early_best, stale = [], math.inf, math.inf, 0
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train()
        if args.cmcr:
            projectors.train()
        training_losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            inputs = arguments(batch, args.device)
            if args.cmcr:
                prediction, (s, g, _) = model(*inputs, return_embeddings=True)
                contrastive = projectors(s, g, ORIGINAL_CMCR["temperature"]) * ORIGINAL_CMCR["weight"]
            else:
                prediction, contrastive = model(*inputs), 0.
            loss = F.l1_loss(prediction.flatten(), standardized(batch, scaling, args.device)) + contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            training_losses.append(float(loss.detach()))
        val = metrics(model, loaders["val"], scaling, args.device)
        score = val["loss"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(training_losses)), "val": val})
        scheduler.step(score)
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "best_epoch": epoch, "scaling": scaling, "inputs": binding,
                        "config_sha256": digest(run / "config.json"),
                        "cmcr_projectors": projectors.state_dict() if args.cmcr else None}, run / "best.pt")
        write(run / "history.json", history)
        if score < early_best - tr["early_stop_min_delta"]:
            early_best, stale = score, 0
        else:
            stale += 1
        if stale >= tr["early_stop_patience"]:
            break
    return {"epochs": epoch, "best_val_loss": best, "checkpoint": str(run / "best.pt")}


def load_run(run, cache, split=None, phase="val", device="cpu", checkpoint=None):
    run = Path(run)
    cfg = read(run / "config.json")
    archive = None
    if checkpoint is not None:
        file_hash = digest(checkpoint)
        with (ROOT / "results/entmax_checkpoints.csv").open(encoding="utf-8", newline="") as handle:
            archive = next((r for r in csv.DictReader(handle) if r["checkpoint_sha256"] == file_hash), None)
        if archive is not None:
            if (archive["dataset"], archive["target"], int(archive["seed"])) != (cfg["dataset"], cfg["target"], cfg["seed"]):
                raise ValueError("The archived checkpoint is for a different dataset, target, or seed.")
            cfg = dict(cfg, gate="entmax15", architecture=PAPER["architecture"], dense_context="embeddings", modalities="SGP")
    data, subsets, binding = partitions(cfg, cache, split, (phase,))
    saved = torch.load(checkpoint or run / "best.pt", map_location="cpu", weights_only=archive is None)
    if "inputs" in saved:
        if saved["inputs"] != binding or saved["config_sha256"] != digest(run / "config.json"):
            raise ValueError("Checkpoint/config/input mismatch.")
    else:
        if (archive is None or archive["split_sha256"] != binding["split_sha256"]
                or saved["binding"]["cache_sha256"] != binding["cache_sha256"]
                or saved["binding"]["split_sha256"] != binding["split_sha256"]):
            raise ValueError("Archived checkpoint/inputs differ.")
    model = build_model(cfg, data.dims, device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, DataLoader(subsets[phase], batch_size=cfg["training"]["batch_size"],
                             shuffle=False, collate_fn=collate_qm9), saved["scaling"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("fit")
    train.add_argument("--dataset", required=True, choices=PAPER["datasets"])
    train.add_argument("--target", required=True)
    train.add_argument("--seed", type=int, choices=(42, 44, 46), required=True)
    train.add_argument("--gate", choices=("softmax", "entmax15"), required=True)
    train.add_argument("--dense-context", choices=("embeddings", "pairwise"), default="embeddings")
    train.add_argument("--modalities", choices=MODALITIES, default="SGP")
    train.add_argument("--cmcr", action="store_true")
    train.add_argument("--epochs", type=int)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--phase", choices=("val", "test"), required=True)
    for cmd in (train, ev):
        cmd.add_argument("--threads", type=int, default=PAPER["training"]["torch_num_threads"])
        cmd.add_argument("--cache", required=True)
        cmd.add_argument("--split", help="Explicit custom split JSON. Defaults to the saved paper split.")
        cmd.add_argument("--run", required=True, type=Path)
        cmd.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "fit":
        result = fit(args)
    else:
        torch.set_num_threads(args.threads)
        cfg = read(args.run / "config.json")
        model, loader, scale = load_run(args.run, args.cache, args.split, args.phase, args.device)
        result = dict(metrics(model, loader, scale, args.device), phase=args.phase, unit=cfg["unit"],
                      target=cfg["target"], seed=cfg["seed"])
        write(args.run / f"{args.phase}_metrics.json", result)
    print(json.dumps(result, allow_nan=False))
