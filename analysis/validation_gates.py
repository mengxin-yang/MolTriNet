"""Export validation gate weights and measure fixed-gate channel deletion (Figures 4–5)."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from moltrinet.independent_feature_gate import IndependentFeatureFusion
from moltrinet.runner import arguments, load_run


RATIOS = (0., .25, .5, 1.)
RANDOM_SEEDS = range(2026110700, 2026110720)
MODALITIES = ("S", "G", "P")


def channel_order(weights, values, ids, strategy, random_seed=None):
    """Order strictly positive-weight channels; ties resolve by channel index."""
    weights, values, ids = np.asarray(weights), np.asarray(values), np.asarray(ids)
    if weights.shape != values.shape or weights.ndim != 2 or len(ids) != len(weights):
        raise ValueError("Expected matching [sample, channel] arrays and sample IDs")
    if not np.isfinite(weights).all() or not np.isfinite(values).all() or (weights < 0).any():
        raise ValueError("Invalid weight/value arrays")
    orders = []
    for weight, value, molecule_id in zip(weights, values, ids):
        active = np.flatnonzero(weight > 0)
        if strategy == "random":
            if random_seed is None:
                raise ValueError("Random deletion requires an explicit fixed seed")
            rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), int(molecule_id)]))
            order = rng.permutation(active)
        else:
            score = {"high_weight": -weight, "low_weight": weight,
                     "high_abs_contribution": -np.abs(weight * value)}.get(strategy)
            if score is None:
                raise ValueError(f"Unknown strategy: {strategy}")
            order = active[np.argsort(score[active], kind="stable")]
        orders.append(order)
    return orders


def deletion_mask(orders, ratio, width):
    if ratio not in RATIOS:
        raise ValueError("Only the predeclared deletion ratios are permitted")
    mask = np.zeros((len(orders), width), dtype=bool)
    for row, order in enumerate(orders):
        mask[row, order[:math.ceil(ratio * len(order))]] = True
    return mask


def deleted_prediction(fusion, details, mask, modality):
    """Use original q/v, subtract one branch's contribution, retain original head."""
    mask = torch.as_tensor(mask, dtype=torch.bool, device=details["mixed"].device)
    contribution = details["weights"][:, modality] * details["features"][:, modality]
    mixed = details["mixed"] - mask.to(contribution.dtype) * contribution
    return fusion._predict(mixed).reshape(-1), mixed


@contextmanager
def observe_fusion(fusion):
    original, captured = fusion.forward, []
    had_instance_forward = "forward" in fusion.__dict__

    def observe(*inputs, **kwargs):
        output, details = original(*inputs, **kwargs, return_details=True)
        captured.append(details)
        return output

    fusion.forward = observe
    try:
        yield captured
    finally:
        if had_instance_forward:
            fusion.forward = original
        else:
            del fusion.forward


@torch.inference_mode()
def export(args):
    model, loader, scaling = load_run(args.run, args.cache, args.split, phase="val",
                                      device=args.device, checkpoint=args.checkpoint)
    model.eval()
    fusion = model.fusion_model
    if not isinstance(fusion, IndependentFeatureFusion) or fusion.mode != "entmax15":
        raise ValueError("Select a three-modality MolTriNet (1.5-entmax) checkpoint.")
    all_ids, all_weights = [], []
    errors = defaultdict(lambda: np.zeros(2, dtype=np.float64))
    magnitude_share = np.zeros(3, dtype=np.float64)
    defined_shares = 0
    baseline_abs = 0.
    with observe_fusion(fusion) as captured:
        for batch in loader:
            captured.clear()
            prediction = model(*arguments(batch, args.device)).reshape(-1)
            (details,) = captured
            q, v = details["weights"], details["features"]
            if not torch.isfinite(q).all() or (q < 0).any() or not torch.isfinite(v).all():
                raise ValueError("Invalid validation weights or features")
            torch.testing.assert_close(q.sum(1), torch.ones_like(q[:, 0]), atol=2e-6, rtol=0.)
            ids = batch["id"].cpu().numpy()
            target = batch["y"].reshape(-1).cpu().double().numpy()
            def residual(pred):
                return pred.double().cpu().numpy() * scaling["std"] + scaling["mean"] - target
            baseline_abs += np.abs(residual(prediction)).sum()
            all_ids.append(ids)
            all_weights.append(q.cpu().numpy())
            mass = (q * v).abs().double().sum(2).cpu().numpy()
            total = mass.sum(1)
            valid = total > 0
            magnitude_share += (mass[valid] / total[valid, None]).sum(0)
            defined_shares += int(valid.sum())
            if not args.delete:
                continue
            for m, label in enumerate(MODALITIES):
                weights, values = q[:, m].cpu().numpy(), v[:, m].cpu().numpy()
                excluded, _ = deleted_prediction(fusion, details, weights == 0, m)
                torch.testing.assert_close(excluded, prediction, rtol=0., atol=2e-6)
                full, _ = deleted_prediction(fusion, details, weights > 0, m)
                for strategy, seeds in (("high_abs_contribution", [None]), ("random", RANDOM_SEEDS)):
                    for seed in seeds:
                        orders = channel_order(weights, values, ids, strategy, seed)
                        for ratio in RATIOS:
                            mask = deletion_mask(orders, ratio, weights.shape[1])
                            pred, _ = deleted_prediction(fusion, details, mask, m)
                            if ratio == 0. or ratio == 1.:
                                torch.testing.assert_close(pred, prediction if ratio == 0. else full,
                                                           rtol=0., atol=2e-6)
                            err = residual(pred)
                            errors[label, strategy, seed, ratio] += [np.abs(err).sum(), np.square(err).sum()]
    weights = np.concatenate(all_weights)
    ids = np.concatenate(all_ids)
    split_ids = json.loads(args.split.read_text(encoding="utf-8"))["val"]
    if not np.array_equal(ids, np.asarray(split_ids)):
        raise ValueError("Validation IDs do not match the dataset order.")
    n = len(ids)
    positive = weights > 0
    support = positive.sum(1)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "validation_weights.npz", molecule_id=ids, weights=weights,
                        modality=np.array(MODALITIES))
    summary = dict(n=n, channels=weights.shape[2], baseline_mae=baseline_abs / n,
                   phase="val", modality_order=MODALITIES,
                   support_k1_k2_k3=[float((support == k).mean()) for k in (1, 2, 3)],
                   active_fraction=positive.mean(axis=(0, 2)).tolist(),
                   mean_weight=weights.mean(axis=(0, 2), dtype=np.float64).tolist(),
                   whole_branch_zero=(~positive.any(axis=2)).mean(axis=0).tolist(),
                   defined_magnitude_shares=defined_shares,
                   mean_magnitude_share=(magnitude_share / defined_shares).tolist() if defined_shares else None)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    if args.delete:
        with (args.out / "deletion.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["modality", "strategy", "random_seed", "ratio", "baseline_mae", "mae", "rmse", "delta_mae"])
            for key, (abs_sum, squared_sum) in errors.items():
                writer.writerow([*key, baseline_abs / n, abs_sum / n, math.sqrt(squared_sum / n),
                                 (abs_sum - baseline_abs) / n])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("run", "cache", "split", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, help="Optional trusted checkpoint path")
    parser.add_argument("--delete", action="store_true", help="Also evaluate the two Figure 5 deletion strategies")
    export(parser.parse_args())
