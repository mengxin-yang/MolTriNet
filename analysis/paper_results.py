"""Rebuild Tables 1, 2, S5 and the figure data for Figures 2–5."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("S", "G", "P", "SG", "SP", "GP", "SGP")


def read(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(rows, *columns):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[c] for c in columns)].append(row)
    return groups


def seeds(rows):
    if sorted(int(row["seed"]) for row in rows) != [42, 44, 46]:
        raise ValueError("Expected one observation per training seed")
    return rows


def avg(rows, col):
    return mean(float(row[col]) for row in rows)


def main(inputs, output):
    output.mkdir(exist_ok=True, parents=True)
    abl = group(read(inputs / "modality_test_metrics.csv"), "dataset", "target", "modalities")
    pairs = group(read(inputs / "fusion_test_metrics.csv"), "dataset", "target")
    support = group(read(inputs / "support_validation.csv"), "dataset", "target")
    for ds in ("qm9", "qm8"):
        figs = {2: [], 3: [], 4: []}
        table = []
        for literature in read(inputs / f"literature_{ds}.csv"):
            target = literature["target"]
            subset = {m: avg(seeds(abl[ds, target, m]), "mae") for m in MODALITIES}
            best = min(subset, key=subset.get)
            figs[2].extend(dict(target=target, modalities=m, mean_mae=v,
                                relative_mae=v/subset[best], best=m == best)
                           for m, v in subset.items())
            observations = seeds(pairs[ds, target])
            dense = avg(observations, "baseline_mae")
            sparse = avg(observations, "entmax_mae")
            if dense != subset["SGP"]:
                raise ValueError(f"Inconsistent softmax baseline for {target}")
            table.append(dict(unit=observations[0]["unit"], **literature, softmax=dense, entmax15=sparse))
            figs[3].append(dict(target=target, softmax=dense, entmax15=sparse,
                                improvement_percent=100*(1-sparse/dense)))
            for n in (1, 2, 3):
                col = f"p{n}"
                values = [float(r[col]) for r in seeds(support[ds,target])]
                figs[4].append(dict(target=target, active_modalities=n, mean=mean(values),
                                    minimum=min(values), maximum=max(values)))
        for num, rows in figs.items():
            write(output/f"figure{num}_{ds}.csv", rows)
        write(output/f"table_s5_{ds}.csv", table)
        write(output/f"table_{1 if ds == 'qm9' else 2}.csv",
              [{k:v for k,v in r.items() if k!="entmax15"} for r in table])
        counts = Counter(len(r["modalities"]) for r in figs[2] if r["best"])
        gains = sum(r["improvement_percent"]>0 for r in figs[3])
        print(f"{ds.upper()}  best single/pair/triple: {[counts[n] for n in (1,2,3)]}; "
              f"lower entmax MAE: {gains}/{len(table)}")

    curves, bars = [], []
    gate = group(read(inputs / "gate_validation.csv"), "target", "modality", "seed")
    deletion = group(read(inputs / "deletion_validation.csv"), "target", "modality", "seed", "strategy", "ratio")
    for target in ("heat_capacity_298K","electronic_spatial_extent"):
        for mod in ("S","G","P"):
            for seed in ("42","44","46"):
                full = deletion[target,mod,seed,"high_abs_contribution","1.0"][0]
                base, endpoint = float(full["baseline_mae"]), float(full["mae"])
                g, = gate[target,mod,seed]
                bars.append(dict(target=target, modality=mod, seed=seed,
                                 active=float(g["active_fraction"]), magnitude=float(g["mean_abs_weighted_share"]),
                                 full_deletion_mae_ratio=endpoint/base))
                for strategy in ("high_abs_contribution","random"):
                    for ratio in ("0.0","0.25","0.5","1.0"):
                        rows = deletion[target,mod,seed,strategy,ratio]
                        expected = 20 if strategy=="random" else 1
                        if len(rows)!=expected or len({r["random_seed"] for r in rows})!=expected:
                            raise ValueError("Incomplete or duplicated deletion repetitions")
                        curves.append(dict(target=target, modality=mod, seed=seed, strategy=strategy,
                                           ratio=float(ratio), normalized_delta=(avg(rows,"mae")-base)/(endpoint-base)))
    write(output/"figure5_bars_per_seed.csv", bars)
    write(output/"figure5_curves_per_seed.csv", curves)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT/"results")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.inputs, args.out)
