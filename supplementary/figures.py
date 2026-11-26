"""Regenerate the SI summaries and plots from per-seed measurements."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
QM9 = {"dipole_moment": "μ", "isotropic_polarizability": "α", "energy_HOMO": "HOMO",
       "gap": "gap", "electronic_spatial_extent": "⟨R²⟩", "zero_point_vibrational_energy": "ZPVE",
       "internal_energy_0K_atom_kcal_mol": "U₀", "heat_capacity_298K": "Cᵥ"}
QM8 = [f"{target}-{method}" for target in ("E1", "E2", "f1", "f2") for method in ("CC2", "PBE0", "CAM")]
SEEDS = {42, 44, 46}


def read(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def grouped(rows, fields, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    for key, values in groups.items():
        if len(values) != 3 or {int(row[seed]) for row in values} != SEEDS:
            raise ValueError(f"Expected three training seeds for {key}")
    return groups


def summaries():
    geometry = grouped(read("geometry_per_seed.csv"), ["target", "label"], "seed")
    prediction = grouped(read("cmcr_prediction_per_seed.csv"), ["target"], "seed")
    alignment = grouped(read("cmcr_alignment_per_seed.csv"), ["Target", "Space"], "Seed")
    result = []
    for target in (*QM9, *QM8):
        rows = prediction[(target,)]
        summary = {"dataset": rows[0]["dataset"], "target": target, "unit": rows[0]["unit"]}
        for label, code in (("A","A_prime"), ("B","B_prime"), ("C","C_prime"), ("D","D")):
            for metric in ("mae", "rmse"):
                summary[f"{label}_{metric}"] = mean(float(r[metric]) for r in geometry[(target,code)])
        for metric in ("mae", "rmse"):
            summary[f"D_vs_A_{metric}_pct"] = 100 * (summary[f"D_{metric}"] / summary[f"A_{metric}"] - 1)
        for mode in ("on","off"):
            summary[f"cmcr_{mode}_mae"] = mean(float(r[f"cmcr_{mode}_mae"]) for r in rows)
        summary["cmcr_change_pct"] = 100 * (summary["cmcr_on_mae"] / summary["cmcr_off_mae"] - 1)
        for space in ("Encoder_h","Projection_z"):
            summary[space+"_gap"] = mean(float(r["Cosine_Gap"]) for r in alignment[(target,space)])
        summary["Projection_z_retrieval_pct"] = mean(float(r["Bidirectional_Top1_pct"])
                                                     for r in alignment[(target,"Projection_z")])
        result.append(summary)
    return result, alignment


def save(fig, out, name):
    for ext in ("png","pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw(out, cache=None):
    out.mkdir(parents=True, exist_ok=True)
    rows, alignment = summaries()
    write(out / "supplement_summary.csv", rows)
    corr_inputs = read("qm9_pearson.csv")
    names = [r["target"] for r in corr_inputs]
    correlation = np.array([[float(r[key]) for key in names] for r in corr_inputs])
    if cache is not None:
        import json
        metadata = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        with np.load(cache / "cache.npz", allow_pickle=False) as archive:
            y = archive["targets"][:, [metadata["target_names"].index(k) for k in names]]
        if len(y) != 133885:
            raise ValueError("Figure S4 uses all 133885 QM9 rows")
        np.testing.assert_allclose(np.corrcoef(y.T), correlation, rtol=0, atol=1e-10)
    labels = ["A","B","C","μ","α","HOMO","LUMO","gap","⟨R²⟩","ZPVE","U₀","U₂₉₈","H₂₉₈","G₂₉₈","Cᵥ"]
    fig, ax = plt.subplots(figsize=(8,7), layout="constrained")
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for (i,j), value in np.ndenumerate(correlation):
        ax.text(j,i,f"{value:.2f}",ha="center",va="center",fontsize=6,
                color="white" if abs(value)>.7 else "black")
    fig.colorbar(im,ax=ax,label="Pearson correlation")
    save(fig,out,"figure_S4")
    f5, a5 = plt.subplots(1,2,figsize=(10,6),layout="constrained")
    f6, a6 = plt.subplots(2,2,figsize=(11,11),layout="constrained")
    f7, a7 = plt.subplots(1,2,figsize=(9,6),sharex=True,layout="constrained")
    for i, ds in enumerate(("qm9","qm8")):
        records = [r for r in rows if r["dataset"] == ds]
        targets = [r["target"] for r in records]
        names = [QM9.get(t,t) for t in targets]
        ypos = np.arange(len(targets))
        ax = a5[i]
        for j, row in enumerate(records):
            x,y = row["D_vs_A_mae_pct"],row["D_vs_A_rmse_pct"]
            ax.plot([x,y],[j-.1,j+.1],color="0.7",zorder=0)
        for key,offset,color,marker in (("mae",-.1,"#23737a","o"),("rmse",.1,"#b85c36","s")):
            ax.scatter([r[f"D_vs_A_{key}_pct"] for r in records],ypos+offset,
                       color=color,marker=marker,label=key.upper())
        ax.axvline(0,c="0.4",ls="--",lw=.8)
        ax.set_xlabel("Error change, D relative to A (%)")
        ax.legend()
        pred = a7[i]
        pred.barh(ypos,[r["cmcr_change_pct"] for r in records],color="#38767a")
        pred.axvline(0,c="black",lw=.7)
        pred.set_xlabel("Change in test MAE with CMCR (%)")
        for j, target in enumerate(targets):
            for space,offset,color in (("Encoder_h",-.12,"#ad5831"),("Projection_z",.12,"#23737a")):
                metrics = alignment[(target,space)]
                for k,axis in enumerate(a6[i]):
                    if k and space != "Projection_z":
                        continue
                    values = [float(r["Cosine_Gap" if k==0 else "Bidirectional_Top1_pct"]) for r in metrics]
                    pos = j + (offset if k==0 else 0)
                    axis.scatter(values, pos+np.array([-.046,0,.046]),s=17,facecolors="none",edgecolors=color)
                    axis.scatter(mean(values),pos,s=24,marker="D",color=color,
                                 label=space if j==0 and k==0 else None)
        a6[i,0].set_xlabel("Matched − mismatched cosine")
        a6[i,1].set_xlabel("Bidirectional projection-space top-1 (%)")
        a6[i,0].legend(fontsize=8)
        for axes in (ax, pred, *a6[i]):
            axes.set_yticks(ypos, names)
            axes.invert_yaxis()
            axes.set_title(ds.upper())
    for fig, name in ((f5,"figure_S5"),(f6,"figure_S6"),(f7,"figure_S7")):
        save(fig,out,name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--qm9-cache", type=Path, help="Validate S4 using all original molecular labels")
    args = parser.parse_args()
    draw(args.out, args.qm9_cache)
