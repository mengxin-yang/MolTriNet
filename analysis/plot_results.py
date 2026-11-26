"""Plot Figures 2–5 from the output of paper_results.py."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from paper_results import MODALITIES, group, read

LABEL = dict(dipole_moment="μ", isotropic_polarizability="α", energy_HOMO="HOMO",
             electronic_spatial_extent="⟨R²⟩", zero_point_vibrational_energy="ZPVE",
             internal_energy_0K_atom_kcal_mol="U₀", heat_capacity_298K="Cᵥ")
COLORS = ("#3877a7", "#579785", "#cc8455")


def finish(fig, folder, name):
    fig.savefig(folder/f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(folder):
    tables = [read(folder/f"figure2_{ds}.csv") for ds in ("qm9","qm8")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), width_ratios=(3,3,1.5), layout="constrained")
    maximum = max(float(r["relative_mae"]) for tab in tables for r in tab)
    counts = Counter()
    for ax, ds, tab in zip(axes, ("QM9","QM8"), tables):
        targets = list(dict.fromkeys(r["target"] for r in tab))
        values = np.array([float(r["relative_mae"]) for r in tab]).reshape(-1,7)
        artist = ax.imshow(values, cmap="Blues", norm=LogNorm(1,maximum), aspect="auto")
        for i, target in enumerate(targets):
            winner = int(values[i].argmin())
            counts[len(MODALITIES[winner])] += 1
            ax.add_patch(Rectangle((winner-.49, i-.49),.98,.98,fc="none",ec="black"))
        ax.set(xticks=range(7), xticklabels=MODALITIES, yticks=range(len(targets)),
               yticklabels=[LABEL.get(t,t) for t in targets], title=ds)
    fig.colorbar(artist, ax=axes[:2], location="bottom", label="Mean test MAE / lowest mean test MAE", shrink=.65)
    axes[2].bar(("1","2","3"),[counts[j] for j in (1,2,3)], color=COLORS)
    axes[2].set(xlabel="Number of modalities", ylabel="Targets with minimum MAE")
    finish(fig, folder, "figure2")
    for number in (3, 4):
        fig, axes = plt.subplots(2,1,figsize=(9,7.5),layout="constrained")
        for ax, ds in zip(axes, ("qm9","qm8")):
            rows = read(folder/f"figure{number}_{ds}.csv")
            targets = list(dict.fromkeys(r["target"] for r in rows))
            x = np.arange(len(targets))
            if number==3:
                values=[float(r["improvement_percent"]) for r in rows]
                ax.bar(x,values,color=[COLORS[0] if y>=0 else COLORS[2] for y in values])
                ax.axhline(0,color=".35",lw=.8)
                ax.set_ylabel("Reduction in mean test MAE (%)")
            else:
                for j in (1,2,3):
                    part = [r for r in rows if int(r["active_modalities"])==j]
                    y, lo, hi = (np.array([float(r[c]) for r in part]) for c in ("mean","minimum","maximum"))
                    ax.bar(x+(j-2)*.25, y, .24, yerr=[y-lo,hi-y], capsize=2,
                           color=COLORS[j-1], label=f"{j} active")
                ax.legend(frameon=False, ncol=3)
                ax.set_ylabel("Fraction of molecule–channel positions")
            ax.set(xticks=x, xticklabels=[LABEL.get(t,t) for t in targets], title=ds.upper())
        finish(fig, folder, f"figure{number}")

    bars = group(read(folder/"figure5_bars_per_seed.csv"),"target","modality")
    curves = group(read(folder/"figure5_curves_per_seed.csv"),"target","modality","strategy","ratio")
    targets = ("heat_capacity_298K","electronic_spatial_extent")
    fig = plt.figure(figsize=(9,10),layout="constrained")
    top, bottom = fig.subfigures(2,1,height_ratios=[1,3])
    for ax, key, title in zip(top.subplots(1,3), ("active","magnitude","full_deletion_mae_ratio"),
                              ("Active fraction", "Magnitude share", "Full deletion / baseline MAE")):
        for j, mod in enumerate("SGP"):
            values=np.array([[float(r[key]) for r in bars[t,mod]] for t in targets])
            y=values.mean(axis=1)
            ax.bar(np.arange(2)+(j-1)*.23,y,.22,yerr=[y-values.min(1),values.max(1)-y],
                   capsize=2,color=COLORS[j],label=mod)
        ax.set(xticks=[0,1],xticklabels=[LABEL[t] for t in targets],title=title)
        ax.legend(ncol=3, frameon=False,fontsize=8)
    for axes, mod in zip(bottom.subplots(3,2), "SGP"):
        for ax,t in zip(axes,targets):
            for strategy, label, color in (("high_abs_contribution","High |qv|",COLORS[0]),("random","Random",COLORS[2])):
                ratios=(0.,.25,.5,1.)
                values=np.array([[float(r["normalized_delta"]) for r in curves[t,mod,strategy,str(x)]] for x in ratios])
                ax.plot(ratios,values.mean(1), marker="o",color=color,label=label)
                ax.fill_between(ratios,values.min(1),values.max(1),color=color,alpha=.18)
            ax.set(title=f"{LABEL[t]} · {mod}",xticks=ratios, xticklabels=["0","25","50","100"],
                   xlabel="Active channels removed (%)",ylabel="ΔMAE / ΔMAE(full)")
            ax.legend(frameon=False)
    finish(fig, folder, "figure5")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    main(parser.parse_args().data)
