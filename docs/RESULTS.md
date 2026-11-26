# Reproducing the main tables and figures

Run these commands at the repository root. The first command uses the Python standard library. The second requires NumPy and Matplotlib (tested with Matplotlib 3.8).

```bash
python analysis/paper_results.py --out build/paper
python analysis/plot_results.py --data build/paper
```

| Paper item | Input under `results/` | Output under `build/paper/` |
|---|---|---|
| Tables 1 and 2 | `fusion_test_metrics.csv`, `literature_qm9.csv`, `literature_qm8.csv` | `table_1.csv`, `table_2.csv` |
| Table S5 | Same inputs | `table_s5_qm9.csv`, `table_s5_qm8.csv` |
| Figure 2 | `modality_test_metrics.csv` | `figure2_qm9.csv`, `figure2_qm8.csv`, `figure2.pdf` |
| Figure 3 | `fusion_test_metrics.csv` | `figure3_qm9.csv`, `figure3_qm8.csv`, `figure3.pdf` |
| Figure 4 | `support_validation.csv` | `figure4_qm9.csv`, `figure4_qm8.csv`, `figure4.pdf` |
| Figure 5 | `gate_validation.csv`, `deletion_validation.csv` | `figure5_bars_per_seed.csv`, `figure5_curves_per_seed.csv`, `figure5.pdf` |

**Prediction.** Each row in `modality_test_metrics.csv` is the held-out test result for one target, modality subset and seed. S denotes SMILES, G the molecular graph and P the 3D atomic point cloud. The seven subsets for a given seed use the same split. The script ranks the unrounded test MAEs averaged over seeds 42, 44 and 46 and divides each mean by its target's lowest mean. The numbers of best single-, dual- and triple-modality configurations are 0/5/3 on QM9 and 2/9/1 on QM8. U₀ refers to the atomization energy at 0 K in kcal/mol.

`fusion_test_metrics.csv` pairs the three-modality softmax test metrics from the per-seed SGP workbooks with the independent-score 1.5-entmax test metrics. The SHA-256 values identify the dense metric workbook and the sparse best checkpoint. The entmax checkpoint identity, split digest and best epoch for each run appear in `entmax_checkpoints.csv`. For Cᵥ seed 42 the selected checkpoint is from a repeat run, epoch 54. For μ, the dense runs for seeds 42 and 44 end at epoch 100, while the sparse recipe permits up to 200 epochs. Dense-workbook metadata identify metrics, splits and epochs. They do not retain an immutable scorer source or checkpoint digest. The separate softmax runs with independent modality scorers on gap and Cᵥ are excluded from the paper-result tables.

Figure 3 uses `100 * (1 - mean_entmax / mean_softmax)` for each property. It compares the reported model configurations. Mean MAE decreases for 7/8 QM9 targets and 11/12 QM8 targets. QM9 α and QM8 E₁-PBE0 have higher mean MAE with 1.5-entmax. The literature comparison columns contain published results. GC, KRR, MPNN and DTNN are from Wu *et al.*, *MoleculeNet*, **Chem. Sci.** 2018, 9, 513–530, SI Tables S8–S9. The QM9 AttentiveFP and PointGAT values are from Zhang, Yuan and Tian, *PointGAT*, **J. Chem. Theory Comput.** 2024, 20, 4115–4128, Table 4 and Table S4.

**Gate activity.** Each checkpoint was evaluated on its validation partition (13,388 QM9 or 2,174 QM8 molecules). Each molecule has 256 feature channels. `p1`, `p2` and `p3` are the fractions of molecule–channel positions with exactly one, two or three strictly positive modal weights. Figure 4 averages these fractions over the three seeds and shows the minimum and maximum. An individual modality's active fraction uses the same molecule–channel denominator. Its magnitude share is first computed within each molecule as the sum of absolute weighted features for that modality divided by the sum across all modalities and channels, then averaged over molecules with a positive denominator.

**Fixed-gate deletion.** Figure 5 uses the validation sets for Cᵥ and electronic spatial extent ⟨R²⟩. With a branch's weights and normalized features fixed, its selected weighted contributions are subtracted from the original fusion sum before the output LayerNorm and prediction head. At nominal fraction `r`, exactly `ceil(r * n_active)` positive-weight channels are selected per molecule. Magnitude ranking uses descending absolute weighted features with ties resolved by ascending channel index. Random deletion uses nested prefixes of a molecule-specific permutation for each of 20 repeat seeds. The CSV retains each repetition's MAE and measured deletion fraction. The figure first averages random MAEs within model seed. Within each seed and branch it scales the MAE difference from baseline by the difference at full deletion, then shows the mean and range across the three model seeds. These are inference interventions on a trained model. Figure 2 measures separately trained modality subsets.
