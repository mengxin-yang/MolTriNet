# Supplementary experiments

Run the examples from the repository root with Python 3.11. The stored measurements are sufficient to redraw four figures and recover the prediction summaries without training.

```sh
python supplementary/figures.py --out results/si
```

This command uses NumPy and Matplotlib. It writes PNG and PDF versions of Figures S4–S7 and `supplement_summary.csv` in original target units. Use `--qm9-cache DATA/qm9` to recompute all 225 Pearson coefficients for Figure S4 from the full 133,885-row QM9 target array. Figure S4 uses the reported internal energy at 0 K. The regression target called U₀ uses the atomization-related energy column in kcal/mol.

| Paper item | Measurements in `supplementary/data/` | Summary |
| --- | --- | --- |
| Figure S4 | `qm9_pearson.csv` | Pearson correlations across 15 QM9 properties |
| Table S6 and Figure S5 | `geometry_per_seed.csv` | MAE and RMSE for four coordinate conditions, 20 properties and three training seeds. The figure plots 100 × (mean D / mean A − 1) for each metric. |
| Figure S6 | `cmcr_alignment_per_seed.csv` | SMILES/graph cosine gap and full-test-pool retrieval for two spaces within each CMCR-trained model |
| Figure S7 | `cmcr_prediction_per_seed.csv` | Paired CMCR-off/on test errors from validation-selected checkpoints for each property and seed |

Table S6 reports means over training seeds 42, 44 and 46. Multiply the CSV values by 1,000 for the displayed QM9 HOMO, gap and ZPVE entries (milli-hartree), and for all QM8 entries (milli-hartree for excitation energies, dimensionless × 1,000 for oscillator strengths).

## Coordinate sensitivity

| Configuration | Training coordinates | Test coordinates | Checkpoint |
| --- | --- | --- | --- |
| A | DFT | DFT | DFT-trained |
| B | DFT | ETKDGv3 | Same as A |
| C | DFT | MMFF94 | Same as A |
| D | MMFF94 | MMFF94 | Independently trained on MMFF94 |

All conditions use the same independent 1.5-entmax fusion, three modalities, target labels and eligible molecule IDs. The comparison has its own split and cache schema. QM9 retains 105,992/13,251/13,243 train/validation/test molecules from 133,885 by filtering a fixed random split. QM8 retains 17,356/2,164/2,169 from its 21,747-molecule starting cohort by filtering a fixed structure-grouped split. The three model seeds share these partitions. The main QM8 tables use seed-specific random splits and a different SMILES/graph preprocessing protocol.

**Source inputs.** QM9 uses the table and ID-matched original XYZ files described under data preparation. QM8 uses the full 21,786-row raw table with explicit-H SMILES and twelve label columns, and the authors’ `XYZ_B3LYP_631G2dfp.xyz` file. The supplied IDs select the 21,747 original rows. Group membership, tokenizer vocabulary and SHA-256 checks for each numerical input array are provided in `geometry/generation/qm8_reference.json`. Both original table order and raw SMILES spelling matter. Large source files, generated coordinates and model weights are external inputs or local outputs. [QM8 author data](https://github.com/raghurama123/ExcitedStatesQM8/tree/main/22k_electronic_spectra_TDDFT_CC2) · [QM9 data](https://doi.org/10.6084/m9.figshare.978904)

```sh
python preparation/qm9.py --table DATA/QM9.xlsx --xyz-dir DATA/qm9_xyz --output DATA/qm9_source
python supplementary/geometry/generation/qm8_source.py --table DATA/QM8.xlsx --xyz DATA/XYZ_B3LYP_631G2dfp.xyz --out DATA/qm8_source
python supplementary/geometry/generation/prepare.py --dataset QM9 --cache DATA/qm9_source --xyz DATA/qm9_xyz --out DATA/geometry_qm9
python supplementary/geometry/generation/prepare.py --dataset QM8 --cache DATA/qm8_source --xyz DATA/XYZ_B3LYP_631G2dfp.xyz --out DATA/geometry_qm8
```

For coordinate generation the authors used Python 3.11.4, RDKit 2026.03.3 and NumPy 2.1.2. Preparing SMILES features also requires SmilesPE. Each molecule receives one seeded ETKDGv3 embedding with one random-coordinate retry on failure. MMFF94 refines a copy of that conformer for up to 300 iterations, with another 700 on nonconvergence. The source records, generation outcomes and masks remain in the generation output. The program checks the resulting cohort against the deposited split in `supplementary/geometry/splits/` before writing training caches. Point clouds include explicit hydrogens, are centered in Å, and retain element channels in C/F/H/N/O order. DFT and generated structures are compared at molecule level without atom-index correspondence. Labels remain those of the reference calculation.

**Training and evaluation.** The following example trains the two coordinate conditions for QM8 E1-CC2 and evaluates all four. Use the property keys in `supplementary/geometry/properties.json` for the other targets and provide the corresponding generated data directory. Use an empty work directory on the first invocation. Training and final evaluation use CUDA on one visible device.

```sh
python supplementary/geometry/run.py prepare --experiment qm8_e1cc2 --data DATA/geometry_qm8/data --work runs/geometry_e1
python supplementary/geometry/run.py train --work runs/geometry_e1
python supplementary/geometry/evaluate.py --work runs/geometry_e1
```

The geometry trainers use train-only label scaling, FP32, batch size 128, AdamW at 0.0002, weight decay 0.0001, gradient clipping at 0.5, a 200-epoch maximum, a 50-epoch minimum and early-stopping patience 30. Each fit has a 420-minute cumulative limit. A rerun can resume its own last checkpoint. Final evaluation checks that all six fits for that property are complete, that DFT/MMFF labels, splits and paired initializations agree, and that A, B and C use the same best-checkpoint digest. Per-condition metrics and prediction files are written under `runs/geometry_e1/assessment/`. The repository includes the checksums and metrics from the paper. A new training run can produce different weights due to software or device differences.

## SMILES–graph contrastive regularization

The auxiliary loss uses separate 256→256→64 projection heads with ReLU and dropout 0.1. The SMILES and graph projections are L2-normalized. Symmetric in-batch InfoNCE pairs representations of the same molecule and uses the other records in the batch as negatives. The regression model minimizes standardized MAE + 0.03 × InfoNCE at temperature 0.2. Supervised validation MAE selects the checkpoint. The point-cloud branch contributes to regression only.

```sh
python -m moltrinet fit --dataset qm8 --target E1-CC2 --seed 42 --gate entmax15 --modalities SGP --cache DATA/qm8_main --run runs/e1_off --device cuda
python -m moltrinet fit --dataset qm8 --target E1-CC2 --seed 42 --gate entmax15 --modalities SGP --cache DATA/qm8_main --run runs/e1_on --device cuda --cmcr
python -m moltrinet evaluate --phase test --run runs/e1_off --cache DATA/qm8_main --device cuda
python -m moltrinet evaluate --phase test --run runs/e1_on --cache DATA/qm8_main --device cuda
python -m supplementary.alignment --run runs/e1_on --cache DATA/qm8_main --split splits/qm8_seed_42.json --device cuda --out results/e1_alignment.csv
```

The alignment diagnostic evaluates the 256-dimensional encoder outputs and the 64-dimensional projection outputs from the same CMCR-on checkpoint. For negative cosine, it samples one mismatched partner per query with a seed-specific permutation and replaces fixed points. Bidirectional top-1 is the arithmetic mean of SMILES→graph and graph→SMILES nearest-partner accuracy over all held-out molecules (13,389 QM9 or 2,176 QM8). Results are averaged first over three seeds per target, then equally over targets. The reported projection-space bidirectional accuracies round to 63.2% (QM9) and 64.3% (QM8). CMCR reduces mean test MAE on 6 of 20 properties and on 24 of 60 seed pairs. These summaries measure different effects of the auxiliary objective.
