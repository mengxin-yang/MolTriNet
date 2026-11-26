# MolTriNet

MolTriNet predicts quantum-chemical properties from three representations of a molecule. A Transformer–CNN encodes SMILES (S), a directed message-passing network encodes the molecular graph (G), and a point-cloud network encodes atoms and their 3D coordinates (P). One model is trained for each target. The experiments cover eight QM9 properties and twelve QM8 properties.

[*More Is Not Always Better: Selective Multimodal Fusion for Quantum-Chemical Property Prediction*](CITATION.cff) asks when a third representation helps and which retained features contribute to a trained predictor. Across the 20 targets, the lowest mean test MAE came from a single modality for two targets, a pair for fourteen, and the full model for four.

## Run a small example

Clone the repository and run the commands below from its root with Python 3.11. The example includes 32 QM8 molecules from the saved seed-42 split. Sixteen train the model, eight select the checkpoint and eight belong to the test partition. The commands evaluate the saved checkpoint on the eight validation molecules.

```bash
git clone https://github.com/mengxin-yang/MolTriNet.git
cd MolTriNet
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the CPU dependencies and run:

```bash
python -m pip install -r requirements-cpu.txt
python -m examples.check_model
python -m moltrinet fit --dataset qm8 --target E1-CC2 --seed 42 --gate entmax15 --modalities SGP --cache examples/qm8 --split examples/split.json --epochs 1 --run runs/example
python -m moltrinet evaluate --run runs/example --cache examples/qm8 --split examples/split.json --phase val
python -m analysis.validation_gates --run runs/example --cache examples/qm8 --split examples/split.json --out runs/example-gates --delete
```

The example saves a checkpoint, reloads it, reports validation MAE and RMSE in hartree, and writes per-molecule gate weights and fixed-gate deletion errors. A one-epoch run checks the data and inference path. Use a fresh output directory for another run.

## Data and model choices

Download the encoded inputs and reference labels for [QM8 (21,747 molecules)](https://github.com/mengxin-yang/MolTriNet/releases/download/data-v1/qm8-cache.zip) or [QM9 (133,885 molecules)](https://github.com/mengxin-yang/MolTriNet/releases/download/data-v1/qm9-cache.zip). Extract each archive into its own directory. The training entry point reads `cache.npz` and `metadata.json` from that directory, uses the saved split for seed 42, 44 or 46, and fits the label scaler on training molecules. The QM9 U₀ task uses atomization energy in kcal/mol. [Data release](https://github.com/mengxin-yang/MolTriNet/releases/tag/data-v1) · [Checksums, sources and preparation](docs/TRAINING.md#full-inputs-and-split-identity)

| Model | How a channel's three modal weights are obtained |
|---|---|
| MolTriNet (softmax) | The SI dense gate jointly scores the normalized embeddings and applies softmax across the enabled modalities. |
| MolTriNet (1.5-entmax) | Three independent networks score their own normalized embeddings. For each molecule and each of 256 channels, 1.5-entmax normalizes the scores across modalities and can assign an exact zero. |

Figure 3 compares these two reported configurations. The dense per-seed metric records identify the reported errors. The available records do not bind the historical dense scorer to an immutable trained checkpoint. Separate independent-scorer softmax controls on two QM9 targets are excluded from this comparison. See [the results and provenance](docs/RESULTS.md).

## Reproduce the paper figures and tables

```bash
python analysis/paper_results.py --out build/paper
python analysis/plot_results.py --data build/paper
python supplementary/figures.py --out build/si
```

| Paper item | Configuration and inputs | What the commands produce |
|---|---|---|
| Tables 1–2 and S5 | Held-out test metrics, three seeds per target, plus the published comparison values | Tables as CSV under `build/paper/` |
| Figure 2 | Seven separately trained subsets (S, G, P, SG, SP, GP, SGP), ranked by unrounded mean test MAE | Per-target normalized MAEs and the modality-comparison plot |
| Figure 3 | Dense and sparse SGP configurations, same target and split | Percent change in mean test MAE per target and the comparison plot |
| Figures 4–5 | Sparse validation weights and fixed-gate deletion for Cᵥ and electronic spatial extent ⟨R²⟩ | Active-modality fractions, deletion curves and the two figures |
| Figures S4–S7 and Table S6 | QM9 target correlations, matched geometry cohorts, and optional SMILES–graph contrastive regularization | Supplementary plots and geometry comparisons under `build/si/` |

The plotting commands read the included results and require no GPU. To compute validation gates and deletions from a trained model, use the example above with the relevant full cache, run and split. Deletion subtracts selected weighted features before the output LayerNorm with the original gate fixed. The three encoders still execute. [Main figure inputs and output filenames](docs/RESULTS.md) · [Supplementary experiment commands](docs/SUPPLEMENT.md)

## Citation and data sources

See [CITATION.cff](CITATION.cff) for the authors and paper title. Please also cite the dataset used in your analysis. [Data sources and third-party acknowledgments](THIRD_PARTY_NOTICES.md)
