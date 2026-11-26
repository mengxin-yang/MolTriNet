# Train and evaluate MolTriNet

Run commands from the repository root with Python 3.11. The CPU environment uses the native PyTorch Scatter extension. For GPU training, install the PyTorch 2.7.1 CUDA build and the matching PyG Scatter wheel for the same CUDA ABI, followed by torch-geometric 2.6.1, entmax 1.3 and NumPy 2.1.2. The archived GPU runs used PyTorch 2.7.1 with CUDA 12.8.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m examples.check_model
```

In Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1`. The structural check exercises all seven modality subsets and verifies per-channel normalization, strict entmax zeros and subtraction of the fixed SMILES contribution.

## Small example

The example contains 32 real QM8 molecules with their cached features and dataset IDs. It takes the first 16, 8 and 8 IDs from the training, validation and test partitions of the seed-42 split. A single CPU epoch exercises model fitting, checkpoint selection, reloading and evaluation. Its errors describe this example only.

```bash
python -m moltrinet fit --dataset qm8 --target E1-CC2 --seed 42 --gate entmax15 --modalities SGP --cache examples/qm8 --split examples/split.json --epochs 1 --run runs/example
python -m moltrinet evaluate --run runs/example --cache examples/qm8 --split examples/split.json --phase val
```

## Full inputs and split identity

The repository contains the 32-molecule example, the six full ordered molecule-ID splits and the feature-generation scripts. Full training requires the dataset-specific `cache.npz` and `metadata.json`, or the corresponding input table and DFT coordinate files to generate them. The code checks the cache and split digests in `configs/paper.json`. The full processed caches and the exact source workbooks are external inputs. The example and paper-result plots run from the repository alone.

Each random split has 17,397/2,174/2,176 training/validation/test IDs for QM8 or 107,108/13,388/13,389 for QM9. Each paper run uses the complete cache and the corresponding saved partition. Supply an explicit `--split` for the 32-molecule example.

QM8 uses 21,747 molecules identified by their original SDF record number, cleaned canonical SMILES and row-aligned DFT coordinates. Its 12 properties comprise two excitation energies (hartree) and two dimensionless oscillator strengths at each of the CC2, PBE0 and CAM levels. Coordinate identity matters for 36 structures that differ between QM8 sources. [QM8 source](https://github.com/raghurama123/ExcitedStatesQM8) · [QM8 notice](../preparation/notices/QM8-MIT.txt)

QM9 uses all 133,885 stable GDB molecule IDs. The property names, units and column order appear in [paper.json](../configs/paper.json). For the `U0` row, select `internal_energy_0K_atom_kcal_mol`, the 0 K atom-referenced energy with the sign of molecular energy minus isolated-atom energies. The data and isolated-atom reference values are available from the original authors. [QM9 archive](https://doi.org/10.6084/m9.figshare.978904_D12) · [Atomic references](https://doi.org/10.6084/m9.figshare.978904_D9)

For a feature-generation audit using the corresponding tabulated labels, SMILES and raw DFT geometries, install the optional dependencies with `python -m pip install -r preparation/requirements.txt`. Raw-input paths below are supplied by the reader. The raw input workbook SHA256 values are in `configs/paper.json`. The QM8 extractor checks cleaned row IDs against each XYZ header. The QM9 extractor checks the GDB ID, SMILES, properties and atom counts against the XYZ file. Both encode atomwise SMILES tokens, directed molecular graphs and centered 3D coordinates with radial distance and element indicators. Dataset-level feature vocabularies precede splitting. Training-label scaling is fitted per split.

```bash
python preparation/qm8.py --table QM8_clean.xlsx --xyz-dir qm8_xyz --output data/qm8
python preparation/qm9.py --table QM9.xlsx --xyz-dir qm9_xyz --output data/qm9
```

## Paper runs

```bash
python -m moltrinet fit --dataset qm9 --target heat_capacity_298K --seed 42 --gate entmax15 --cache data/qm9 --device cuda --run runs/qm9_cv_42_entmax
python -m moltrinet evaluate --run runs/qm9_cv_42_entmax --cache data/qm9 --device cuda --phase test
python -m moltrinet fit --dataset qm8 --target E1-CC2 --seed 42 --gate softmax --modalities GP --cache data/qm8 --device cuda --run runs/qm8_E1_CC2_42_GP
```

Repeat with each target listed in `configs/paper.json` and seeds 42, 44 and 46. Use a separate output directory per fit. To run the supplementary SMILES–graph contrastive regularizer, add `--cmcr` to a three-branch entmax fit. Every run writes the effective configuration, training/validation history and `best.pt` into its output directory. Evaluation writes `val_metrics.json` or `test_metrics.json` with MAE and RMSE in the original physical units. Select the model on validation data before evaluating the held-out test split.

| Option | Fusion and encoder paths |
| --- | --- |
| `--gate softmax --modalities SGP` | One shared MLP reads the concatenated normalized S/G/P embeddings and produces channel-specific scores, following SI Equations S24–S25. |
| `--gate softmax --modalities SG` (also SP, GP) | The shared gate uses the two available encoders and masks out the absent modality before softmax. |
| `--gate softmax --modalities S` (also G, P) | A regression head reads a single encoder, bypassing fusion. |
| `--gate entmax15 --modalities SGP` | Three independent scoring networks read their respective normalized features. Fixed-alpha entmax1.5 normalizes each channel over S/G/P. The head reads the weighted feature sum after LayerNorm. |
| `--gate softmax --dense-context pairwise` | Available implementation of a joint gate with projected pairwise Hadamard products. |

The archived dense result tables identify their predictions by target, seed and modality subset. An immutable gate-selector configuration tied to each dense run was unavailable. The dense default implements the equations in the SI. The available pairwise implementation is exposed explicitly for readers studying that architectural distinction. The entmax results have per-seed checkpoint identities. The two scoring architectures differ in both information access and normalization.

Each fit predicts one property in FP32. The label mean and population standard deviation come solely from the training partition. The objective is L1 loss on standardized labels. Batch size is 128, with AdamW (learning rate 0.0002, weight decay 0.0001, betas 0.9/0.999) and gradient-norm clipping at 0.5. The learning-rate scheduler halves the rate after a validation-loss plateau with patience 8. Validation loss is the unweighted mean of per-batch standardized MAEs in the training code. Early stopping uses patience 10 and a minimum improvement of 0.0001. The checkpoint is the strict minimum validation loss seen, including improvements below that early-stopping threshold. The configured ceiling is 200 epochs. Two dense dipole-moment logs (seeds 42 and 44) end at epoch 100. The entmax dipole recipe uses the 200-epoch ceiling for all three seeds.

The saved split fixes sample identity and order. Hardware kernels, software builds, nondeterministic GPU reductions and early stopping can affect a new fit. Use the released per-seed tables for the numerical values reported in the paper.
