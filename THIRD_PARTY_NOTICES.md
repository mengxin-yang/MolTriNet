# Data sources and third-party acknowledgments

**QM9.** Ramakrishnan, Dral, Rupp and von Lilienfeld, *Quantum chemistry structures and properties of 134 kilo molecules*, Scientific Data 1, 140022 (2014). The featurized QM9 release uses the molecular records and the atomic reference values published on Figshare under CC0 1.0. [Molecular data](https://doi.org/10.6084/m9.figshare.978904_D12) · [Atomic reference values](https://doi.org/10.6084/m9.figshare.978904_D9) · [CC0](https://creativecommons.org/publicdomain/zero/1.0/)

**QM8.** Ramakrishnan, Hartmann, Tapavicza and von Lilienfeld, *Electronic spectra from TDDFT and machine learning in chemical space*, J. Chem. Phys. 143, 084111 (2015). The QM8 input distribution includes the source repository's copyright and MIT license notice. [Source repository](https://github.com/raghurama123/ExcitedStatesQM8) · [License text](preparation/notices/QM8-MIT.txt)

**Sparse normalization.** The model calls the `entmax15` operator distributed in the `entmax` Python package. The method is described by Peters, Niculae and Martins in *Sparse Sequence-to-Sequence Models* (ACL, 2019). [Method](https://aclanthology.org/P19-1146/) · [Implementation](https://github.com/deep-spin/entmax)

The encoders use Transformer attention, directed message passing and PointNet-style pooling. Training, feature preparation and plotting use separately installed libraries, including PyTorch, PyTorch Geometric, torch-scatter, RDKit, SmilesPE, NumPy and Matplotlib. Consult the installed distributions for their licenses. The data notices apply to their respective sources. Citations for the comparison values appear in [the results documentation](docs/RESULTS.md).
