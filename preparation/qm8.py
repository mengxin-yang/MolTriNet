"""Encode QM8 cleaned SMILES and DFT geometries."""
from collections import Counter
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from SmilesPE.pretokenizer import atomwise_tokenizer
from torch_geometric.data import Data
from qm9 import read_xlsx, sha256


def graphs_from_smiles(smiles_list, y):
    all_elements = []
    all_num_heavy_atom = []
    all_num_h = []
    all_formal_charges = []
    all_hybridizations = []
    all_chiralities = []
    all_atom_ring_sizes = []
    all_bond_ring_sizes = []

    all_bond_types = []
    all_bond_stereos = []

    invalid_smiles = []
    valid_smiles = []
    valid_labels = []


    def get_chirality_type(x):
        chiral_tag = x.GetChiralTag()
        return {
            Chem.ChiralType.CHI_UNSPECIFIED: 'CHI_UNSPECIFIED',
            Chem.ChiralType.CHI_TETRAHEDRAL_CW: 'CHI_TETRAHEDRAL_CW',
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 'CHI_TETRAHEDRAL_CCW',
            Chem.ChiralType.CHI_OTHER: 'CHI_OTHER'
        }.get(chiral_tag, 'CHI_UNSPECIFIED')


    def get_min_ring_sizes(mols):
        """
        返回：
        - min_atom_ring_sizes: 每个原子的最小环大小；不在环中为 0
        - min_bond_ring_sizes: 每条键的最小环大小；不在环中为 0
        """
        ring_info = mols.GetRingInfo()

        min_atom_ring_size = []
        for atom_idx in range(mols.GetNumAtoms()):
            if ring_info.NumAtomRings(atom_idx) > 0:
                min_atom_ring_size.append(ring_info.MinAtomRingSize(atom_idx))
            else:
                min_atom_ring_size.append(0)

        min_bond_ring_size = []
        for bond_idx in range(mols.GetNumBonds()):
            if ring_info.NumBondRings(bond_idx) > 0:
                min_bond_ring_size.append(ring_info.MinBondRingSize(bond_idx))
            else:
                min_bond_ring_size.append(0)

        return min_atom_ring_size, min_bond_ring_size


    for idx, (smiles, label) in enumerate(zip(smiles_list, y)):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            invalid_smiles.append({
                "data_index": idx,
                "excel_row": idx + 2,
                "smiles": smiles,
                "label": label,
                "reason": "RDKit MolFromSmiles/RemoveHs sanitize failed"
            })
            continue
        valid_smiles.append(smiles)
        valid_labels.append(label)

        min_atom_ring_sizes, min_bond_ring_sizes = get_min_ring_sizes(mol)

        for atom in mol.GetAtoms():
            all_elements.append(atom.GetSymbol())
            all_num_heavy_atom.append(atom.GetDegree())
            all_num_h.append(atom.GetTotalNumHs())
            all_formal_charges.append(atom.GetFormalCharge())
            all_hybridizations.append(str(atom.GetHybridization()))
            all_chiralities.append(get_chirality_type(atom))
            all_atom_ring_sizes.append(min_atom_ring_sizes[atom.GetIdx()])

        for bond in mol.GetBonds():
            all_bond_types.append(str(bond.GetBondType()))
            all_bond_stereos.append(str(bond.GetStereo()))
            all_bond_ring_sizes.append(min_bond_ring_sizes[bond.GetIdx()])



    element_list = sorted(set(all_elements))

    degree_counts = Counter(all_num_heavy_atom)
    max_degree = max(all_num_heavy_atom)
    num_heavy_atom_list = list(range(0, max_degree + 1))

    num_h_counts = Counter(all_num_h)
    max_num_h = max(all_num_h)
    num_h_list = list(range(0, max_num_h + 1))

    formal_charge_counts = Counter(all_formal_charges)
    formal_charge_list = sorted(set(all_formal_charges))

    hybridization_counts = Counter(all_hybridizations)
    hybridization_list = sorted(set(all_hybridizations))

    chirality_counts = Counter(all_chiralities)
    chirality_list = sorted(set(all_chiralities))

    atom_ring_size_counts = Counter(all_atom_ring_sizes)
    atom_ring_size_list = sorted(set(all_atom_ring_sizes + [0]))

    bond_ring_size_counts = Counter(all_bond_ring_sizes)
    bond_ring_size_list = sorted(set(all_bond_ring_sizes + [0]))

    bond_type_counts = Counter(all_bond_types)
    bond_type_list = sorted(set(all_bond_types))

    bond_stereo_counts = Counter(all_bond_stereos)
    bond_stereo_list = sorted(set(all_bond_stereos))

    atom_type_map = {element: idx for idx, element in enumerate(element_list)}
    num_heavy_atom_map = {degree: idx for idx, degree in enumerate(num_heavy_atom_list)}
    num_h_map = {num: idx for idx, num in enumerate(num_h_list)}
    formal_charge_map = {charge: idx for idx, charge in enumerate(formal_charge_list)}
    hybridization_map = {hybrid: idx for idx, hybrid in enumerate(hybridization_list)}
    chirality_map = {chiral: idx for idx, chiral in enumerate(chirality_list)}
    atom_ring_size_map = {size: idx for idx, size in enumerate(atom_ring_size_list)}
    bond_ring_size_map = {size: idx for idx, size in enumerate(bond_ring_size_list)}

    bond_stereo_map = {bs: idx for idx, bs in enumerate(bond_stereo_list)}
    bond_type_map = {bt: idx for idx, bt in enumerate(bond_type_list)}


    def get_one_hot(value, value_map, num_classes):
        idx = value_map.get(value, -1)
        if idx == -1:
            return [0] * num_classes

        one_hot = [0] * num_classes
        one_hot[idx] = 1
        return one_hot


    _node_dim = (
        len(element_list)
        + len(num_heavy_atom_list)
        + len(num_h_list)
        + len(formal_charge_list)
        + len(hybridization_list)
        + len(chirality_list)
        + 1
        + len(atom_ring_size_list)
        + 1
    )

    _edge_dim = (
        len(bond_type_list)
        + 1
        + len(bond_stereo_list)
        + len(bond_ring_size_list)
    )



    def smiles_to_graph(x):

        mols = Chem.MolFromSmiles(x)
        if mols is None:
            return None

        canonical_smiles = Chem.MolToSmiles(mols, isomericSmiles=True, canonical=True)
        mols = Chem.MolFromSmiles(canonical_smiles)

        min_ring_sizes, min_ring_sizes_bond = get_min_ring_sizes(mols)

        node_features = []
        for atom_idx, atom in enumerate(mols.GetAtoms()):

            atom_type = atom.GetSymbol()
            atom_type_one_hot = get_one_hot(atom_type, atom_type_map, len(element_list))

            num_heavy_atom = atom.GetDegree()
            num_heavy_atom_one_hot = get_one_hot(num_heavy_atom, num_heavy_atom_map, len(num_heavy_atom_list))

            num_h = atom.GetTotalNumHs()
            num_h_one_hot = get_one_hot(num_h, num_h_map, len(num_h_list))

            formal_charge = atom.GetFormalCharge()
            formal_charge_one_hot = get_one_hot(formal_charge, formal_charge_map, len(formal_charge_list))

            hybridization = str(atom.GetHybridization())
            hybridization_one_hot = get_one_hot(hybridization, hybridization_map, len(hybridization_list))

            chirality_type = get_chirality_type(atom)
            chirality_one_hot = get_one_hot(chirality_type, chirality_map, len(chirality_list))

            is_aromatic = [1] if atom.GetIsAromatic() else [0]

            atom_ring_size = min_ring_sizes[atom_idx]
            atom_ring_size_one_hot = get_one_hot(atom_ring_size, atom_ring_size_map, len(atom_ring_size_list))

            atomic_weight = atom.GetMass()
            scaled_weight = atomic_weight / 100.0

            atom_features = np.concatenate([
                atom_type_one_hot,
                num_heavy_atom_one_hot,
                num_h_one_hot,
                formal_charge_one_hot,
                hybridization_one_hot,
                chirality_one_hot,
                is_aromatic,
                atom_ring_size_one_hot,
                [scaled_weight]
            ])

            node_features.append(atom_features)

        edges_index = []
        edge_features = []

        for bond in mols.GetBonds():
            i = bond.GetBeginAtomIdx()  # the initial atom
            j = bond.GetEndAtomIdx()  # the initial atom
            edges_index.append((i, j))
            edges_index.append((j, i))

            bond_type = str(bond.GetBondType())
            bond_type_one_hot = get_one_hot(bond_type, bond_type_map, len(bond_type_list))

            is_conjugated = [1] if bond.GetIsConjugated() else [0]

            stereo_str = str(bond.GetStereo())
            stereo_one_hot = get_one_hot(stereo_str, bond_stereo_map, len(bond_stereo_list))

            bond_idx = bond.GetIdx()
            bond_ring_size = min_ring_sizes_bond[bond_idx]

            bond_ring_size_one_hot = get_one_hot(bond_ring_size, bond_ring_size_map, len(bond_ring_size_list))

            bond_features = np.concatenate([
                bond_type_one_hot,
                is_conjugated,
                stereo_one_hot,
                bond_ring_size_one_hot
            ])

            edge_features.append(bond_features)
            edge_features.append(bond_features)

        node_feature = torch.tensor(np.array(node_features), dtype=torch.float)

        if len(edges_index) == 0:
            edges_index = torch.empty((2, 0), dtype=torch.long)
            edge_feature = torch.empty((0, _edge_dim), dtype=torch.float)
        else:
            edges_index = torch.tensor(edges_index, dtype=torch.long).t().contiguous()
            edge_feature = torch.tensor(np.array(edge_features), dtype=torch.float)

        return Data(x=node_feature, edge_index=edges_index,edge_attr=edge_feature)


    graphs = [smiles_to_graph(smi) for smi in valid_smiles]
    node_dim, edge_dim = _node_dim, _edge_dim
    if invalid_smiles or len(graphs) != len(smiles_list):
        raise ValueError("Invalid SMILES or dropped molecules")
    return graphs


def tokens_from_smiles(smiles_list):
    smiles_list = [i.replace('\xa0', '').replace(' ', '') for i in smiles_list]


    all_atom_tokens = []          # List of tokens for each SMILES
    parsed_smiles = []            # Flattened list of all tokens

    for smi in smiles_list:
        tokens = atomwise_tokenizer(smi)
        all_atom_tokens.append(tokens)
        parsed_smiles.extend(tokens)

    # Build token_to_idx
    token_to_idx = {'PAD': 0}
    unique_tokens = sorted(set(parsed_smiles))
    token_to_idx.update({token: idx + 1 for idx, token in enumerate(unique_tokens)})

    vocb_size = len(token_to_idx)
    max_length = max(len(tokens) for tokens in all_atom_tokens)


    # 4. Convert SMILES to index sequences (padding to max_length)
    def smiles_to_index(token_list, vocab, length):
        smiles_idx = np.zeros(length, dtype=np.int32)
        # tokens is a list of tokens for a SMILES
        for i, token in enumerate(token_list):
            if i >= length:
                break
            smiles_idx[i] = vocab[token]
        return smiles_idx


    indices = np.array([smiles_to_index(tokens, token_to_idx, max_length) for tokens in all_atom_tokens])

    return indices


def fortran_float(value):
    """Convert Fortran-style floats, such as 2.1997*^-6, to Python floats."""
    return re.sub(r"(\d+\.?\d*)\*\^([+-]?\d+)", r"\1e\2", value)


def parse_qm8_xyz_and_center(file_path):
    """Read one QM8 xyz file and return centered coordinates in xyz atom order."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"Invalid xyz file with fewer than 3 lines: {file_path}")

    n_atom = int(lines[0].strip())
    elements = []
    raw_coords = []

    atom_lines = lines[2: 2 + n_atom]
    if len(atom_lines) != n_atom:
        raise ValueError(f"Expected {n_atom} atom lines but found {len(atom_lines)} in {file_path}")

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line in {file_path}: {line.rstrip()}")
        elements.append(parts[0])
        raw_coords.append([
            float(fortran_float(parts[1])),
            float(fortran_float(parts[2])),
            float(fortran_float(parts[3])),
        ])

    raw_coords = np.asarray(raw_coords, dtype=np.float32)
    centroid = np.mean(raw_coords, axis=0, keepdims=True)
    centered_coords = raw_coords - centroid
    return centered_coords, elements


def get_qm8_xyz_path(xyz_dir, row_idx_1based):
    """Return xyz file path for the QM8 valid molecule index."""
    filename = f"qm8_valid_{row_idx_1based:05d}.xyz"
    filepath = os.path.join(xyz_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing xyz file: {filepath}")
    return filepath


def points_from_xyz(XYZ_DIR, total_files):
    all_elements = []
    for idx in range(1, total_files + 1):
        filepath = get_qm8_xyz_path(XYZ_DIR, idx)
        _, xyz_elements = parse_qm8_xyz_and_center(filepath)
        all_elements.extend(xyz_elements)


    element_list = sorted(set(all_elements))
    atom_type_to_idx = {atom: idx for idx, atom in enumerate(element_list)}
    print(f"element_list: {element_list}")

    point_clouds = []
    point_cloud_lengths = []
    qm8_xyz_files = []

    for idx in range(1, total_files + 1):
        filepath = get_qm8_xyz_path(XYZ_DIR, idx)
        coords, xyz_elements = parse_qm8_xyz_and_center(filepath)

        n_atoms = len(xyz_elements)

        atom_type_onehot = np.zeros((n_atoms, len(element_list)), dtype=np.float32)
        for atom_idx, symbol in enumerate(xyz_elements):
            type_idx = atom_type_to_idx.get(symbol)
            if type_idx is None:
                raise ValueError(f"Unknown atom type {symbol} in {filepath}")
            atom_type_onehot[atom_idx, type_idx] = 1.0

        # Since coordinates are already centered, rho_i = ||r_i^c||_2.
        rho = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)

        # Same feature logic as QM9: [centered_x, centered_y, centered_z, rho, atom_type_onehot].
        point_cloud_feature = np.concatenate(
            [coords.astype(np.float32), rho, atom_type_onehot],
            axis=1,
        )

        point_clouds.append(point_cloud_feature)
        point_cloud_lengths.append(n_atoms)
        qm8_xyz_files.append(os.path.basename(filepath))

        if idx % 5000 == 0:
            print(f"Processed {idx} files...")

    if len(point_clouds) != total_files:
        raise ValueError(
            f"Number of processed point clouds ({len(point_clouds)}) does not match label rows ({total_files})."
        )

    max_atoms = max(point_cloud_lengths)
    d_input = point_clouds[0].shape[1]

    print(f"d_input = {d_input}, type: {type(d_input)}")
    print(f"Maximum atoms in dataset: {max_atoms}")
    print(f"Feature dimension per atom: {d_input}")
    print(f"Total molecules processed: {len(point_clouds)}")

    padded_point_clouds = []
    for pc, length in zip(point_clouds, point_cloud_lengths):
        if length < max_atoms:
            pad_size = max_atoms - length
            padding = np.zeros((pad_size, pc.shape[1]), dtype=np.float32)
            padded_pc = np.vstack([pc, padding])
        else:
            padded_pc = pc
        padded_point_clouds.append(padded_pc)

    point_cloud = [torch.tensor(pc, dtype=torch.float32) for pc in padded_point_clouds]
    point_cloud_lengths = torch.tensor(point_cloud_lengths, dtype=torch.long)
    qm8_element_list = element_list
    return point_cloud, point_cloud_lengths


def build(table, xyz, output):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/paper.json').read_text())['datasets']['qm8']
    rows = iter(read_xlsx(table))
    names = next(rows)
    if names != ['SMILES', *config['target_names']]:
        raise ValueError('QM8 target columns do not match the reference table')
    records = list(rows)
    smiles = [r[0] for r in records]
    targets = np.asarray([r[1:] for r in records], dtype=np.float64)
    ids = []
    for i, smi in enumerate(smiles, 1):
        with (xyz / f'qm8_valid_{i:05d}.xyz').open(encoding='utf-8') as f:
            next(f)
            header = next(f).strip()
        original, index, descriptor = header.split('; ', 2)
        if index != f'xyz_index={i}' or descriptor != f'smiles={smi}' or not original.startswith('original_sdf_index='):
            raise ValueError(f'XYZ molecule identity mismatch at cleaned row {i}')
        ids.append(int(original.split('=')[1]) + 1)
    graphs = graphs_from_smiles(smiles, targets[:, -2])
    sequences = tokens_from_smiles(smiles)
    points, lengths = points_from_xyz(xyz, len(records))
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'cache.npz'
    np.savez_compressed(path, ids=np.asarray(ids, dtype=np.int64), targets=targets,
                        graph_x=np.concatenate([g.x.numpy() for g in graphs]),
                        edge_index=np.concatenate([g.edge_index.numpy() for g in graphs],axis=1).astype(np.int16),
                        edge_attr=np.concatenate([g.edge_attr.numpy() for g in graphs]),
                        node_offsets=np.r_[0,np.cumsum([len(g.x) for g in graphs])].astype(np.int64),
                        edge_offsets=np.r_[0,np.cumsum([len(g.edge_attr) for g in graphs])].astype(np.int64),
                        smiles_ids=sequences.astype(np.int16), points=np.stack([v.numpy() for v in points]),
                        point_lengths=lengths.numpy().astype(np.int16))
    digest = sha256(path)
    if digest != config['cache_sha256']:
        raise ValueError('Prepared inputs differ from the paper cache; check the original table, DFT geometries and feature library versions.')
    (output / 'metadata.json').write_text(json.dumps(dict(dataset='qm8', cache_sha256=digest,
        dims=config['dims'], target_names=config['target_names'], target_units=config['targets']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', type=Path, required=True)
    parser.add_argument('--xyz-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.table, args.xyz_dir, args.output)
