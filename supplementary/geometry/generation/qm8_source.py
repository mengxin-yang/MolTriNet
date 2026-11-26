"""Build and verify the starting QM8 geometry cache using the full raw SMILES/label table and author XYZ."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer

from feature_reference import canonical_mol
from source_checks import graph_features, point_features, qm8_selected_xyz, sha256

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "preparation"))
from qm9 import read_xlsx


def build(table, xyz, out):
    ref = json.loads((HERE / "qm8_reference.json").read_text())
    if sha256(xyz) != ref["author_xyz_sha256"]:
        raise ValueError("QM8 XYZ digest differs from the author data used for geometry evaluation")
    split = json.loads((HERE.parent / "splits/qm8_original.json").read_text())
    ids = np.array(sorted(mid for members in split.values() for mid in members), dtype=np.int64)
    meta = json.loads((HERE.parent / "templates/qm8/dft_metadata.json").read_text())
    if table.suffix.lower() == ".xlsx":
        header, *rows = read_xlsx(table)
    else:
        with table.open(encoding="utf-8-sig", newline="") as stream:
            header, *rows = csv.reader(stream)
    if len(rows) != 21786:
        raise ValueError("The raw table must contain 21786 rows in source order")
    smi_col = next(i for i, col in enumerate(header) if col.lower() == "smiles")
    columns = [header.index(target) for target in meta["target_names"]]
    selected = [rows[mid - 1] for mid in ids]
    targets = np.array([[float(row[col]) for col in columns] for row in selected], dtype=np.float64)
    smiles = [r[smi_col] for r in selected]
    groups = {mid: g["group_id"] for g in ref["nonsingleton_groups"] for mid in g["ids"]}
    dims = meta["dims"]
    points = np.zeros((len(ids), dims["max_atoms"], 9), np.float32)
    lengths = np.empty(len(ids), np.int16)
    encoded = np.zeros((len(ids), dims["smiles_length"]), np.int16)
    nodes, edges, attrs = [], [], []
    coordinates = qm8_selected_xyz(xyz, ids - 1)
    for i, (mid, smi) in enumerate(zip(ids, smiles)):
        tokens = atomwise_tokenizer(smi)
        encoded[i,:len(tokens)] = [ref["token_to_idx"][token] for token in tokens]
        x, attr, edge = graph_features(canonical_mol(smi, mid), meta)
        nodes.append(x)
        edges.append(edge)
        attrs.append(attr)
        record = coordinates[mid - 1]
        features = point_features(record["elements"], record["coordinates"])
        points[i,:len(features)] = features
        lengths[i] = len(features)
    arrays = dict(ids=ids, targets=targets, graph_x=np.concatenate(nodes),
                  node_offsets=np.cumsum([0, *map(len, nodes)], dtype=np.int64),
                  edge_offsets=np.cumsum([0, *map(len, attrs)], dtype=np.int64),
                  edge_index=np.concatenate(edges, axis=1), edge_attr=np.concatenate(attrs),
                  smiles_ids=encoded, points=points, point_lengths=lengths,
                  group_ids=np.array([groups.get(mid, mid) for mid in ids], dtype=np.int64))
    for key, value in arrays.items():
        if hashlib.sha256(value.tobytes()).hexdigest() != ref["array_sha256"][key]:
            raise ValueError(f"Source array differs from the recorded QM8 cache: {key}")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "cache.npz", **arrays)
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_id", "raw_smiles"])
        writer.writerows(zip(ids, smiles))
    metadata = {key: meta[key] for key in ("dims","atom_schema","bond_schema","target_names","target_units","point_features")}
    metadata.update(dataset="QM8", cache_sha256=sha256(out / "cache.npz"), manifest_sha256=sha256(out / "manifest.csv"),
                    token_to_idx=ref["token_to_idx"], points_source_sha256=ref["author_xyz_sha256"])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.table, args.xyz, args.out)
