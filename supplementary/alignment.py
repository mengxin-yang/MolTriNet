"""SMILES/graph alignment and complete-pool top-1 retrieval."""
import numpy as np


def l2_normalize_np(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def paired_cosine_np(a, b):
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return np.sum(a * b, axis=1)


def sampled_negative_cosine_np(a, b, seed):
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    same = perm == np.arange(n)
    if np.any(same):
        perm[same] = np.roll(perm, 1)[same]
    return np.sum(a * b[perm], axis=1)


def top1_cross_modal_retrieval_np(query, key, chunk_size=1024):
    query = l2_normalize_np(query)
    key = l2_normalize_np(key)
    n = query.shape[0]
    correct = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = query[start:end] @ key.T
        pred = np.argmax(sim, axis=1)
        true = np.arange(start, end)
        correct += int(np.sum(pred == true))

    return correct / max(n, 1)


def summarize_alignment_space(space_name, smiles_matrix, graph_matrix, seed):
    if smiles_matrix.shape[1] != graph_matrix.shape[1]:
        return {
            "Space": space_name,
            "Dim_SMILES": smiles_matrix.shape[1],
            "Dim_Graph": graph_matrix.shape[1],
            "Positive_Cosine_Mean": np.nan,
            "Positive_Cosine_Std": np.nan,
            "Negative_Cosine_Mean": np.nan,
            "Negative_Cosine_Std": np.nan,
            "Cosine_Gap": np.nan,
            "S_to_G_Top1": np.nan,
            "G_to_S_Top1": np.nan,
            "Note": "Skipped because SMILES and graph embedding dimensions differ.",
        }

    pos = paired_cosine_np(smiles_matrix, graph_matrix)
    neg = sampled_negative_cosine_np(smiles_matrix, graph_matrix, seed=seed)
    s_to_g = top1_cross_modal_retrieval_np(smiles_matrix, graph_matrix)
    g_to_s = top1_cross_modal_retrieval_np(graph_matrix, smiles_matrix)

    return {
        "Space": space_name,
        "Dim_SMILES": smiles_matrix.shape[1],
        "Dim_Graph": graph_matrix.shape[1],
        "Positive_Cosine_Mean": float(np.mean(pos)),
        "Positive_Cosine_Std": float(np.std(pos, ddof=1)) if len(pos) > 1 else 0.0,
        "Negative_Cosine_Mean": float(np.mean(neg)),
        "Negative_Cosine_Std": float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0,
        "Cosine_Gap": float(np.mean(pos) - np.mean(neg)),
        "S_to_G_Top1": float(s_to_g),
        "G_to_S_Top1": float(g_to_s),
        "Note": "",
    }

def main():
    import argparse
    import csv
    import json
    from pathlib import Path
    import torch
    from torch.nn.functional import normalize
    from moltrinet.runner import load_run
    from supplementary.cmcr import SGProjectors

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", required=True, type=Path)
    cli.add_argument("--cache", required=True, type=Path)
    cli.add_argument("--split", required=True, type=Path)
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf8"))
    model, loader, _ = load_run(args.run, args.cache, args.split, phase="test", device=args.device)
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    heads = SGProjectors().to(args.device)
    heads.load_state_dict(checkpoint["cmcr_projectors"], strict=True)
    model.eval()
    heads.eval()
    matrices = {key: [] for key in ("h_s", "h_g", "z_s", "z_g")}
    with torch.inference_mode():
        for batch in loader:
            h_s = model.smiles_model(batch["smiles_ids"].to(args.device))
            h_g = model.graph_model(batch["graph"].to(args.device))
            z_s = normalize(heads.S(h_s), dim=-1)
            z_g = normalize(heads.G(h_g), dim=-1)
            for key, tensor in zip(matrices, (h_s, h_g, z_s, z_g)):
                matrices[key].append(tensor.cpu().numpy())
    stacked = {key: np.concatenate(parts) for key, parts in matrices.items()}
    rows = []
    for space, s, g in (("Encoder_h", "h_s", "h_g"), ("Projection_z", "z_s", "z_g")):
        row = summarize_alignment_space(space, stacked[s], stacked[g], config["seed"])
        row["Bidirectional_Top1_pct"] = 50 * (row["S_to_G_Top1"] + row["G_to_S_Top1"])
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
