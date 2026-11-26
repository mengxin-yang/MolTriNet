"""SMILES/graph CMCR: two projection heads and symmetric in-batch InfoNCE."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


ORIGINAL_CMCR = {
    "weight": 0.03,
    "temperature": 0.2,
    "modalities": ["S", "G"],
    "projection_hidden_dim": 256,
    "projection_dim": 64,
    "dropout": 0.1,
}


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, proj_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(input_dim, proj_dim),
        )

    def forward(self, value):
        return self.net(value)


class SGProjectors(nn.Module):
    def __init__(self, input_dim=256, projection_dim=64, dropout=0.1):
        super().__init__()
        self.S = ProjectionHead(input_dim, projection_dim, dropout)
        self.G = ProjectionHead(input_dim, projection_dim, dropout)

    def forward(self, smiles_embedding, graph_embedding, temperature=0.2):
        z_s = F.normalize(self.S(smiles_embedding), dim=-1)
        z_g = F.normalize(self.G(graph_embedding), dim=-1)
        logits = z_s @ z_g.t() / temperature
        labels = torch.arange(z_s.size(0), device=z_s.device)
        return 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))


def build_projectors(recipe, device):
    """Head initialization does not advance the backbone/dropout RNG state.

    Construction is on CPU, followed by transfer. Training head dropout keeps
    the original shared Torch RNG; its future draws are not paired with CMCR-off.
    """
    with torch.random.fork_rng(devices=[]):
        heads = SGProjectors(recipe["projection_hidden_dim"], recipe["projection_dim"], recipe["dropout"])
    return heads.to(device)


def projector_gradient_summary(rows):
    result = {}
    for name in ("S", "G"):
        values = [row["projector"][name] for row in rows if row["projector"]]
        if not values:
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite CMCR projector gradient")
        result[name] = {"mean": sum(values) / len(values), "min": min(values),
                        "max": max(values), "zero_fraction": sum(value == 0 for value in values) / len(values)}
    return result
