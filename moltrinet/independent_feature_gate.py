"""Independent per-modality channel scores, normalized only across modalities."""
import copy

import torch
from torch import nn
from entmax import entmax15

from .model import FusionBase, GatedFusion


class IndependentFeatureFusion(FusionBase):
    """Each score MLP sees only its own LayerNorm-normalized encoder output."""

    def __init__(self, d_smiles, d_graph, d_point_output, d_output,
                 dropout=0.1, hidden_dim=None, mode="softmax"):
        super().__init__(d_smiles, d_graph, d_point_output, d_output, dropout)
        if mode not in ("softmax", "entmax15"):
            raise ValueError(f"Unknown independent-score normalization: {mode}")
        hidden_dim = 3 * self.d_model if hidden_dim is None else hidden_dim
        self._validate_hidden_dim(hidden_dim)
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.score_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d_model, hidden_dim), nn.GELU(),
                          nn.Dropout(dropout), nn.Linear(hidden_dim, self.d_model))
            for _ in range(3)
        ])
        self.last_logits = None
        self.last_weights = None

    @classmethod
    def from_existing(cls, fusion, mode):
        """Keep the original separate norms and prediction head; replace scoring."""
        if not isinstance(fusion, GatedFusion):
            raise TypeError("Expected the original GatedFusion")
        reference = next(fusion.parameters())
        result = cls(fusion.d_model, fusion.d_model, fusion.d_model,
                     fusion.fc[-1].out_features, dropout=fusion.gate_mlp[2].p,
                     hidden_dim=fusion.hidden_dim, mode=mode)
        result.to(device=reference.device, dtype=reference.dtype)
        for name in ("norm_S", "norm_G", "norm_P", "norm_final", "fc"):
            setattr(result, name, copy.deepcopy(getattr(fusion, name)))
        result.train(fusion.training)
        return result

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_details=False):
        if any(value is None for value in (smiles_emb, graph_emb, point_cloud_emb)):
            raise ValueError("This comparison requires all three encoder outputs")
        features, _ = self._prepare_features(smiles_emb, graph_emb, point_cloud_emb)
        logits = torch.stack([score(feature) for score, feature in
                              zip(self.score_mlps, features)], dim=1)
        weights = (torch.softmax(logits, dim=1) if self.mode == "softmax"
                   else entmax15(logits, dim=1))
        values = torch.stack(features, dim=1)
        mixed = (weights * values).sum(dim=1)
        output = self._predict(mixed)
        self.last_logits = logits.detach()
        self.last_weights = weights.detach()
        self.gates = tuple(weight.detach() for weight in weights.unbind(dim=1))
        if return_details:
            return output, {"features": values, "logits": logits,
                            "weights": weights, "mixed": mixed}
        return output


def configure_independent_gate(model, mode="softmax"):
    """Install independently scored channel weights on the three-modal backbone."""
    model.fusion_model = IndependentFeatureFusion.from_existing(model.fusion_model, mode)
    return model
