"""Run the modality-subset and channel-normalization checks on CPU."""
import torch
from torch.utils.data import DataLoader
from entmax import entmax15

from moltrinet.runner import MODALITIES, PAPER, ROOT, arguments, build_model
from moltrinet.data import QM9Dataset, collate_qm9


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = QM9Dataset(ROOT / "examples/qm8", "E1-CC2", indices=[0, 1, 2])
    batch = next(iter(DataLoader(data, batch_size=3, collate_fn=collate_qm9)))
    inputs = arguments(batch, "cpu")
    cfg = {"architecture": PAPER["architecture"], "gate": "softmax", "dense_context": "embeddings"}
    for subset in MODALITIES:
        model = build_model(dict(cfg, modalities=subset), data.dims, "cpu").eval()
        pred = model(*inputs)
        assert pred.shape == (3, 1) and torch.isfinite(pred).all()
        if len(subset) > 1:
            weights = torch.stack(model.fusion_model.gates, dim=1)
            torch.testing.assert_close(weights.sum(1), torch.ones_like(weights[:, 0]))
            for i, m in enumerate("SGP"):
                if m not in subset:
                    assert torch.count_nonzero(weights[:, i]) == 0
        else:
            assert model.fusion_model is None

    model = build_model(dict(cfg, modalities="SGP", gate="entmax15"), data.dims, "cpu").eval()
    pred, embeddings = model(*inputs, return_embeddings=True)
    head = model.fusion_model
    again, detail = head(*embeddings, return_details=True)
    torch.testing.assert_close(pred, again)
    q, v = detail["weights"], detail["features"]
    torch.testing.assert_close(q.sum(1), torch.ones_like(q[:, 0]))
    torch.testing.assert_close(q, entmax15(detail["logits"], dim=1))
    full = head._predict(detail["mixed"] - q[:, 0] * v[:, 0])
    torch.testing.assert_close(full, head._predict((q[:, 1:] * v[:, 1:]).sum(1)), rtol=1e-4, atol=1e-6)
    head.score_mlps[0][-1].bias.sub_(100.)
    _, excluded = head(*embeddings, return_details=True)
    assert torch.count_nonzero(excluded["weights"][:, 0]) == 0
    print("PASS: seven subsets, modality normalization, exact entmax zeros and fixed-gate deletion.")


if __name__ == "__main__":
    main()
