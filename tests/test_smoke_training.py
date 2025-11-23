import pytest
import torch


@pytest.mark.smoke
def test_single_training_step_runs(device):
    from src.data.dummy_dataset import DummyDataset
    from src.models.llm_model import LLMModel

    model = LLMModel(...).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ds = DummyDataset(num_samples=8)
    batch = ds[0]  # (input_ids, labels) など

    model.train()
    optim.zero_grad()
    loss = model(**{k: v.to(device) for k, v in batch.items()})
    loss.backward()
    optim.step()

    assert torch.isfinite(loss).all(), "Loss is NaN or inf"
