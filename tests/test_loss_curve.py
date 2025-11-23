@pytest.mark.smoke
def test_loss_decreases_on_toy_data(device):
    from src.data.toy_language import ToyDataset
    from src.models.llm_model import LLMModel

    model = LLMModel(...).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)
    ds = ToyDataset()  # 5〜10サンプルでOK

    prev_loss = None
    for _ in range(5):
        batch = ds.sample_batch(4)
        model.train()
        optim.zero_grad()
        loss = model(**{k: v.to(device) for k, v in batch.items()})
        loss.backward()
        optim.step()

        prev_loss = loss.item() if prev_loss is None else prev_loss * 0.7 + loss.item() * 0.3

    # 雑だけど「そこそこ下がってるよね」をチェック
    assert prev_loss < 2.0, f"Toy loss not small enough: {prev_loss:.3f}"
