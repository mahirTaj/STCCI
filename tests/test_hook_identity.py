"""Mechanical tensor tests only: no simulated dataset or model experiments."""
import torch

from data.validate_counterfactuals import validate_record
from models.intervention_hooks import batch_tensors, bounded_score, patch_head


def test_noop_and_exact_coordinate_replacement():
    # Index sentinels are not model activations and are never saved as effects.
    z = torch.arange(2 * 3 * 12 * 64).reshape(2, 3, 12, 64).float()
    positions = torch.tensor([1, 2])
    rows = torch.arange(2)
    source = z[rows, positions].clone()
    assert torch.equal(patch_head(source, positions, 1)(z, None), z)
    source[:, 1] = 0
    result = patch_head(source, positions, 1)(z, None)
    changed = torch.zeros_like(z, dtype=torch.bool)
    changed[rows, positions, 1] = True
    assert torch.equal(result[~changed], z[~changed])
    assert torch.equal(result[changed], torch.zeros(2 * 64))
    assert torch.equal(z, torch.arange(z.numel()).reshape_as(z).float())


def test_bounded_score_uses_final_token_and_float32():
    logits = torch.arange(2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
    positions = torch.tensor([0, 2])
    result = bounded_score(logits, positions, torch.tensor([3, 0]), torch.tensor([0, 3]))
    expected = torch.tanh(torch.tensor([1.5, -1.5]))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected)


def test_actual_record_padding(official_rows, tokenizer, config):
    records = [validate_record(row, i, tokenizer, config) for i, row in enumerate(official_rows)]
    clean, corrupt, mask, positions, target, distractor = batch_tensors(records, "cpu")
    for i, record in enumerate(records):
        length = len(record["clean_tokens"])
        assert positions[i] == length - 1
        assert mask[i].sum() == length
        assert clean[i, :length].tolist() == record["clean_tokens"]
        assert corrupt[i, :length].tolist() == record["corrupt_tokens"]


def test_hook_shape_failure():
    import pytest
    with pytest.raises(ValueError, match="dimensions"):
        patch_head(torch.empty(1, 12, 64), torch.tensor([0]), 0)(torch.empty(1, 1, 768), None)
