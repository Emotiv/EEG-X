import pytest
import torch

from pretrain.channel_identity import load_channel_vocabulary
from pretrain.models.position import FixedChannelEmbedding


@pytest.mark.parametrize("width", [64, 128, 512])
def test_fixed_positions_have_unit_gain_and_are_headset_independent(width):
    vocabulary = load_channel_vocabulary()
    rng = torch.get_rng_state()
    embedding = FixedChannelEmbedding(vocabulary, width)
    assert torch.equal(rng, torch.get_rng_state())
    assert not list(embedding.parameters())
    ids = torch.tensor([channel.index for channel in vocabulary.channels[:3]])
    torch.testing.assert_close(embedding(ids).norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(embedding(ids[:1]), embedding(ids)[:1], rtol=0, atol=0)
    assert torch.count_nonzero(embedding(ids)[..., -(width % 6):]) == 0
    reserved = torch.tensor([vocabulary.reserved_indices[0]])
    with pytest.raises(ValueError, match="Inactive or reserved"):
        embedding(reserved)
