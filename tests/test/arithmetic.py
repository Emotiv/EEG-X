"""Dense formulas must match real reference forward and backward execution."""

import pytest
import torch

from pretrain.models.factory import create_model
from tools.queue.flops import dense_counts, verify


@pytest.mark.parametrize('width,channels,batch', [(16, 3, 1), (32, 4, 2)])
@pytest.mark.parametrize('dictionary', [False, True])
def test_dense_counts_match_executed_attention_and_dictionary_backward(width, channels, batch, dictionary):
    objective = ({'name': 'raw_soft_dict_v2', 'weight': 1.0, 'temperature': 1.0,
                  'groups': 2, 'k': 4, 'kernel_sizes': [3, 5], 'dilations': [1, 2], 'seed': 11}
                 if dictionary else None)
    model = create_model({'width': width, 'depth': 1, 'heads': 8, 'seed': 11,
                          'input_samples': 128, 'objective': objective})
    result = verify(model, torch.randn(batch, channels, 128),
                    torch.arange(channels)[None].expand(batch, -1), classes=3)
    expected = dense_counts(model, channels, 128, 3)
    assert result['inference'] == batch * expected['inference']
    if dictionary:
        convolution = 2 * channels * 128 * 2 * 4 * 2 * (3 + 5)
        assert expected['soft_dict_training'] == 4 * convolution
    else:
        assert expected['soft_dict_training'] == 0


def test_counts_reject_a_different_input_duration():
    model = create_model({'width': 16, 'depth': 1, 'heads': 8, 'input_samples': 128})
    with pytest.raises(ValueError, match='geometry'):
        dense_counts(model, 4, 256, 2)
