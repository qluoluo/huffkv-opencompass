import torch

from quantized_cache import QuantizedCache


def test_quantized_cache_keeps_fp_and_2bit_quantized_keys():
    cache = QuantizedCache(key_bits=2)

    key_step1 = torch.tensor(
        [
            [
                [[0.0, 1.0]],
                [[2.0, 3.0]],
            ]
        ],
        dtype=torch.float32,
    )
    value_step1 = torch.tensor(
        [
            [
                [[10.0, 20.0]],
                [[30.0, 40.0]],
            ]
        ],
        dtype=torch.float32,
    )

    keys, values = cache.update(key_step1, value_step1, layer_idx=0)
    torch.testing.assert_close(keys, key_step1)
    torch.testing.assert_close(values, value_step1)

    layer = cache.layers[0]
    assert layer.key_quantized.shape[-1] == 1  # packed 2-bit keys -> 4 vals per byte
    assert layer.key_quantized.dtype == layer.key_quant_dtype
    expected_scale = torch.tensor([[[2.0 / 3.0, 2.0 / 3.0]]], dtype=key_step1.dtype)
    assert layer.key_scale.shape == expected_scale.shape
    torch.testing.assert_close(layer.key_scale, expected_scale)

    expected_quant_step1 = torch.tensor(
        [
            [
                [[0]],
                [[15]],
            ]
        ],
        dtype=layer.key_quant_dtype,
    )
    assert torch.equal(layer.key_quantized, expected_quant_step1)

    key_step2 = torch.tensor([[[[-1.0, 5.0]]]], dtype=torch.float32)
    value_step2 = torch.tensor([[[[50.0, 60.0]]]], dtype=torch.float32)
    keys, values = cache.update(key_step2, value_step2, layer_idx=0)

    torch.testing.assert_close(keys, torch.cat([key_step1, key_step2], dim=1))
    torch.testing.assert_close(values, torch.cat([value_step1, value_step2], dim=1))
    torch.testing.assert_close(layer.key_scale, expected_scale)  # scale stays fixed

    expected_quant_full = torch.tensor(
        [
            [
                [[0]],
                [[15]],
                [[12]],
            ]
        ],
        dtype=layer.key_quant_dtype,
    )
    assert torch.equal(layer.key_quantized, expected_quant_full)


if __name__ == "__main__":
    test_quantized_cache_keeps_fp_and_2bit_quantized_keys()
    print("quantized cache basic test passed")
