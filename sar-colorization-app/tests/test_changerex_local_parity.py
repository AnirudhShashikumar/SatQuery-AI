import numpy as np

from changerex_local.parity import ParityThresholds, compare_outputs


def test_parity_acceptance_thresholds():
    probability = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
    mask = probability >= 0.5
    exact = compare_outputs(probability, mask, probability.copy(), mask.copy())
    assert exact.accepted and exact.mask_iou == 1.0 and exact.binary_mask_agreement == 1.0

    divergent = compare_outputs(probability, mask, 1 - probability, ~mask)
    assert not divergent.accepted
    assert not divergent.mask_pass


def test_parity_rejects_shape_mismatch():
    result = compare_outputs(
        np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 3)), np.zeros((3, 3))
    )
    assert not result.accepted
    assert not result.probability_shape_match
