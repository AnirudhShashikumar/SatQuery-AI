import torch

from changerex_local.architecture import build_model, model_parameter_count
from changerex_local.config import PARAMETER_COUNT


def test_exact_architecture_shapes_and_parameter_count():
    model = build_model().eval()
    pair = torch.zeros(1, 6, 64, 64)
    with torch.inference_mode():
        features = model.backbone(pair[:, :3], pair[:, 3:])
        logits = model(pair)
    assert [tuple(feature.shape) for feature in features] == [
        (1, 128, 16, 16),
        (1, 256, 8, 8),
        (1, 512, 4, 4),
        (1, 1024, 2, 2),
    ]
    assert logits.shape == (1, 2, 64, 64)
    assert model_parameter_count(model) == PARAMETER_COUNT == 11_390_946
    assert len(model.state_dict()) == 173


def test_official_normalization_and_decoder_details():
    model = build_model()
    assert isinstance(model.backbone.stem[1], torch.nn.SyncBatchNorm)
    assert isinstance(model.decode_head.convs[0].bn, torch.nn.SyncBatchNorm)
    assert isinstance(model.decode_head.neck_layer.flow_make[1], torch.nn.InstanceNorm2d)
    assert model.decode_head.neck_layer.flow_make[0].groups == 128
    assert model.decode_head.conv_seg.out_channels == 2
