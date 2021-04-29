# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn

@registry.ROI_BOX_PREDICTOR.register("FPNPredictorDouble")
class FPNPredictorDouble(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictorDouble, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x_fc, x_conv):
        if x_fc.ndimension() == 4:
            assert list(x_fc.shape[2:]) == [1, 1]
            x_fc = x_fc.view(x_fc.size(0), -1)
        if x_conv.ndimension() == 4:
            assert list(x_conv.shape[2:]) == [1, 1]
            x_conv = x_conv.view(x_conv.size(0), -1)
        scores = self.cls_score(x_fc)
        bbox_deltas = self.bbox_pred(x_conv)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
