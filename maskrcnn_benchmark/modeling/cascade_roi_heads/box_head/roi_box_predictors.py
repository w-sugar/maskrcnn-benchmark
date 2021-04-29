# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.modeling import registry
from torch import nn


# @registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
# class FastRCNNPredictor(nn.Module):
#     def __init__(self, config, in_channels):
#         super(FastRCNNPredictor, self).__init__()
#         assert in_channels is not None

#         num_inputs = in_channels

#         num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.cls_score = nn.Linear(num_inputs, num_classes)
#         num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
#         self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

#         nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
#         nn.init.constant_(self.cls_score.bias, 0)

#         nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
#         nn.init.constant_(self.bbox_pred.bias, 0)

#     def forward(self, x):
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         cls_logit = self.cls_score(x)
#         bbox_pred = self.bbox_pred(x)
#         return cls_logit, bbox_pred


# @registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
# class FPNPredictor(nn.Module):
#     def __init__(self, cfg, in_channels):
#         super(FPNPredictor, self).__init__()
#         num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         representation_size = in_channels

#         self.cls_score = nn.Linear(representation_size, num_classes)
#         num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
#         self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

#         nn.init.normal_(self.cls_score.weight, std=0.01)
#         nn.init.normal_(self.bbox_pred.weight, std=0.001)
#         for l in [self.cls_score, self.bbox_pred]:
#             nn.init.constant_(l.bias, 0)

#     def forward(self, x):
#         if x.ndimension() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#             x = x.view(x.size(0), -1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)

#         return scores, bbox_deltas

@registry.ROI_BOX_PREDICTOR.register("FPNIOUPredictorCascade")
class FPNIOUPredictorCascade(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNIOUPredictorCascade, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score_stage1 = nn.Linear(representation_size, num_classes)
        self.cls_score_stage2 = nn.Linear(representation_size, num_classes)
        self.cls_score_stage3 = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred_stage1 = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred_stage2 = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred_stage3 = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.iou_pred = nn.Linear(representation_size, num_classes)

        nn.init.normal_(self.cls_score_stage1.weight, std=0.01)
        nn.init.normal_(self.cls_score_stage2.weight, std=0.01)
        nn.init.normal_(self.cls_score_stage3.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_stage1.weight, std=0.001)
        nn.init.normal_(self.bbox_pred_stage2.weight, std=0.001)
        nn.init.normal_(self.bbox_pred_stage3.weight, std=0.001)
        nn.init.normal_(self.iou_pred.weight, std=0.01)
        for l in [self.cls_score_stage1, self.cls_score_stage2, self.cls_score_stage3, self.bbox_pred_stage1, self.bbox_pred_stage2, self.bbox_pred_stage3, self.iou_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x, stage=1):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        if stage == 1:
            scores = self.cls_score_stage1(x)
            bbox_deltas = self.bbox_pred_stage1(x)
        elif stage == 2:
            scores = self.cls_score_stage2(x)
            bbox_deltas = self.bbox_pred_stage2(x)
        elif stage == 3:
            scores = self.cls_score_stage3(x)
            bbox_deltas = self.bbox_pred_stage3(x)
        else:
            raise ValueError("Cascade stage is wrong!")
        iou_pred = self.iou_pred(x)

        return scores, bbox_deltas, iou_pred

@registry.ROI_BOX_PREDICTOR.register("FPNPredictorCascade")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score_stage1 = nn.Linear(representation_size, num_classes)
        self.cls_score_stage2 = nn.Linear(representation_size, num_classes)
        self.cls_score_stage3 = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred_stage1 = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred_stage2 = nn.Linear(representation_size, num_bbox_reg_classes * 4)
        self.bbox_pred_stage3 = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score_stage1.weight, std=0.01)
        nn.init.normal_(self.cls_score_stage2.weight, std=0.01)
        nn.init.normal_(self.cls_score_stage3.weight, std=0.01)
        nn.init.normal_(self.bbox_pred_stage1.weight, std=0.001)
        nn.init.normal_(self.bbox_pred_stage2.weight, std=0.001)
        nn.init.normal_(self.bbox_pred_stage3.weight, std=0.001)
        for l in [self.cls_score_stage1, self.cls_score_stage2, self.cls_score_stage3, self.bbox_pred_stage1, self.bbox_pred_stage2, self.bbox_pred_stage3]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x, stage=1):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        if stage == 1:
            scores = self.cls_score_stage1(x)
            bbox_deltas = self.bbox_pred_stage1(x)
        elif stage == 2:
            scores = self.cls_score_stage2(x)
            bbox_deltas = self.bbox_pred_stage2(x)
        elif stage == 3:
            scores = self.cls_score_stage3(x)
            bbox_deltas = self.bbox_pred_stage3(x)
        else:
            raise ValueError("Cascade stage is wrong!")

        return scores, bbox_deltas

def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
