# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.box_coder import BoxCoder

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))

    def box_regression_to_proposals(self, box_regression, boxes, is_train=True, class_logits=None):
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        boxes_per_level = 
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        # if self.cls_agnostic_bbox_reg:
        #     box_regression = box_regression[:, -4:]
        if is_train:
            labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0)
            sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
            labels_pos = labels[sampled_pos_inds_subset]
            # if self.cls_agnostic_bbox_reg:
            #     map_inds = torch.tensor([4, 5, 6, 7], device=device)
            # else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

            box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
        else:
            if class_logits is not None:
                bbox_label = [s.argmax(dim=1) for s in class_logits]
                map_inds = 4 * bbox_label + torch.tensor(
                    [0, 1, 2, 3], device=device)
                box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
    

        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        # if self.cls_agnostic_bbox_reg:
        #     proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        # 按照图像split
        proposals = proposals.split(boxes_per_image, dim=0)
        return proposals

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets, stage=1)

            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x, stage=1)

            # 2nd stage
            with torch.no_grad():
                proposals_stage2 = self.box_regression_to_proposals(box_regression, proposals)
                proposals_stage2 = self.loss_evaluator.subsample(proposals_stage2, targets, stage=2)

            x = self.feature_extractor(features, proposals_stage2)
            class_logits_stage2, box_regression_stage2 = self.predictor(x, stage=2)

            # 3rd stage
            with torch.no_grad():
                proposals_stage3 = self.box_regression_to_proposals(box_regression_stage2, proposals_stage2)
                proposals_stage3 = self.loss_evaluator.subsample(proposals_stage3, targets, stage=3)
            
            x = self.feature_extractor(features, proposals_stage3)
            class_logits_stage3, box_regression_stage3 = self.predictor(x, stage=3)
        else:
            x = self.feature_extractor(features, proposals)
            # final classifier that converts the features into predictions
            class_logits, box_regression = self.predictor(x, stage=1)
            proposals_stage2 = self.box_regression_to_proposals(box_regression, proposals, False, class_logits)
            x = self.feature_extractor(features, proposals_stage2)
            class_logits_stage2, box_regression_stage2 = self.predictor(x, stage=2)
            proposals_stage3 = self.box_regression_to_proposals(box_regression_stage2, proposals_stage2, False, class_logits_stage2)
            x = self.feature_extractor(features, proposals_stage3)
            class_logits_stage3, box_regression_stage3 = self.predictor(x, stage=3)

            class_logits_average = (class_logits + class_logits_stage2 + class_logits_stage3) / 3


        if not self.training:
            result = self.post_processor((class_logits_average, box_regression_stage3), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        loss_classifier_stage2, loss_box_reg_stage2 = self.loss_evaluator(
            [class_logits_stage2], [box_regression_stage2]
        )
        loss_classifier_stage3, loss_box_reg_stage3 = self.loss_evaluator(
            [class_logits_stage3], [box_regression_stage3]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg,
            loss_classifier_stage2=loss_classifier_stage2, loss_box_reg_stage2=loss_box_reg_stage2,
            loss_classifier_stage3=loss_classifier_stage3, loss_box_reg_stage3=loss_box_reg_stage3
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
