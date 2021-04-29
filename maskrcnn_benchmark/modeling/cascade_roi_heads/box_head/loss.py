# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss, CrossEntropyLossSmooth
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, bbox_overlaps
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler, WeightedPositiveSampler
)
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList

class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher_stage1,
        proposal_matcher_stage2,
        proposal_matcher_stage3,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher_stage1 = proposal_matcher_stage1
        self.proposal_matcher_stage2 = proposal_matcher_stage2
        self.proposal_matcher_stage3 = proposal_matcher_stage3
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.CrossEntropyLossSmooth = CrossEntropyLossSmooth()

    def match_targets_to_proposals(self, proposal, target, stage=1):
        match_quality_matrix = boxlist_iou(target, proposal)
        if stage == 1:
            matched_idxs, matched_vals = self.proposal_matcher_stage1(match_quality_matrix)
        elif stage == 2:
            matched_idxs, matched_vals = self.proposal_matcher_stage2(match_quality_matrix)
        elif stage == 3:
            matched_idxs, matched_vals = self.proposal_matcher_stage3(match_quality_matrix)
        else:
            raise ValueError("Cascade stage is wrong!")
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        matched_targets.add_field("matched_vals", matched_vals)
        return matched_targets

    def prepare_targets(self, proposals, targets, stage=1):
        labels = []
        ious = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image, stage
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            ious_per_image = matched_targets.get_field("matched_vals")
            ious_per_image = ious_per_image.to(dtype=torch.float32)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            ious_per_image[bg_inds] = 1 #1

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            ious.append(ious_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, ious, regression_targets

    def prepare_iou_targets(self, proposals, box_regression, targets):
        concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        boxes_per_image = [len(box) for box in proposals]
        bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
        box_coder = BoxCoder(weights=bbox_reg_weights)

        # [N, 4 * num_classes]
        pred_boxes = box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        pred_boxes = pred_boxes.split(boxes_per_image, dim=0)

        proposals = list(proposals)
        device = box_regression.device
        for pred_boxes_per_image, proposals_per_image, targets_per_image in zip(pred_boxes, proposals, targets):

            # NOTE: Steps here may generate some wrong indices for box regression.
            # However, these cases would be filtered in the loss by sampled_pos_inds_subset
            labels_per_image = proposals_per_image.get_field("labels")
            if self.cls_agnostic_bbox_reg:
                labels = labels_per_image.new_zeros(labels_per_image.shape)
                map_inds = 4 * labels[:, None] + torch.tensor([4, 5, 6, 7], device=device)
            else:
                map_inds = 4 * labels[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
            # map_inds = 4 * labels_per_image[:, None] + torch.tensor([0, 1, 2, 3], device=device)
            pred_boxes_per_image = torch.gather(pred_boxes_per_image, 1, map_inds)
            if pred_boxes_per_image.shape[0] < 1:
                matched_ious = proposals_per_image.get_field("ious")
                proposals_per_image.add_field("iou_pred_targets_final", matched_ious)
                continue

            pred_boxlist_per_image = BoxList(pred_boxes_per_image, proposals_per_image.size, mode='xyxy')
            # [target_num, pred_boxes_num]
            match_quality_matrix = boxlist_iou(targets_per_image, pred_boxlist_per_image)
            # [pred_boxes_num]
            matched_ious, _ = match_quality_matrix.max(dim=0)
            # matched_ious, matches = match_quality_matrix.max(dim=0)
            # Assign candidate matches with low quality to negative (unassigned) values
            sampled_neg_inds_subset = torch.nonzero(labels_per_image == 0).squeeze(1)
            matched_ious[sampled_neg_inds_subset] = 1
            proposals_per_image.add_field("iou_pred_targets_final", matched_ious)

        return proposals

    def subsample(self, proposals, targets, stage=1):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, ious, regression_targets = self.prepare_targets(proposals, targets, stage)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, ious_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, ious, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("ious", ious_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression, proposals, iou_pred=None, targets=None):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        # proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        ious = cat([proposal.get_field("ious") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        # weight_ce = torch.tensor([1, 2, 1.5, 1.5], device=labels.device).to(dtype=torch.float32)
        classification_loss = F.cross_entropy(class_logits, labels)
        # classification_loss = self.CrossEntropyLossSmooth(class_logits, labels, ious)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        if iou_pred:
            iou_pred = cat(iou_pred, dim=0)
        # iou_targets = ious.new_zeros(ious.shape)
        # boxes_per_image = [len(box) for box in proposals]
        # concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        # pos_decode_bbox_pred = self.box_coder.decode(
        #     box_regression[sampled_pos_inds_subset[:, None], map_inds], concat_boxes[sampled_pos_inds_subset]
        # )
        # gt_bboxes = self.box_coder.decode(
        #     regression_targets[sampled_pos_inds_subset], concat_boxes[sampled_pos_inds_subset]
        # )
        # iou_targets[sampled_pos_inds_subset] = bbox_overlaps(pos_decode_bbox_pred, gt_bboxes, is_aligned=True)
        # iou_loss = F.binary_cross_entropy_with_logits(iou_pred.squeeze(), iou_targets.squeeze())
        # iou_targets = ious.new_ones(ious.shape)
        # boxes_per_image = [len(box) for box in proposals]
        # concat_boxes = torch.cat([a.bbox for a in proposals], dim=0)
        # pos_decode_bbox_pred = self.box_coder.decode(
        #     box_regression[sampled_pos_inds_subset[:, None], map_inds], concat_boxes[sampled_pos_inds_subset]
        # )
        # gt_bboxes = self.box_coder.decode(
        #     regression_targets[sampled_pos_inds_subset], concat_boxes[sampled_pos_inds_subset]
        # )
        # iou_targets[sampled_pos_inds_subset] = bbox_overlaps(pos_decode_bbox_pred, gt_bboxes, is_aligned=True)
            proposals = self.prepare_iou_targets(proposals, box_regression, targets)
            iou_pred_targets = cat([proposal.get_field("iou_pred_targets_final") for proposal in proposals], dim=0)
            iou_loss = self.CrossEntropyLossSmooth(iou_pred, labels, iou_pred_targets)
            # one_hot_lable = torch.FloatTensor(iou_pred.shape[0], iou_pred.shape[1]).to(device=labels.device)
            # print('a',iou_pred_targets)
            # print('b',iou_pred)
            # one_hot_lable.zero_()
            # one_hot_lable.scatter_(1, torch.reshape(labels, (class_logits.shape[0], 1)), iou_pred)
            # print('c',one_hot_lable)
            # iou_loss = smooth_l1_loss(
            #     iou_pred[sampled_pos_inds_subset, labels_pos][:, None], # if cls-aware
            #     iou_pred_targets[sampled_pos_inds_subset][:, None],
            #     size_average=False,
            #     beta=1,
            # )
            # iou_loss = iou_loss / labels_pos.numel()
        # one_hot_lable = torch.FloatTensor(class_logits.shape[0], class_logits.shape[1]).to(device=labels.device)
        # one_hot_lable.zero_()
        # one_hot_lable.scatter_(1, torch.reshape(labels, (class_logits.shape[0], 1)), ious.unsqueeze(1))
        # iou_loss = smooth_l1_loss(iou_pred, one_hot_lable,size_average=True, beta=1)
            return classification_loss, box_loss, iou_loss
        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher_stage1 = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    matcher_stage2 = Matcher(
        0.6,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    matcher_stage3 = Matcher(
        0.7,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    # fg_bg_sampler = BalancedPositiveNegativeSampler(
    #     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    # )
    fg_bg_sampler = WeightedPositiveSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION 
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher_stage1,
        matcher_stage2,
        matcher_stage3,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
