# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from mmcv.cnn import ConvModule, normal_init, xavier_init
from maskrcnn_benchmark.modeling.backbone.resnet import BottleneckWithFixedBatchNorm


class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@registry.ROI_BOX_FEATURE_EXTRACTORS.register("FPNConvFCFeatureExtractor")
class FPNConvFCFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNConvFCFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_BOX_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size
        # increase the channel of input features
        self.res_block = BasicResBlock(in_channels,
                                       1024)
        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # avepooling
        self.avg_pool = torch.nn.AvgPool2d(resolution)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers."""
        branch_convs = nn.ModuleList()
        for i in range(4):
            branch_convs.append(
                BottleneckWithFixedBatchNorm(
                    in_channels=1024,
                    bottleneck_channels=256,
                    out_channels=1024))
        return branch_convs

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x_fatter = x.view(x.size(0), -1)

        # fc-->用于分类头
        x_fc = F.relu(self.fc6(x_fatter))
        x_fc = F.relu(self.fc7(x_fc))

        # conv-->用于回归头
        x_conv = self.res_block(x)
        for conv in self.conv_branch:
            x_conv = conv(x_conv)
        x_conv = self.avg_pool(x_conv)
        x_conv = x_conv.view(x_conv.size(0), -1)

        return x_fc, x_conv


def make_roi_box_feature_extractor(cfg, in_channels):
    func = registry.ROI_BOX_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
