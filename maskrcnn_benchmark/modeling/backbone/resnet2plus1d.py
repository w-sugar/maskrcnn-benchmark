from .resnet3d import ResNet3d
import torch.nn as nn

class ResNet2Plus1d(ResNet3d):
    """ResNet (2+1)d backbone.
    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.pretrained2d is False
        assert self.conv_cfg['type'] == 'Conv2plus1d'
        self.lateral_convs = nn.ModuleList()
        in_channels = [64, 128]
        temporal_strides = [4, 2]
        for i in range(2):
            l_conv = nn.Conv3d(
                in_channels[i],
                in_channels[i],
                (temporal_strides[i], 1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True)
            self.lateral_convs.append(l_conv)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)


        output = []
        for layer_name in self.res_layers:
            res_layer = getattr(self, layer_name)
            # no pool2 in R(2+1)d
            x = res_layer(x)
            output.append(x)

        result = []
        for idx, feat in enumerate(output):
            if idx < 2:
                feat = self.lateral_convs[idx](feat)
            feat = feat.squeeze(2)
            result.append(feat)
            
        return result #x