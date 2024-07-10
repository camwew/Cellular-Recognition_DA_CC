# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
import numpy as np


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class PAFPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, cfg, in_channels, out_channels, conv_block, middle_channels = 128
    ):
        """
        Arguments:

        """
        super(PAFPN, self).__init__()

        # When defining each conv, its name wil be like conv_{scale}_{No. of conv for this scale}

        self.conv_8_1 = conv_block(in_channels, middle_channels, 3, 1)
        self.conv_8_2 = conv_block(middle_channels, middle_channels, 3, 1)
        self.conv_8_3 = conv_block(middle_channels, middle_channels, 3, 1)

        self.conv_4_1 = conv_block(in_channels, middle_channels, 3, 1)
        self.conv_4_2 = conv_block(middle_channels, middle_channels, 3, 1)

        self.conv_2_1 = conv_block(in_channels, middle_channels, 3, 1)

        self.conv_1_1 = conv_block(in_channels, middle_channels, 3, 1)

        self.conv_final = conv_with_kaiming_uniform()(middle_channels, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        """

        """

        feature_1x = x[0]
        feature_2x = x[1]
        feature_4x = x[2]
        feature_8x = x[3]

        feature_1x_out = self.conv_1_1(feature_1x)

        feature_2x_out_1 = self.conv_2_1(feature_2x)
        feature_2x_out = F.interpolate(feature_2x_out_1, scale_factor=2, mode="nearest")

        feature_4x_out_1 = self.conv_4_1(feature_4x)
        feature_4x_out_1 = F.interpolate(feature_4x_out_1, scale_factor=2, mode="nearest")
        feature_4x_out_2 = self.conv_4_2(feature_4x_out_1)
        feature_4x_out = F.interpolate(feature_4x_out_2, scale_factor=2, mode="nearest")

        feature_8x_out_1 = self.conv_8_1(feature_8x)
        feature_8x_out_1 = F.interpolate(feature_8x_out_1, scale_factor=2, mode="nearest")
        feature_8x_out_2 = self.conv_8_2(feature_8x_out_1)
        feature_8x_out_2 = F.interpolate(feature_8x_out_2, scale_factor=2, mode="nearest")
        feature_8x_out_3 = self.conv_8_3(feature_8x_out_2)
        feature_8x_out = F.interpolate(feature_8x_out_3, scale_factor=2, mode="nearest")

        feature_out_1248 = feature_1x_out + feature_2x_out + feature_4x_out + feature_8x_out
        final_out_1248 = self.conv_final(feature_out_1248)
        semseg_pred = F.interpolate(final_out_1248, scale_factor=4, mode="nearest")
        semseg_entropy = prob_2_entropy(F.softmax(semseg_pred, dim=1))
        # print('semseg pred, ', semseg_pred.size())
        # print('feature out 1248 size', feature_out_1248.size())

        # return x, semseg_pred

        return semseg_pred, semseg_entropy



from torch.autograd import Function
class GCLayerF(Function):

    @staticmethod
    def forward(ctx, input, grl_alpha):
        ctx.alpha= grl_alpha
        # print('grl alpha in cfg is, ', grl_alpha)
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs * ctx.alpha
        return output, None

def grad_cut(x, grl_alpha = 1.0):
    return GCLayerF.apply(x, grl_alpha)
    
class PAFPN_filler(nn.Module):

    def __init__(
        self, cfg, in_channels, out_channels, conv_block, middle_channels = 128
    ):
        super(PAFPN_filler, self).__init__()
        self.cfg = cfg

        # When defining each conv, its name wil be like conv_{scale}_{No. of conv for this scale}
        
        self.conv_up_1 = conv_block(in_channels, middle_channels, 3, 1)
        self.conv_up_2 = conv_block(middle_channels, middle_channels, 3, 1)

        self.conv_final = conv_with_kaiming_uniform()(middle_channels, 3, 1)

        self.tanh = nn.Tanh()


    def forward(self, x, grl):
        grl = grl * self.cfg.MODEL.PAFPN_FILLER_GRL
        feature_1x = grad_cut(x[0], grl)

        feature_out_1248 = feature_1x
        
        feature_out_1248_up_1 = self.conv_up_1(feature_out_1248)
        feature_out_1248_up_1_inpo = F.interpolate(feature_out_1248_up_1, scale_factor=2, mode="nearest")
        feature_out_1248_up_2 = self.conv_up_2(feature_out_1248_up_1_inpo)
        feature_out_1248 = F.interpolate(feature_out_1248_up_2, scale_factor=2, mode="nearest")
        
        final_out_1248 = self.conv_final(feature_out_1248)
        filled_image = self.tanh(final_out_1248)
        # semseg_entropy = prob_2_entropy(F.softmax(semseg_pred, dim=1))

        return filled_image