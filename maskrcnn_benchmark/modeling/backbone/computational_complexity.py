# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
import numpy as np
from thop import profile
    
class PAFPN_filler(nn.Module):

    def __init__(
        self, in_channels=256, out_channels=3, conv_block=conv_with_kaiming_uniform(True, True), middle_channels = 128
    ):
        super(PAFPN_filler, self).__init__()
        
        self.conv_up_1 = conv_block(in_channels, middle_channels, 3, 1)
        self.conv_up_2 = conv_block(middle_channels, middle_channels, 3, 1)

        self.conv_final = conv_with_kaiming_uniform()(middle_channels, 3, 1)

        #self.tanh = nn.Tanh()


    def forward(self, x):

        feature_out_1248 = x
        
        feature_out_1248_up_1 = self.conv_up_1(feature_out_1248)
        feature_out_1248_up_1_inpo = F.interpolate(feature_out_1248_up_1, scale_factor=2, mode="nearest")
        feature_out_1248_up_2 = self.conv_up_2(feature_out_1248_up_1_inpo)
        feature_out_1248 = F.interpolate(feature_out_1248_up_2, scale_factor=2, mode="nearest")
        
        final_out_1248 = self.conv_final(feature_out_1248)
        #filled_image = self.tanh(final_out_1248)
        # semseg_entropy = prob_2_entropy(F.softmax(semseg_pred, dim=1))

        return final_out_1248
        
model = PAFPN_filler()
flops, params = profile(model, inputs=(torch.randn(1, 256, 64, 64),))
print(flops / 1000 ** 3)
print(params / 1000 ** 2)
