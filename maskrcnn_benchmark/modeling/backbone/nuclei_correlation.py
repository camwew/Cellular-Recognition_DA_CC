# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
import numpy as np
from maskrcnn_benchmark.modeling.poolers import Pooler_single as Pooler
from functools import partial
from timm.models.vision_transformer import Block
from .mae.util.pos_embed import get_2d_sincos_pos_embed
from .mae.util.position import find_center

from torch.autograd import Function


from thop import profile


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
    
class Transformer_nuclei_filler(nn.Module):
    def __init__(
        self, cfg, embed_dim=256, depth=12, num_heads=16,
                 decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Transformer_nuclei_filler, self).__init__()
        
        self.cfg = cfg
        
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=(0.25,),
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler
        
        self.pos_embed = nn.Parameter(torch.zeros(256*256 + 1, embed_dim, requires_grad=False))
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 256, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(256*256 + 1, decoder_embed_dim), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 256, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float())
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred_recon = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        self.decoder_pred_class = nn.Linear(decoder_embed_dim, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, bias=True)

        self.dropout = nn.Dropout(p=0.5, inplace=False)
        
        self.l1loss = nn.L1Loss()
        self.l1loss_noreduce = nn.L1Loss(reduction = 'none')
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, features, proposals, mrcn_class_logits, mrcn_uncertainty, grl):
        nuclei_filler_grl = grl * self.cfg.MODEL.NUCLEI_FILLER_GRL
        features_0_cut = grad_cut(features[0], nuclei_filler_grl)
        
        x = self.pooler(features_0_cut, proposals)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        centers_index, center_nuclei_index = find_center(proposals[0].bbox)
        nuclei_pos_embed = self.pos_embed[centers_index]
        decoder_nuclei_pos_embed = self.decoder_pos_embed[centers_index]
        gt_feature = x[center_nuclei_index].unsqueeze(0)
        
        pos_x = x + nuclei_pos_embed
        
        # center_nuclei_pos_embed = nuclei_pos_embed[center_nuclei_index].unsqueeze(0)
        # neighbors_pos_embed = nuclei_pos_embed[torch.arange(nuclei_pos_embed.size(0)) != center_nuclei_index] 
        # center_nuclei_x = pos_x[center_nuclei_index].unsqueeze(0)
        neighbors_x = pos_x[torch.arange(pos_x.size(0)) != center_nuclei_index]
        decoder_center_nuclei_pos_embed = decoder_nuclei_pos_embed[center_nuclei_index].unsqueeze(0)
        decoder_neighbors_pos_embed = decoder_nuclei_pos_embed[torch.arange(decoder_nuclei_pos_embed.size(0)) != center_nuclei_index] 
        decoder_nuclei_pos_embed = torch.cat([decoder_neighbors_pos_embed, decoder_center_nuclei_pos_embed], dim=0)
        decoder_nuclei_pos_embed = decoder_nuclei_pos_embed.unsqueeze(0)
        
        neighbors_x = neighbors_x.unsqueeze(0)
        for blk in self.blocks:
            neighbors_x = blk(neighbors_x)
        neighbors_x = self.norm(neighbors_x)
        
        neighbors_x = self.decoder_embed(neighbors_x)
        out_x = torch.cat([neighbors_x, self.mask_token], dim=1)
        out_x = out_x + decoder_nuclei_pos_embed
        for blk in self.decoder_blocks:
            out_x = blk(out_x)
            
            
        out_x = self.decoder_norm(out_x)
        recon_out = self.decoder_pred_recon((out_x.squeeze(0))[-1].unsqueeze(0))
        
        l1_loss = self.l1loss(gt_feature, recon_out) * 10
        
        class_logits = self.decoder_pred_class(out_x.squeeze(0))
        device = class_logits.device
        class_labels = proposals[0].get_field("labels").type(torch.LongTensor).to(device)
        classification_loss = F.cross_entropy(class_logits, class_labels) * 2
        
        ############################################################################
        
        mrcn_class_logits_grl = grl * self.cfg.MODEL.MRCN_CLASS_LOGITS_GRL
        mrcn_class_logits = grad_cut(mrcn_class_logits, mrcn_class_logits_grl)
        
        pred_pos_x = pos_x.unsqueeze(0)
        for blk in self.blocks:
            pred_pos_x = blk(pred_pos_x)
        pred_pos_x = self.norm(pred_pos_x)
        
        pred_pos_x = self.decoder_embed(pred_pos_x)
        #out_x = torch.cat([pred_pos_x, self.mask_token], dim=1)
        decoder_nuclei_pos_embed = self.decoder_pos_embed[centers_index]
        pred_out_x = pred_pos_x + decoder_nuclei_pos_embed
        for blk in self.decoder_blocks:
            pred_out_x = blk(pred_out_x)
        pred_out_x = self.decoder_norm(pred_out_x).squeeze(0)
        
        #############################################################################
        
        pred_class_logits = self.decoder_pred_class(pred_out_x)
        
        forward_passes = 5
        
        dropout_predictions = torch.empty((0, pred_out_x.shape[0], self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)).to(pred_out_x.device)
        with torch.no_grad():
            for forward_pass in range(forward_passes):
                dropout_pred = self.decoder_pred_class(self.dropout(pred_out_x))
                dropout_predictions = torch.cat((dropout_predictions, dropout_pred.unsqueeze(0)), dim = 0)      
        dropout_predictions = torch.mean(F.softmax(dropout_predictions, dim = -1), dim = 0)
        trans_uncertainty = -1.0 * torch.sum(dropout_predictions * torch.log(dropout_predictions + 1e-6), dim = -1)
        
        #print('trans_uncertainty = ', torch.mean(trans_uncertainty).detach().cpu().numpy())
        
        uncertainty_weights = mrcn_uncertainty / trans_uncertainty
        uncertainty_weights = uncertainty_weights / torch.sum(uncertainty_weights + 1e-6)
        
        class_logits_l1_loss = torch.mean(self.l1loss_noreduce(F.softmax(mrcn_class_logits, dim=-1), F.softmax(pred_class_logits, dim=-1)), dim = -1) * 10
        class_logits_l1_loss = torch.sum(uncertainty_weights * class_logits_l1_loss)
        
        return l1_loss, classification_loss, class_logits_l1_loss
