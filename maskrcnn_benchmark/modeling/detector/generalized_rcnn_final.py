# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import cv2
import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from maskrcnn_benchmark.modeling.backbone.backbone import build_panoptic_fpn, build_panoptic_fpn_filler
from ..domain_adaption.domain_adaption_head import build_domain_adaption_head
from ..domain_adaption.domain_adaption_head import build_mi_max_head

from ..backbone.nuclei_correlation import Transformer_nuclei_filler

from thop import profile


class GeneralizedRCNN_final(nn.Module):

    def __init__(self, cfg):
        super(GeneralizedRCNN_final, self).__init__()
        
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.cfg = cfg
        
        self.pan_fpn_filler = build_panoptic_fpn_filler(cfg)
        self.filler_img_da = build_domain_adaption_head(cfg, modal='filler_image', filler_is_source=True)
        self.t_filler_img_da = build_domain_adaption_head(cfg, modal='filler_image', filler_is_source=False)
        self.criterionIdt = nn.L1Loss()
        
        self.nuclei_filler = Transformer_nuclei_filler(cfg)

    def forward(self, images, images_inpaint=None, t_images=None, t_images_inpaint=None, targets=None, t_targets=None, grl_alpha = 0.1):
      
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.training:
            losses = {}

            for proposal_loss_name in proposal_losses:
                proposal_losses[proposal_loss_name] = proposal_losses[proposal_loss_name]

            losses.update(proposal_losses)


            # Instance level adaptation

            x, result, detector_losses, s_ins_logits_da, s_roi_features \
                = self.roi_heads(features, proposals,targets, is_source=True)
                
            for det_loss_name in detector_losses:
                detector_losses[det_loss_name] = detector_losses[det_loss_name]

            losses.update(detector_losses)
            
            
            t_images = to_image_list(t_images)
            t_features = self.backbone(t_images.tensors)
            t_proposals, t_proposal_losses = self.rpn(t_images, t_features, t_targets)
            _, _, t_detector_losses, _, _ = self.roi_heads(t_features, t_proposals, t_targets, is_source=True)
            
            losses.update({'loss_objectness_t': t_proposal_losses['loss_objectness']})
            losses.update({'loss_rpn_box_reg_t': t_proposal_losses['loss_rpn_box_reg']})
            losses.update({'loss_box_reg_t': t_detector_losses['loss_box_reg']})
            losses.update({'loss_mask_t': t_detector_losses['loss_mask']})
            
            #######################################################################################
            
            images_inpaint = to_image_list(images_inpaint)
            features_inpaint = self.backbone(images_inpaint.tensors)
            filled_image = self.pan_fpn_filler(features_inpaint, grl_alpha) + images_inpaint.tensors
            
            fill_loss = self.criterionIdt(filled_image, images.tensors) * 10
            losses.update({'fill_loss_s': fill_loss})
            da_img_losses = self.filler_img_da(images.tensors, filled_image, 1.0)
            losses.update(da_img_losses)
            
            t_images_inpaint = to_image_list(t_images_inpaint)
            t_features_inpaint = self.backbone(t_images_inpaint.tensors)
            t_filled_image = self.pan_fpn_filler(t_features_inpaint, grl_alpha) + t_images_inpaint.tensors
            
            t_fill_loss = self.criterionIdt(t_filled_image, t_images.tensors) * 10
            losses.update({'fill_loss_t': t_fill_loss})
            t_da_img_losses = self.t_filler_img_da(t_images.tensors, t_filled_image, 1.0)
            losses.update(t_da_img_losses)
            
            #######################################################################################
            
            forward_passes = 5
                
            s_mrcn_class_logits = self.roi_heads.forward_class_logits(features, targets)
            t_mrcn_class_logits = self.roi_heads.forward_class_logits(t_features, t_targets)
            
            s_dropout_predictions = torch.empty((0, s_mrcn_class_logits.shape[0], self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)).to(s_mrcn_class_logits.device)
            t_dropout_predictions = torch.empty((0, t_mrcn_class_logits.shape[0], self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES)).to(t_mrcn_class_logits.device)
            
            with torch.no_grad():
                for forward_pass in range(forward_passes):
                    s_dropout = self.roi_heads.forward_class_logits_uncertainty(features, targets)
                    s_dropout_predictions = torch.cat((s_dropout_predictions, s_dropout.unsqueeze(0)), dim = 0)
                    t_dropout = self.roi_heads.forward_class_logits_uncertainty(t_features, t_targets)
                    t_dropout_predictions = torch.cat((t_dropout_predictions, t_dropout.unsqueeze(0)), dim = 0)
               
            s_dropout_predictions = torch.mean(F.softmax(s_dropout_predictions, dim = -1), dim = 0)
            t_dropout_predictions = torch.mean(F.softmax(t_dropout_predictions, dim = -1), dim = 0)
            s_mrcn_uncertainty = -1.0 * torch.sum(s_dropout_predictions * torch.log(s_dropout_predictions + 1e-6), dim = -1)
            t_mrcn_uncertainty = -1.0 * torch.sum(t_dropout_predictions * torch.log(t_dropout_predictions + 1e-6), dim = -1)

            nuclei_correlation_loss, nuclei_correlation_class_loss, class_logits_l1_loss = self.nuclei_filler(features, targets, s_mrcn_class_logits, s_mrcn_uncertainty, grl_alpha)
            losses.update({'nuc_cor_loss_s': nuclei_correlation_loss})
            losses.update({'nuc_cor_class_loss_s': nuclei_correlation_class_loss})
            losses.update({'nuc_cor_logits_l1_loss_s': class_logits_l1_loss})
            
            t_nuclei_correlation_loss, t_nuclei_correlation_class_loss, t_class_logits_l1_loss = self.nuclei_filler(t_features, t_targets, t_mrcn_class_logits, t_mrcn_uncertainty, grl_alpha)
            losses.update({'nuc_cor_loss_t': t_nuclei_correlation_loss})
            losses.update({'nuc_cor_logits_l1_loss_t': t_class_logits_l1_loss})
            
            return losses, images_inpaint.tensors, filled_image, images.tensors, t_images_inpaint.tensors, t_filled_image, t_images.tensors

        if self.roi_heads:
            x, result, detector_losses, s_ins_logits_da, s_roi_features \
                = self.roi_heads(features, proposals, targets, is_source = True)
        else:
            # RPN-only models don't have roi_heads
            result = proposals

        return result
        
        
    def forward_infer(self, images, targets):
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        #proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses, s_ins_logits_da, s_roi_features \
                = self.roi_heads(features, targets, None, is_source = True)
        else:
            result = proposals
        return result
        
        
        
