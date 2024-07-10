# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

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
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, features, proposals, targets=None, is_source = True):

        if self.training and is_source:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        box_logits = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(box_logits)

        if not self.training:
            result, keep_list, keep_list2 = self.post_processor((class_logits, box_regression), proposals)
            return box_logits, result, {}

        if not is_source:
            result, keep_list, keep_list2 = self.post_processor((class_logits, box_regression), proposals)
            box_logits_tgt = torch.cat([box_logits[keep_list_class] for keep_list_class in keep_list], dim = 0)
            if keep_list2 is not None:
                box_logits_tgt = box_logits_tgt[keep_list2]
            return box_logits_tgt, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            box_logits,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        )
        

    def forward_class_logits(self, features, proposals):
    
        box_logits = self.feature_extractor(features, proposals)
        class_logits, box_regression = self.predictor(box_logits)
        
        return class_logits
        

    def forward_class_logits_uncertainty(self, features, proposals):
    
        box_logits = self.feature_extractor(features, proposals)
        box_logits = self.dropout(box_logits)
        class_logits, box_regression = self.predictor(box_logits)
        
        return class_logits
        
def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
