# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
#from .generalized_rcnn_grl import GeneralizedRCNN_GRL
#from .generalized_rcnn_mrcn import GeneralizedRCNN_MRCN
#from .generalized_rcnn_filler_wogrl import GeneralizedRCNN_filler_wogrl
#from .generalized_rcnn_nuclei_filler import GeneralizedRCNN_nuclei_filler
from .generalized_rcnn_final import GeneralizedRCNN_final

#_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN, "GeneralizedRCNN_GRL": GeneralizedRCNN_GRL, "GeneralizedRCNN_MRCN": GeneralizedRCNN_MRCN, "GeneralizedRCNN_filler_wogrl": GeneralizedRCNN_filler_wogrl, "GeneralizedRCNN_nuclei_filler": GeneralizedRCNN_nuclei_filler}
_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN_final": GeneralizedRCNN_final}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
