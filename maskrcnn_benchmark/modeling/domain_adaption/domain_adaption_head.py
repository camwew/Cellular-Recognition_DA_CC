

from ..domain_adaption.DA import _ImageDA
from ..domain_adaption.DA import _filler_ImageDA
from ..domain_adaption.DA import _SemsegDA
from ..domain_adaption.DA import _InstanceDA
from ..domain_adaption.DA import _SemsegDARes
from ..domain_adaption.DA import _InstanceReconstructor
from ..domain_adaption.DA import grad_reverse, grad_cut


import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from ..domain_adaption.DA import InsMiEstimatorFC
from .LabelResizeLayer import ImageLabelResizeLayer



thres_low = 0.5
thres_up = 1.5

# Original implementation for source label: 1, for target label: 0
# If reverse like GAN, then source label: 0, target label: 1
def build_source_label(cfg):
    if cfg.MODEL.DOMAIN_ADAPTION.REVERSE_LABEL:
        np_need_backprop = np.zeros((1,), dtype=np.float32)
    else:
        np_need_backprop = np.ones((1,), dtype=np.float32)
    need_backprop = torch.tensor(np_need_backprop)
    device = torch.device(cfg.MODEL.DEVICE)
    if device is 'cuda':
        need_backprop = need_backprop.cuda()
    need_backprop = Variable(need_backprop)

    return need_backprop


def build_target_label(cfg):
    if cfg.MODEL.DOMAIN_ADAPTION.REVERSE_LABEL:
        np_need_backprop = np.ones((1,), dtype=np.float32)
    else:
        np_need_backprop = np.zeros((1,), dtype=np.float32)
    need_backprop = torch.tensor(np_need_backprop)
    device = torch.device(cfg.MODEL.DEVICE)
    if device is 'cuda':
        need_backprop = need_backprop.cuda()
    need_backprop = Variable(need_backprop)

    return need_backprop

def bce_loss(y_pred, y_label, weight=None):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCELoss(weight=weight)(y_pred, y_truth_tensor)


class ImageDomainAdaption(torch.nn.ModuleDict):
    def __init__(self, cfg):
        super(ImageDomainAdaption, self).__init__()
        self.ImageDA = _ImageDA(cfg, feat_out=cfg.MODEL.DOMAIN_ADAPTION.FEATURE_CHANNEL)
        self.cfg = cfg
        self.LabelResizeLayer = ImageLabelResizeLayer()


    def forward(self, source, target=None, grl_alpha = 1):
    
        grl_alpha = grl_alpha * self.cfg.MODEL.GLOBAL_FEA_GRL

        assert len(source) == len(target)
        source_da_label = build_source_label(self.cfg)
        target_da_label = build_target_label(self.cfg)

        # down sample
        s_fpn_feature = source[-2]
        t_fpn_feature = target[-2]
        da_patch_scale = s_fpn_feature.size()[-1]
        for i in range (3, len(source) + 1):
            s_fpn_feature = F.adaptive_avg_pool2d(source[-i], output_size =da_patch_scale) + s_fpn_feature
            t_fpn_feature = F.adaptive_avg_pool2d(target[-i], output_size = da_patch_scale) + t_fpn_feature

        source_score, _ = self.ImageDA(grad_reverse(s_fpn_feature, grl_alpha=grl_alpha))

        source_label = self.LabelResizeLayer(source_score, source_da_label)

        source_prob = F.log_softmax(source_score, dim=1)
        DA_img_loss_s = F.nll_loss(source_prob, source_label)

        target_score, _ = self.ImageDA(grad_reverse(t_fpn_feature, grl_alpha = grl_alpha))

        target_label = self.LabelResizeLayer(target_score, target_da_label)

        target_prob = F.log_softmax(target_score, dim=1)
        DA_img_loss_t = F.nll_loss(target_prob, target_label)

        source_prob_img = torch.exp(source_prob)
        task_specific_weight = (1 - source_prob_img) / source_prob_img

        task_specific_weight_img_raw = task_specific_weight.contiguous().view(1, -1).mean(dim=1).detach()
        task_specific_weight_img = torch.clamp(task_specific_weight_img_raw,  thres_low, thres_up)


        if self.cfg.MODEL.DOMAIN_ADAPTION.LOSS_AVG:
            DA_img_losses = (DA_img_loss_s + DA_img_loss_t) /2
            DA_img_loss = {
                "loss_da_fpn": DA_img_losses,
            }
        else:
            DA_img_loss = {
                "loss_da_fpn_s": DA_img_loss_s,
                "loss_da_fpn_t": DA_img_loss_t,
            }

        return DA_img_loss, task_specific_weight_img
        
        
class filler_ImageDomainAdaption(torch.nn.ModuleDict):
    def __init__(self, cfg, filler_is_source):
        super(filler_ImageDomainAdaption, self).__init__()
        self.ImageDA = _filler_ImageDA(cfg)
        self.cfg = cfg
        self.filler_is_source = filler_is_source
        self.LabelResizeLayer = ImageLabelResizeLayer()


    def forward(self, source, target, grl_alpha = 1):

        assert len(source) == len(target)
        source_da_label = build_source_label(self.cfg)
        target_da_label = build_target_label(self.cfg)

        # down sample
        source_score = self.ImageDA(grad_reverse(source, grl_alpha=grl_alpha))

        source_label = self.LabelResizeLayer(source_score, source_da_label)

        source_prob = F.log_softmax(source_score, dim=1)
        DA_img_loss_s = F.nll_loss(source_prob, source_label)

        target_score = self.ImageDA(grad_reverse(target, grl_alpha = grl_alpha))

        target_label = self.LabelResizeLayer(target_score, target_da_label)

        target_prob = F.log_softmax(target_score, dim=1)
        DA_img_loss_t = F.nll_loss(target_prob, target_label)

        DA_img_losses = (DA_img_loss_s + DA_img_loss_t) /2
        if self.filler_is_source:
            DA_img_loss = {
                "loss_image_adv_s": DA_img_losses,
            }
        else:
            DA_img_loss = {
                "loss_image_adv_t": DA_img_losses,
            }

        return DA_img_loss



class SemanticDomainAdaption(torch.nn.ModuleDict):
    def __init__(self, cfg, input_type='global'):
        super(SemanticDomainAdaption, self).__init__()
        # self.SemsegDA = _SemsegDA()
        if input_type == 'global':
            sem_type = cfg.MODEL.DOMAIN_ADAPTION.SEM_TYPE
        elif input_type == 'ins_mask':
            sem_type = 'CNN4'
        else:
            raise ValueError("Incorrect SemDA type!!!")
        assert sem_type == 'CNN4' or sem_type == 'RES4', \
            "semantic level model name should be in {CNN6, RES4}"
        if sem_type == 'CNN4':
            self.SemsegDA = _SemsegDA(cfg, feat_out=cfg.MODEL.DOMAIN_ADAPTION.FEATURE_CHANNEL)
        else:
            self.SemsegDA = _SemsegDARes(cfg, feat_out=cfg.MODEL.DOMAIN_ADAPTION.FEATURE_CHANNEL)
        self.input_type = input_type
        self.cfg = cfg
        self.LabelResizeLayer = ImageLabelResizeLayer()


    def forward(self, source, target=None, grl_alpha = 1):
        assert len(source) == len(target)
        source_da_label = build_source_label(self.cfg)
        target_da_label = build_target_label(self.cfg)

        s_fpn_feature = source
        t_fpn_feature = target

        source_score, _ = self.SemsegDA(grad_reverse(s_fpn_feature, grl_alpha = grl_alpha))

        source_label = self.LabelResizeLayer(source_score, source_da_label)
        source_prob = F.log_softmax(source_score, dim=1)
        DA_img_loss_s = F.nll_loss(source_prob, source_label)

        source_prob_sem = torch.exp(source_prob)
        task_specific_weight = (1- source_prob_sem) / source_prob_sem

        task_specific_weight_sem_raw = task_specific_weight.contiguous().view(1, -1).mean(dim=1).detach()
        task_specific_weight_sem = torch.clamp(task_specific_weight_sem_raw,  thres_low, thres_up)

        target_score, _ = self.SemsegDA(grad_reverse(t_fpn_feature, grl_alpha = grl_alpha))

        target_label = self.LabelResizeLayer(target_score, target_da_label)
        target_prob = F.log_softmax(target_score, dim=1)
        DA_img_loss_t = F.nll_loss(target_prob, target_label)

        if self.cfg.MODEL.DOMAIN_ADAPTION.LOSS_AVG:
            DA_sem_losses = (DA_img_loss_s + DA_img_loss_t) /2
            if self.input_type == 'global':
                DA_sem_loss = {"loss_da_sem": DA_sem_losses,}
            elif self.input_type == 'ins_mask':
                DA_sem_loss = {"loss_da_mask_sem": DA_sem_losses,}
        else:
            if self.input_type == 'global':
                DA_sem_loss = {"loss_da_sem_s": DA_img_loss_s,"loss_da_sem_t": DA_img_loss_t,}
            elif self.input_type == 'ins_mask':
                DA_sem_loss = {"loss_da_mask_sem_s": DA_img_loss_s,"loss_da_mask_sem_t": DA_img_loss_t,}

        return DA_sem_loss, task_specific_weight_sem



class InstanceDomainAdaption(torch.nn.ModuleDict):
    def __init__(self, cfg, category):
        super(InstanceDomainAdaption, self).__init__()

        self.cfg = cfg
        self.category = category
        ins_da_input_channel = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.InsDA = _InstanceDA(ins_da_input_channel)

    def forward(self, source, target, grl_alpha = 1):
        # assert len(source) == len(target)
        if source.shape[0] == 0 or target.shape[0] == 0:
            DA_ins_loss = {}
            return DA_ins_loss, 1.0

        if self.cfg.MODEL.DOMAIN_ADAPTION.REVERSE_LABEL:
            source_domain_label = 0
            target_domain_label = 1
        else:
            source_domain_label = 1
            target_domain_label = 0

        s_ins_feature = source
        t_ins_feature = target


        source_score = self.InsDA(grad_reverse(s_ins_feature, grl_alpha = grl_alpha))

        DA_ins_loss_s = bce_loss(source_score, source_domain_label)

        task_specific_weight_raw = (1-source_score) / source_score

        task_specific_weight_ins_raw = task_specific_weight_raw.contiguous().view(1, -1).mean(dim=1).detach()
        task_specific_weight_ins = torch.clamp(task_specific_weight_ins_raw, thres_low, thres_up)

        target_score = self.InsDA(grad_reverse(t_ins_feature, grl_alpha = grl_alpha))

        DA_ins_loss_t = bce_loss(target_score, target_domain_label)


        if self.cfg.MODEL.DOMAIN_ADAPTION.LOSS_AVG:
            DA_ins_losses = (DA_ins_loss_s + DA_ins_loss_t) /2
            DA_ins_loss = {
                "loss_da_ins_" + str(self.category): DA_ins_losses,
            }
        else:

            DA_ins_loss = {
                "loss_da_ins_s_" + str(self.category): DA_ins_loss_s,
                "loss_da_ins_t_" + str(self.category): DA_ins_loss_t,
            }

        return DA_ins_loss, task_specific_weight_ins
        
        
class InstanceDomainClassifier(torch.nn.ModuleDict):
    def __init__(self, cfg):
        super(InstanceDomainClassifier, self).__init__()

        self.cfg = cfg
        ins_da_input_channel = int(cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM / 4)
        self.InsDA = _InstanceDA(ins_da_input_channel)

    def forward(self, source, target, grl_alpha = 1):
        # assert len(source) == len(target)
        if source.shape[0] == 0 or target.shape[0] == 0:
            DA_ins_loss = {}
            return DA_ins_loss

        if self.cfg.MODEL.DOMAIN_ADAPTION.REVERSE_LABEL:
            source_domain_label = 0
            target_domain_label = 1
        else:
            source_domain_label = 1
            target_domain_label = 0

        s_ins_feature = source
        t_ins_feature = target

        source_score = self.InsDA(grad_cut(s_ins_feature, gcl_alpha = grl_alpha * 10))
        DA_ins_loss_s = bce_loss(source_score, source_domain_label)
        target_score = self.InsDA(grad_cut(t_ins_feature, gcl_alpha = grl_alpha * 10))
        DA_ins_loss_t = bce_loss(target_score, target_domain_label)


        if self.cfg.MODEL.DOMAIN_ADAPTION.LOSS_AVG:
            DA_ins_losses = (DA_ins_loss_s + DA_ins_loss_t) /2
            DA_ins_loss = {
                "loss_da_ins_style_classify": DA_ins_losses,
            }
        else:

            DA_ins_loss = {
                "loss_da_ins_s_style_classify": DA_ins_loss_s,
                "loss_da_ins_t_style_classify": DA_ins_loss_t,
            }

        return DA_ins_loss
        
        
class InstanceDisentangle(torch.nn.ModuleDict):
    def __init__(self, cfg):
        super(InstanceDisentangle, self).__init__()

        self.cfg = cfg
        ins_da_input_channel = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        ins_da_style_input_channel = int(ins_da_input_channel / 4)
        self.InsRecon = _InstanceReconstructor(cfg, ins_da_input_channel, ins_da_style_input_channel)

    def forward(self, s_cat_f, s_regre_f, s_style_f, s_roi_box_f, t_cat_f, t_regre_f, t_style_f, t_roi_box_f, grl_alpha = 1):
        # assert len(source) == len(target)
        if s_cat_f.shape[0] == 0 or t_cat_f.shape[0] == 0:
            recon_ins_loss = {}
            return recon_ins_loss

        if self.cfg.MODEL.DOMAIN_ADAPTION.REVERSE_LABEL:
            source_domain_label = 0
            target_domain_label = 1
        else:
            source_domain_label = 1
            target_domain_label = 0

        s_ins_feature = source
        t_ins_feature = target

        source_score = self.InsDA(grad_cut(s_ins_feature, gcl_alpha = grl_alpha * 10))
        DA_ins_loss_s = bce_loss(source_score, source_domain_label)
        target_score = self.InsDA(grad_cut(t_ins_feature, gcl_alpha = grl_alpha * 10))
        DA_ins_loss_t = bce_loss(target_score, target_domain_label)


        if self.cfg.MODEL.DOMAIN_ADAPTION.LOSS_AVG:
            DA_ins_losses = (DA_ins_loss_s + DA_ins_loss_t) /2
            DA_ins_loss = {
                "loss_da_ins_style_classify": DA_ins_losses,
            }
        else:

            DA_ins_loss = {
                "loss_da_ins_s_style_classify": DA_ins_loss_s,
                "loss_da_ins_t_style_classify": DA_ins_loss_t,
            }

        return DA_ins_loss



class InforMaxFC(torch.nn.ModuleDict):
    def __init__(self, cfg):
        super(InforMaxFC, self).__init__()

        self.cfg = cfg

        mi_input_channel = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM * 2
        self.MIEstimator = InsMiEstimatorFC(in_size=mi_input_channel, interm_size=256)

        self.beta = 0.1

    def forward(self, source, target):

        target_fake = torch.cat((target[1:], target[0].unsqueeze(0)), dim=0)

        x = torch.cat([source, target], dim=1)

        x_fake = torch.cat([source, target_fake], dim=1)

        Ej = -F.softplus(-self.MIEstimator(x)).mean()
        Em = F.softplus(self.MIEstimator(x_fake)).mean()

        informax_loss = (Em - Ej) * self.beta

        DA_informax_loss = {
            "loss_da_informax": informax_loss
        }


        return DA_informax_loss




def build_domain_adaption_head(cfg, modal, category = None, semantic_type='global', filler_is_source=True):
    #assert modal == 'semantic' or modal == 'instance'or modal == 'image', \
    #"domain adaption modal should be in {semantic, instance}"

    if modal == 'semantic':
        da_head = SemanticDomainAdaption(cfg, semantic_type)
    elif modal == 'image':
        da_head = ImageDomainAdaption(cfg)
    elif modal == 'filler_image':
        da_head = filler_ImageDomainAdaption(cfg, filler_is_source)
    elif modal == 'instance':
        da_head = InstanceDomainAdaption(cfg, category)
    elif modal == 'instance_style':
        da_head = InstanceDomainClassifier(cfg)
    else:
        raise ValueError("Incorrect DA model type!!!")


    return da_head



def build_mi_max_head(cfg):

    mi_max_head = InforMaxFC(cfg)

    return mi_max_head












