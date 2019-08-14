import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer, _ProposalTargetLayer_r
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_proposal_target_r = _ProposalTargetLayer_r(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_boxes_r=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        gt_boxes_r = gt_boxes_r.data if gt_boxes_r is not None else None

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox,\
         rois_r, rpn_loss_cls_r, rpn_loss_bbox_r = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, gt_boxes_r)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            roi_data_r = self.RCNN_proposal_target_r(rois_r, gt_boxes_r, num_boxes)
            rois_r, rois_r_label, rois_r_target, rois_r_inside_ws, rois_r_outside_ws = roi_data_r

            rois_r_label = Variable(rois_r_label.view(-1).long())
            rois_r_target = Variable(rois_r_target.view(-1, rois_r_target.size(2)))
            rois_r_inside_ws = Variable(rois_r_inside_ws.view(-1, rois_r_inside_ws.size(2)))
            rois_r_outside_ws = Variable(rois_r_outside_ws.view(-1, rois_r_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rois_r_label = None
            rois_r_target = None
            rois_r_inside_ws = None
            rois_r_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            rpn_loss_cls_r = 0
            rpn_loss_bbox_r = 0
        # print(rois)
        # print(rois_r.size())
        from model.utils.bbox_convert import convert_r_to_h
        rois_rh = convert_r_to_h(rois_r)
        # print(rois_rh.size())
        rois = Variable(rois)
        rois_r = Variable(rois_r)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            pooled_feat_r = self.RCNN_roi_align(base_feat, rois_rh.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
            pooled_feat_r = self.RCNN_roi_pool(base_feat, rois_rh.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        pooled_feat_r = self._head_to_tail(pooled_feat_r)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_r_pred = self.RCNN_bbox_r_pred(pooled_feat_r)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            bbox_r_pred_view = bbox_r_pred.view(bbox_r_pred.size(0), int(bbox_r_pred.size(1) / 5), 5)
            bbox_r_pred_select = torch.gather(bbox_r_pred_view, 1, rois_r_label.view(rois_r_label.size(0), 1, 1).expand(rois_r_label.size(0), 1, 5))
            bbox_r_pred = bbox_r_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        # compute r object classification probability
        cls_r_score = self.RCNN_cls_score(pooled_feat_r)
        cls_r_prob = F.softmax(cls_r_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        RCNN_loss_cls_r = 0
        RCNN_loss_bbox_r = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

            # classification loss rotated
            RCNN_loss_cls_r = F.cross_entropy(cls_r_score, rois_r_label)
            # bounding box regression L1 loss
            # print(rois_r_inside_ws)
            RCNN_loss_bbox_r = _smooth_l1_loss(bbox_r_pred, rois_r_target, rois_r_inside_ws, rois_r_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        cls_r_prob = cls_r_prob.view(batch_size, rois_r.size(1), -1)
        bbox_r_pred = bbox_r_pred.view(batch_size, rois_r.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
        rois_r, cls_r_prob, bbox_r_pred, rpn_loss_cls_r, rpn_loss_bbox_r, RCNN_loss_cls_r, RCNN_loss_bbox_r, rois_r_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_r_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score_r, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_r_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
