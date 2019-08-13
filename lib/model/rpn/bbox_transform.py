# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb
from model.box_utils.iou_rotate import iou_rotate_calculate

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets
def bbox_r_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2]
        ex_heights = ex_rois[:, 3]
        ex_ctr_x = ex_rois[:, 0]
        ex_ctr_y = ex_rois[:, 1]
        ex_angles = ex_rois[:, 4]

        gt_widths = gt_rois[:, :, 2].clone()
        gt_heights = gt_rois[:, :, 3].clone()
        gt_ctr_x = gt_rois[:, :, 0]
        gt_ctr_y = gt_rois[:, :, 1]
        gt_angles = gt_rois[:, :, 4]
        gt_widths[gt_widths==0] = 1
        gt_heights[gt_heights==0] = 1
        ex_widths[ex_widths==0] = 1
        ex_heights[ex_heights==0] = 1
        # print(gt_ctr_x)
        # print(ex_ctr_x.contiguous().view(1, -1).expand_as(gt_ctr_x))
        # print((gt_ctr_x - ex_ctr_x.contiguous().view(1, -1).expand_as(gt_ctr_x)))
        targets_dx = (gt_ctr_x - ex_ctr_x.contiguous().view(1, -1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.contiguous().view(1, -1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.contiguous().view(1, -1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.contiguous().view(1, -1).expand_as(gt_heights))
        targets_da = (gt_angles - ex_angles.contiguous().view(1, -1).expand_as(gt_angles)) * np.pi / 180

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2]
        ex_heights = ex_rois[:, :, 3]
        ex_ctr_x = ex_rois[:, :, 0]
        ex_ctr_y = ex_rois[:, :, 1]
        ex_angles = ex_rois[:, :, 4]

        gt_widths = gt_rois[:, :, 2].clone()
        gt_heights = gt_rois[:, :, 3].clone()
        gt_ctr_x = gt_rois[:, :, 0]
        gt_ctr_y = gt_rois[:, :, 1]
        gt_angles = gt_rois[:, :, 4]

        gt_widths[gt_widths==0] = 1
        gt_heights[gt_heights==0] = 1
        ex_widths[ex_widths==0] = 1
        ex_heights[ex_heights==0] = 1

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
        targets_da = (gt_angles - ex_angles) * np.pi / 180
    else:
        raise ValueError('ex_roi input dimension is not correct.')
    # print("targets_dx,", np.unique(targets_dx.cpu().numpy()))
    # print("targets_dy,", np.unique(targets_dy.cpu().numpy()))
    # print("targets_dw,", np.unique(targets_dw.cpu().numpy()))
    # print("targets_dh,", np.unique(targets_dh.cpu().numpy()))
    # print("targets_da,", np.unique(targets_da.cpu().numpy()))

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh, targets_da), 2)
    # print("target,", np.unique(targets.cpu().numpy()))
    return targets

def bbox_transform_batch(ex_rois, gt_rois):
    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def bbox_r_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2]
    heights = boxes[:, :, 3]
    ctr_x = boxes[:, :, 0]
    ctr_y = boxes[:, :, 1]
    angles = boxes[:, :, 4]

    dx = deltas[:, :, 0::5]
    dy = deltas[:, :, 1::5]
    dw = deltas[:, :, 2::5]
    dh = deltas[:, :, 3::5]
    da = deltas[:, :, 4::5]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_a = torch.exp(da) * angles.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x
    pred_boxes[:, :, 0::5] = pred_ctr_x
    # y
    pred_boxes[:, :, 1::5] = pred_ctr_y
    # w
    pred_boxes[:, :, 2::5] = pred_w
    # h
    pred_boxes[:, :, 3::5] = pred_h
    # a
    pred_boxes[:, :, 4::5] = pred_a


    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes

def clip_boxes_r(boxes, im_shape, batch_size):
    from model.utils.bbox_convert import convert_r_to_o
    import cv2
    for i in range(batch_size):
        boxes_o = convert_r_to_o(boxes[i])

        inside_boxes_index = (
            (boxes_o[:, 0] >= 0) &
            (boxes_o[:, 1] >= 0) &
            (boxes_o[:, 2] >= 0) &
            (boxes_o[:, 3] >= 0) & 
            (boxes_o[:, 4] >= 0) &
            (boxes_o[:, 5] >= 0) &
            (boxes_o[:, 6] >= 0) &
            (boxes_o[:, 7] >= 0) & 
            (boxes_o[:, 0] < im_shape[i, 1]) &
            (boxes_o[:, 1] < im_shape[i, 0]) &
            (boxes_o[:, 2] < im_shape[i, 1]) &
            (boxes_o[:, 3] < im_shape[i, 0]) & 
            (boxes_o[:, 4] < im_shape[i, 1]) &
            (boxes_o[:, 5] < im_shape[i, 0]) &
            (boxes_o[:, 6] < im_shape[i, 1]) &
            (boxes_o[:, 7] < im_shape[i, 0])
        )
        border_boxes_index = torch.nonzero(inside_boxes_index).view(-1)

        # print(boxes_o)

        boxes_o[:, 0::2].clamp_(0, im_shape[i, 1]-1)
        boxes_o[:, 1::2].clamp_(0, im_shape[i, 0]-1)
        # print(boxes_o)

        for j in range(boxes_o.size(0)):
            if j in border_boxes_index:
                continue
            box = boxes_o[j].view(-1, 2)
            # print(box)
            box = box.cpu().numpy()
            rect = cv2.minAreaRect(box)
            x, y = rect[0]
            w, h = rect[1]
            a = rect[2]
            if w <= 0 or h <= 0:
                boxes[i, j] = torch.FloatTensor([0, 0, 0, 0 ,0])
            else:
                boxes[i, j] = torch.FloatTensor([x, y, w, h ,a])

    return boxes

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


def bbox_r_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 5) ndarray of float
    gt_boxes: (b, K, 6) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 5).expand(batch_size, N, 5).contiguous()
        gt_boxes = gt_boxes[:,:,:5].contiguous()
        overlaps = torch.Tensor(batch_size, N, K).fill_(0).type_as(gt_boxes)
        for b in range(gt_boxes.size(0)):
            overlap = iou_rotate_calculate(anchors[b], gt_boxes[b])
            overlaps[b] = overlap

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 5:
            anchors = anchors[:,:,:5].contiguous()
        else:
            anchors = anchors[:,:,1:].contiguous()

        gt_boxes = gt_boxes[:,:,:5].contiguous()
        overlaps = torch.Tensor(batch_size, N, K).fill_(0).type_as(gt_boxes)
        for b in range(gt_boxes.size(0)):
            overlap = iou_rotate_calculate(anchors[b], gt_boxes[b])
            overlaps[b] = overlap

    else:
        raise ValueError('anchors input dimension is not correct.')
    return overlaps