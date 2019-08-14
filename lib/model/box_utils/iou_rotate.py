# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
# from libs.box_utils.coordinate_convert import *
from model.box_utils.rbbox_overlaps import rbbx_overlaps
from model.box_utils.iou_cpu import get_iou_matrix


def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):
    '''

    :param boxes_list1:[N, 8] tensor
    :param boxes_list2: [M, 8] tensor
    :return:
    '''

    boxes1 = boxes1.cpu().numpy()
    boxes2 = boxes2.cpu().numpy()

    if use_gpu:
        iou_matrix = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        iou_matrix = get_iou_matrix(boxes1, boxes2)

    iou_matrix = np.reshape(iou_matrix, (boxes1.shape[0], boxes2.shape[0]))
    # ind_tmp = torch.from_numpy((boxes2[:, 2] == 1) & (boxes2[:, 3] == 1) & (boxes2[:, 1] == 0) & (boxes2[:, 0] == 0) & (boxes2[:, 4] == 0))

    # ind1 = torch.from_numpy((boxes1[:, 2] == 0) | (boxes1[:, 3] == 0))
    # ind1 = torch.nonzero(ind1)
    # ind2 = torch.from_numpy((boxes2[:, 2] == 0) | (boxes2[:, 3] == 0))
    # ind2 = torch.nonzero(ind2)

    # iou_matrix[ind1, :] = 0
    # iou_matrix[:, ind2] = 0 

    return torch.from_numpy(iou_matrix)



if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    boxes1 = torch.from_numpy(np.array([[45.4171, 8.0342, 16.0685, 90.8343, -90.000]], np.float32))

    boxes2 = torch.from_numpy(np.array([[0, 0, 0, 0, 0]], np.float32))

    print(iou_rotate_calculate(boxes1, boxes2))

    # start = time.time()
    # with tf.Session() as sess:
    #     ious = iou_rotate_calculate1(boxes1, boxes2, use_gpu=False)
    #     print(sess.run(ious))
    #     print('{}s'.format(time.time() - start))

    # start = time.time()
    # for _ in range(10):
    #     ious = rbbox_overlaps.rbbx_overlaps(boxes1, boxes2)
    # print('{}s'.format(time.time() - start))
    # print(ious)

    # print(ovr)



