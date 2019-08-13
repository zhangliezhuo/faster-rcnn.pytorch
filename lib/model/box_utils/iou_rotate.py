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


    boxes1[boxes1==0]=0.01
    boxes2[boxes2==0]=0.01
    if use_gpu:
        iou_matrix = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        iou_matrix = get_iou_matrix(boxes1, boxes2)

    iou_matrix = np.reshape(iou_matrix, (boxes1.shape[0], boxes2.shape[0]))

    return torch.from_numpy(iou_matrix)



if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    boxes1 = torch.from_numpy(np.array([[0, 0, 0, 0, 0]], np.float32))

    boxes2 = torch.from_numpy(np.array([[103.5000,  55.5000,  88.0000, 176.0000, -90.0000]], np.float32))

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



