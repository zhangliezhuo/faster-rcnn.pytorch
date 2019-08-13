import numpy as np
import cv2
import torch

def convert_r_to_o(boxes_r):
    '''
    :param boxes_r:  shape (N, 5)
    :return:
    '''
    # print(boxes_r)
    assert boxes_r.size(1) == 5
    X = boxes_r[:, 0]
    Y = boxes_r[:, 1]
    W = boxes_r[:, 2]
    H = boxes_r[:, 3]
    theta = boxes_r[:, 4]

    xmin = X - W / 2
    xmax = X + W / 2
    ymin = Y - H / 2
    ymax = Y + H / 2
    '''
    nrx = (x-pointx)*cos(angle) - (y-pointy)*sin(angle)+pointx
    nry = (x-pointx)*sin(angle) + (y-pointy)*cos(angle)+pointy
    '''
    theta = theta * np.pi / 180
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x = xmin
    y = ymin
    x0 = (x - X) * cos_theta - (y - Y) * sin_theta + X
    y0 = (y - Y) * cos_theta + (x - X) * sin_theta + Y
    x = xmax
    y = ymin
    x1 = (x - X) * cos_theta - (y - Y) * sin_theta + X
    y1 = (y - Y) * cos_theta + (x - X) * sin_theta + Y
    x = xmax
    y = ymax
    x2 = (x - X) * cos_theta - (y - Y) * sin_theta + X
    y2 = (y - Y) * cos_theta + (x - X) * sin_theta + Y
    x = xmin
    y = ymax
    x3 = (x - X) * cos_theta - (y - Y) * sin_theta + X
    y3 = (y - Y) * cos_theta + (x - X) * sin_theta + Y
    # print(x0)
    rects = torch.cat((x0.view(-1, 1), y0.view(-1, 1), x1.view(-1, 1), y1.view(-1, 1)
                       , x2.view(-1, 1), y2.view(-1, 1), x3.view(-1, 1), y3.view(-1, 1)), 1)
    # print(rects)
    return rects

def convert_o_to_r(boxes_o):
    assert boxes_o.dim() == 3
    cls = None
    if boxes_o.dim() == 3:
        if boxes_o.size(2) == 9:
            cls = boxes_o[:, :, 8]
            boxes_o = boxes_o[:, :, :8]
        bboxes_r = []
        assert boxes_o.size(2) == 8
        for b in range(boxes_o.size(0)):
            box_o = boxes_o[b]
            _cls = cls[b] if cls is not None else None
            X = box_o[:, 0::2]
            Y = box_o[:, 1::2]
            assert X.size() == Y.size()
            rects = []
            for i in range(len(X)):
                xy = np.array(zip(X[i], Y[i]), dtype=np.int)
                rect = cv2.minAreaRect(xy)
                center_xy, size, theta = rect
                center_x, center_y = center_xy
                width, height = size
                # if width == 0:
                #     width = 1
                # if height == 0:
                #     height = 1
                if cls is None:
                    rects.append([center_x, center_y, width, height, theta])
                else:
                    rects.append([center_x, center_y, width, height, theta, _cls[i]])
            # print(np.unique(np.array(rects)[:, 2:4]))
            bboxes_r.append(rects)
    bboxes_r = torch.FloatTensor(bboxes_r)
    return bboxes_r


if __name__ == "__main__":
    boxes_o = [[
        [0.9, 4.3, 100.5, 20.6, 200.7, 150.1, 50, 150],
        [1, 4, 120, 10, 300, 400, 45, 600]
        ]]
    boxes_r = [[
        [100, 100, 10, 5, 30],
        [200, 200, 10, 100, 45]
    ]]
    # boxes_r = torch.FloatTensor(boxes_r)
    boxes_o = torch.FloatTensor(boxes_o)
    boxes_r = convert_o_to_r(boxes_o)
    # boxes_o = convert_r_to_o(boxes_r)
    # print(boxes_o)
    # boxes_r = convert_o_to_r(boxes_o)
    print(boxes_r)