import argparse

import cv2
import numpy as np


def strtobool(val):
    val = str(val).lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    raise ValueError('invalid truth value %r' % (val,))


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


def processed_image(img, scale):
    """按比例缩放图片并做归一化，返回 [C, H, W] 的浮点数组"""
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    image = np.array(img_resized).astype(np.float32)
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    # 归一化到 [-1, 1] 附近
    image = (image - 127.5) / 128
    return image


def convert_to_square(box):
    """将检测框转换成包含它的最大正方形，避免裁剪时丢失信息

    参数：
      box: 检测框 [n, 5]
    返回：
      转换后的正方形框 [n, 5]
    """
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    max_side = np.maximum(w, h)

    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box


def pad(bboxes, w, h):
    """处理超出图像边界的框

    返回：
      dy, dx   : 框在裁剪图内的左上角坐标（相对框左上角）
      edy, edx : 框在裁剪图内的右下角坐标（相对框左上角）
      y, x     : 框在原图内的左上角坐标
      ey, ex   : 框在原图内的右下角坐标
      tmpw, tmph: 框原始的宽和高
    """
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    """用网络回归出的偏移量校准人脸框

    参数：
      bbox: 上一级网络生成的框
      reg : 框的偏移量
    返回：
      校准后的框（原图绝对坐标）
    """
    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0] + 1
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1] + 1
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


def py_nms(dets, thresh, mode="Union"):
    """非极大值抑制（NMS）

    :param dets: [[x1, y1, x2, y2, score], ...]
    :param thresh: 保留 overlap <= thresh 的框
    :return: 保留下来的框的索引
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def generate_bbox(cls_map, reg, scale, threshold):
    """根据 PNet 的输出生成人脸框（映射回原图坐标）"""
    stride = 2
    cellsize = 12

    t_index = np.where(cls_map > threshold)

    # 没有检测到人脸
    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[i, t_index[0], t_index[1]] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg])
    # shape [n, 9]
    return boundingbox.T
