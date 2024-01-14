import numpy as np
import torch
from torch import nn
from torchvision.ops import nms


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

# heatmap解码
def decode_bbox(pred_hms, pred_whs, confidence, cuda):
    pred_hms = pool_nms(pred_hms)

    b, c, output_h, output_w = pred_hms.shape
    detects = []

    for batch in range(b):
        heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))
        # -------------------------------------------------------------------------#
        #   xv              128*128,    特征点的x轴坐标
        #   yv              128*128,    特征点的y轴坐标
        # -------------------------------------------------------------------------#
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()

        # -------------------------------------------------------------------------#
        #   class_conf      128*128,    特征点的种类置信度
        #   class_pred      128*128,    特征点的种类
        # -------------------------------------------------------------------------#
        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence

        # ----------------------------------------#
        #   计算调整后预测框的中心
        # ----------------------------------------#
        xv_mask = torch.unsqueeze(xv[mask], -1)
        yv_mask = torch.unsqueeze(yv[mask], -1)

        h = w = torch.ones_like(xv_mask)
        bboxes = torch.cat([xv_mask, yv_mask, h, w], dim=1)
        bboxes[:, [0]] /= output_w
        bboxes[:, [1]] /= output_h
        detect = torch.cat(
            [bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
        detects.append(detect)

    return detects


def corner_correct_boxes(box_xy, input_shape, image_shape, letterbox_image):
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale

    boxes = np.concatenate([box_yx[..., 0:1], box_yx[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape], axis=-1)
    return boxes


def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]

    for i, image_pred in enumerate(prediction):
        detections = prediction[i]
        if len(detections) == 0:
            continue

        unique_labels = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
            else:
                max_detections = detections_class

            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy = output[i][:, 0:2]
            output[i][:, :2] = corner_correct_boxes(box_xy, input_shape, image_shape, letterbox_image)
    return output
