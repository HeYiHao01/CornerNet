import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from detect_config import CornerNet
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

if __name__ == "__main__":
    #   map_mode=0 整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map
    #   map_mode=1 获得预测结果
    #   map_mode=2 获得真实框
    #   map_mode=3 计算VOC mAP
    #   map_mode=4 计算当前数据集COCO的0.50:0.95 mAP
    map_mode = 0
    classes_path = 'config/classes_path.txt'

    MINOVERLAP = 0.25

    confidence = 0.02
    nms_iou = 0.5
    score_threhold = 0.5

    map_vis = False

    VOCdevkit_path = 'dataset'
    #   结果输出的文件夹
    map_out_path = 'logs/map_out/mobilenet_v3_v4_SA@.25'
    eval_f = 'config/total_v4.txt'

    image_ids = [image_id.split() for image_id in open(eval_f).readlines()]

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        cornernet = CornerNet(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = image_id[0]
            image_id = image_id[0][-8:-4]
            image = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            cornernet.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            image_path = image_id[0]
            objs = image_id[1:]
            image_id = image_id[0][-8:-4]
            w, h = Image.open(image_path).size
            with open(os.path.join(map_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
                for obj in objs:
                    difficult_flag = False
                    obj_name = 'point'
                    center_x, center_y, _ = obj.split(',')
                    left = float(center_x) - 3 if (float(center_x) - 3) >= 0 else 0
                    top = float(center_y) - 3 if (float(center_y) - 3) >= 0 else 0
                    right = float(center_x) + 3 if (float(center_x) + 3) <= w else w
                    bottom = float(center_y) + 3 if (float(center_y) + 3) <= h else h

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
