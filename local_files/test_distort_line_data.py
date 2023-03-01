import os
import cv2
import shutil
import json
import numpy as np
from tqdm import tqdm
from copy import copy
from utils import parse_calib, get_img_line_map, \
    distort_image, get_distort_img_line_map, parse_distort_img_line_map


def test_distort_line_data(root, dst_root):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    s_img_root = os.path.join(dst_root, "image")
    s_img_show_root = os.path.join(dst_root, "image_show")
    s_line_object_root = os.path.join(dst_root, "line_object")
    s_line_cls_root = os.path.join(dst_root, "line_cls")
    s_line_order_root = os.path.join(dst_root, "line_order")

    s_distort_img_root = os.path.join(dst_root, "distort_image")
    s_distort_img_show_root = os.path.join(dst_root, "distort_image_show")
    s_distort_line_object_root = os.path.join(dst_root, "distort_line_object")
    s_distort_line_cls_root = os.path.join(dst_root, "distort_line_cls")
    s_distort_line_order_root = os.path.join(dst_root, "distort_line_order")

    for _dir in [s_img_root, s_line_object_root,
                 s_line_cls_root, s_line_order_root,
                 s_distort_img_root, s_distort_line_object_root,
                 s_distort_line_cls_root, s_distort_line_order_root,
                 s_img_show_root, s_distort_img_show_root]:
        os.mkdir(_dir)

    img_root = os.path.join(root, "png")
    label_root = os.path.join(root, "json")
    calib_root = os.path.join(root, "yml")

    img_names = [name for name in os.listdir(img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    for img_name in tqdm(img_names[:100]):
        # img_name = "1cf79272-ffac-4136-b962-10d82a21d353.png"

        img_path = os.path.join(img_root, img_name)
        label_path = os.path.join(label_root, img_name[:-3] + "json")
        calib_path = os.path.join(calib_root, img_name[:-3] + "yml")

        calib_infos = parse_calib(calib_path)
        if calib_infos == False:
            continue

        img_line_infos = get_img_line_map(img_path, label_path, show=True)
        img_line_infos["calib_infos"] = calib_infos

        s_base_name = img_name.split(".")[0]

        # 保存img和line的信息
        np.save(os.path.join(s_line_object_root, s_base_name + ".npy"), img_line_infos["object_map"])
        np.save(os.path.join(s_line_cls_root, s_base_name + ".npy"), img_line_infos["cls_map"])
        np.save(os.path.join(s_line_order_root, s_base_name + ".npy"), img_line_infos["order_map"])
        cv2.imwrite(os.path.join(s_img_root, s_base_name + ".jpg"), img_line_infos["img"])
        cv2.imwrite(os.path.join(s_img_show_root, s_base_name + ".jpg"), img_line_infos["img_show"])

        # 保存distort的img和line的信息
        mode = 'nearest'  # 'linear', 'nearest'
        crop_output = True
        crop_type = "corner"  # "corner", 'middle'

        k0, k1, k2, k3 = [-1 / 3, 1 / 5, -1 / 7, 1 / 9]   # 等距投影模型
        distort = np.array([k0, k1, k2, k3])
        calib_infos["dsitort"] = distort

        distort_img_line_infos = get_distort_img_line_map(img_line_infos["img"], img_line_infos["object_map"],
                                 img_line_infos["cls_map"], img_line_infos["order_map"],
                                 calib_infos, mode, crop_output, crop_type)

        np.save(os.path.join(s_distort_line_object_root, s_base_name + ".npy"),
                distort_img_line_infos["dist_object_map"])
        np.save(os.path.join(s_distort_line_cls_root, s_base_name + ".npy"),
                distort_img_line_infos["dist_cls_map"])
        np.save(os.path.join(s_distort_line_order_root, s_base_name + ".npy"),
                distort_img_line_infos["dist_order_map"])
        cv2.imwrite(os.path.join(s_distort_img_root, s_base_name + ".jpg"),
                    distort_img_line_infos["dist_img"])

        # parse distort img and line map
        distort_img_show = parse_distort_img_line_map(distort_img_line_infos["dist_img"],
                                   distort_img_line_infos["dist_object_map"],
                                   distort_img_line_infos["dist_cls_map"],
                                   distort_img_line_infos["dist_order_map"])


        cv2.imwrite(os.path.join(s_distort_img_show_root, s_base_name + ".jpg"),
                    distort_img_show)

        # exit(1)


if __name__ == "__main__":
    print("Start")
    # root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line"
    # dst_root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line_distort"

    root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"
    dst_root = "/mnt/data10/liyj/data/debug"

    test_distort_line_data(root, dst_root)
    print("End")



