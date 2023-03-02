import os
import cv2
import shutil
import json
import numpy as np
from tqdm import tqdm
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    s_calib_root = os.path.join(dst_root, "calib")

    s_distort_img_root = os.path.join(dst_root, "distort_image")
    s_distort_img_show_root = os.path.join(dst_root, "distort_image_show")
    s_distort_line_object_root = os.path.join(dst_root, "distort_line_object")
    s_distort_line_cls_root = os.path.join(dst_root, "distort_line_cls")
    s_distort_line_order_root = os.path.join(dst_root, "distort_line_order")
    s_distort_calib_root = os.path.join(dst_root, "distort_calib")

    for _dir in [s_img_root, s_line_object_root,
                 s_line_cls_root, s_line_order_root,
                 s_distort_img_root, s_distort_line_object_root,
                 s_distort_line_cls_root, s_distort_line_order_root,
                 s_img_show_root, s_distort_img_show_root,
                 s_calib_root, s_distort_calib_root]:
        os.mkdir(_dir)

    img_root = os.path.join(root, "png")
    label_root = os.path.join(root, "json")
    calib_root = os.path.join(root, "yml")

    img_names = [name for name in os.listdir(img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    for img_name in tqdm(img_names[:100]):
        # img_name = "1cf79272-ffac-4136-b962-10d82a21d353.png"
        img_name = "1cf79272-ffac-4136-b962-10d82a21d353.png"

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

        k0, k1, k2, k3 = [-1 / 3, 1 / 5, -1 / 7, 1 / 9]   # 等距投影模型
        distort = np.array([k0, k1, k2, k3])
        calib_infos["dsitort"] = distort

        # 保存内参信息
        s_infos = {"cam_K": calib_infos["cam_K"].reshape(-1).tolist(),
                   "dsitort": calib_infos["dsitort"].reshape(-1).tolist(),
                   "baseline": calib_infos["baseline"]}

        s_calib_path = os.path.join(s_calib_root, s_base_name + ".json")
        s_infos = json.dumps(s_infos)
        with open(s_calib_path, 'w') as fp:
            fp.write(s_infos)

        # 保存distort的img和line的信息
        mode = 'nearest'  # 'linear', 'nearest'
        crop_output = True
        crop_type = "corner"  # "corner", 'middle'


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

        # 保存内参信息
        s_infos = {"cam_K": distort_img_line_infos["cam_K"].reshape(-1).tolist(),
                   "new_cam_k": distort_img_line_infos["new_cam_k"].reshape(-1).tolist(),
                   "dsitort": distort_img_line_infos["dsitort"].reshape(-1).tolist(),
                   "baseline": distort_img_line_infos["baseline"]}

        s_calib_path = os.path.join(s_distort_calib_root, s_base_name + ".json")
        s_infos = json.dumps(s_infos)
        with open(s_calib_path, 'w') as fp:
            fp.write(s_infos)

        exit(1)


def parse_calib(calib_path):
    calib_infos = dict()
    with open(calib_path, 'r') as fp:
        calib_info = json.load(fp)

    cam_K = np.array(calib_info['cam_K']).reshape(3, 3)

    baseline = calib_info['baseline']
    dsitort = calib_info["dsitort"]

    calib_infos["cam_K"] = cam_K
    calib_infos["baseline"] = baseline
    calib_infos["dsitort"] = np.array(dsitort)

    if "new_cam_k" in calib_info.keys():
        calib_infos["new_cam_k"] = np.array(calib_info['new_cam_k']).reshape(3, 3)

    return calib_infos


def undistort_fish_eye_img():
    # root = "/home/liyongjing/Egolee/programs/ULSD-ISPRS/local_files/data/indoor_line_no_crop"
    root = "/home/liyongjing/Egolee/programs/ULSD-ISPRS/local_files/data/indoor_line_corner_crop"
    s_distort_img_root = os.path.join(root, "distort_image")
    s_distort_img_show_root = os.path.join(root, "distort_image_show")
    s_distort_line_object_root = os.path.join(root, "distort_line_object")
    s_distort_line_cls_root = os.path.join(root, "distort_line_cls")
    s_distort_line_order_root = os.path.join(root, "distort_line_order")
    s_distort_calib_root = os.path.join(root, "distort_calib")

    img_names = [name for name in os.listdir(s_distort_img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    for img_name in tqdm(img_names[:100]):
        s_base_name = img_name[:-4]

        img_path = os.path.join(s_distort_img_root, s_base_name + ".jpg")
        object_map_path = os.path.join(s_distort_line_object_root, s_base_name + ".npy")
        cls_map_path = os.path.join(s_distort_line_cls_root, s_base_name + ".npy")
        order_map_path = os.path.join(s_distort_line_order_root, s_base_name + ".npy")
        calib_path = os.path.join(s_distort_calib_root, s_base_name + ".json")

        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)
        img_show2 = copy.deepcopy(img)
        object_map = np.load(object_map_path)
        cls_map = np.load(cls_map_path)
        order_map = np.load(order_map_path)
        calib_info = parse_calib(calib_path)


        K = calib_info["cam_K"]
        D = calib_info["dsitort"]
        if "new_cam_k" in calib_info:
            K_distort = calib_info["new_cam_k"]
        else:
            K_distort = K

        img_undistorted = copy.deepcopy(img)
        img_undistorted = cv2.fisheye.undistortImage(img_undistorted, K_distort, D=D, Knew=K_distort)
        img_undistorted = img_undistorted.astype(np.uint8)

        plt.subplot(2, 1, 1)
        plt.imshow(img)

        plt.subplot(2, 1, 2)
        plt.imshow(img_undistorted)

        plt.show()


if __name__ == "__main__":
    print("Start")
    # root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line"
    # dst_root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line_distort"

    root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"
    dst_root = "/mnt/data10/liyj/data/debug"

    # test_distort_line_data(root, dst_root)

    # 将生成的鱼眼图像通过内参和畸变参数重新生成为无畸变的图像
    # 需要特别注意的是生成鱼眼图像时内参会发生变化，主要是进行crop的时候cx和cy会发生变化
    undistort_fish_eye_img()
    print("End")



