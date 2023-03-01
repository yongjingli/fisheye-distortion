import os
import cv2
import copy
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from generate_line_data import class_dict, class_dict_chinese, color_list


def get_img_dilate_mask(img):
    kernel = np.ones((5, 5), np.uint8)
    # ret, th = cv2.threshold(img[:, :, 0], 127, 255, cv2.THRESH_BINARY)
    img_mask = np.bitwise_and(img[:, :, 0] == 125,
                  img[:, :, 1] == 125,
                  img[:, :, 2] == 125)
    img_mask = img_mask.astype(np.uint8)

    dilation = cv2.dilate(img_mask, kernel, iterations=1)
    dilation = dilation.astype(np.bool)
    img[dilation] = 0
    plt.imshow(dilation)
    plt.show()
    return img


def parse_object_map(object_map, img):
    object_ids = np.unique(object_map)
    object_lines = []
    for object_id in object_ids:
        if object_id == 0:
            continue
        indx_y, indx_x = np.where(object_map == object_id)
        indx_x = indx_x.reshape(-1, 1)
        indx_y = indx_y.reshape(-1, 1)
        object_line = np.concatenate([indx_y, indx_x], axis=1)
        # img_dilate_mask = get_img_dilate_mask(copy.deepcopy(img))
        # img_dilate_mask = img
        #
        # img_value = img_dilate_mask[indx_y, indx_x, :]

        # mask = np.bitwise_or(img_value[:, 0, 0] != 0,
        #                      img_value[:, 0, 1] != 0,
        #                      img_value[:, 0, 2] != 0,)
        # object_line = object_line[mask]
        object_lines.append(object_line)
    return object_lines


def parse_cls_map(cls_map, object_lines):
    cls_lines =[]
    for object_line in object_lines:
        line_y = object_line[:, 0]
        line_x = object_line[:, 1]

        line_cls = cls_map[line_y, line_x]
        cls_lines.append(line_cls)
    return cls_lines


def parse_order_map(order_map, object_lines):
    order_lines =[]
    for object_line in object_lines:
        line_y = object_line[:, 0]
        line_x = object_line[:, 1]

        order_line = order_map[line_y, line_x]
        order_lines.append(order_line)
    return order_lines


def get_curve_lines(object_lines, cls_lines, order_lines):
    curve_lines = []
    for object_line, cls_line, order_line in zip(object_lines, cls_lines, order_lines):
        cls_id = np.argmax(np.bincount(cls_line)) - 1
        order_index = np.argsort(order_line)
        object_line_order = object_line[order_index]

        # 将点的坐标转为(w, h)的形式，也就是(x, y)
        object_line_order[:, :] = object_line_order[:, ::-1]

        curve_lines.append([object_line_order, cls_id])
    return curve_lines


def get_cls_names():
    cls_names = dict()
    for k, v in class_dict.items():
        cls_names.update({v: k})
    return cls_names


def draw_curce_line_on_img(img, points, cls_name, color=(0, 255, 0)):
    pre_point = points[0]
    for i, cur_point in enumerate(points[1:]):
        x1, y1 = int(pre_point[0]), int(pre_point[1])
        x2, y2 = int(cur_point[0]), int(cur_point[1])

        cv2.circle(img, (x1, y1), 1, color, 1)
        cv2.line(img, (x1, y1), (x2, y2), color, 3)
        pre_point = cur_point
        # show order
        # if i % 50 == 0:
        #     img = cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    txt_i = len(points) // 2
    txt_x = int(points[txt_i][0])
    txt_y = int(points[txt_i][1])
    # img = cv2.putText(img, cls_name, (txt_y, txt_x), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    return img


def parse_line_data():
    # root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line"
    # dst_root = root + "line_fish_eye_distort"

    root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"
    dst_root = os.path.join(root, "line_fish_eye_distort")

    s_img_root = os.path.join(dst_root, "image")
    s_line_object_root = os.path.join(dst_root, "line_object")
    s_line_cls_root = os.path.join(dst_root, "line_cls")
    s_line_order_root = os.path.join(dst_root, "line_order")

    img_names = [name for name in os.listdir(s_img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    cls_names = get_cls_names()

    for img_name in img_names:
        img_path = os.path.join(s_img_root, img_name)
        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)

        # s_base_name = img_name.split(".")[0]
        s_base_name = img_name[:-4]
        object_map_path = os.path.join(s_line_object_root, s_base_name + ".npy")
        cls_map_path = os.path.join(s_line_cls_root, s_base_name + ".npy")
        order_map_path = os.path.join(s_line_order_root, s_base_name + ".npy")

        object_map = np.load(object_map_path)
        cls_map = np.load(cls_map_path)
        order_map = np.load(order_map_path)

        # parse line
        object_lines = parse_object_map(object_map, img)
        cls_lines = parse_cls_map(cls_map, object_lines)
        order_lines = parse_order_map(order_map, object_lines)
        curve_lines = get_curve_lines(object_lines, cls_lines, order_lines)

        for curve_id, curve_line in enumerate(curve_lines):
            points, cls = curve_line
            color = color_list[cls]

            cls_name = cls_names[cls]
            img_show = draw_curce_line_on_img(img_show, points, cls_name, color)

        # print("object num:", len(curve_lines))

        plt.imshow(img_show)
        plt.show()

        # debug_path = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/debug"
        debug_path = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219/debug"
        cv2.imwrite((os.path.join(debug_path, img_name)), img_show)

        # plt.subplot(2, 2, 1)
        # plt.imshow(img)
        #
        # plt.subplot(2, 2, 2)
        # plt.imshow(object_map)
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(cls_map)
        #
        # plt.subplot(2, 2, 4)
        # plt.imshow(order_map)
        #
        # plt.show()


if __name__ == "__main__":
    print("Start")
    parse_line_data()
    print("End")





