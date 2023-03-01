import os
import cv2
import json
import shutil
import numpy as np
import copy
import matplotlib.pyplot as plt

color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 215, 255), (0, 255, 255),
              (221, 160, 221),  (128, 0, 128), (203, 192, 255), (238, 130, 238), (0, 69, 255),
              (130, 0, 75), (255, 255, 0), (250, 51, 153), (214, 112, 218), (255, 165, 0),
              (169, 169, 169), (18, 74, 115),
              (240, 32, 160), (192, 192, 192), (112, 128, 105), (105, 128, 128),
              ]


class_dict = {
    'concave_wall_line': 0,
    'convex_wall_line': 1,
    'wall_ground_line': 2,
    'cabinet_wall_line': 3,
    'cabinet_ground_line': 4,
    'sofa_ground_line': 5,
    'stair_line': 6,
    'ceiling_line': 7,
    'wall_ground_curve': 8,
    'stair_curve': 9,
    'cabinet_ground_curve': 10,
    'sofa_ground_curve': 11,
    "concave_wall_line_easy": 12,   # new 0704
    "convex_wall_line_easy": 13,
    "door_wall_line": 14,
    "line_reserve": 15,
    "outside_elevator_door_ground_line": 16,
    "inside_elevator_door_ground_line": 17,
    "outside_elevator_door_concave_wall_line": 18,
    "inside_elevator_door_concave_line": 19,
    "inside_elevator_door_convex_line": 20,
}

class_dict_chinese = {
    '凹墙线': 0,
    '凸墙线': 1,
    '墙地线': 2,
    '柜墙线': 3,
    '柜地线': 4,
    '沙发地线': 5,
    '楼梯线': 6,
    '天花板线': 7,
    '墙地曲线': 8,
    '楼梯曲线': 9,
    '柜地曲线': 10,
    '沙发地曲线': 11,
    "凹墙线-容易": 12,
    "凸墙线-容易": 13,
    "门框墙线": 14,
    "保留线位置": 15,
    "电梯门外地沿线": 16,
    "电梯门内地沿线": 17,
    "电梯门外凹沿线": 18,
    "电梯门内凹沿线": 19,
    "电梯门内凸沿线": 20,
}

CLASS_NUM = len(class_dict.keys())


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "(%d, %d)" % (self.x, self.y)

    def line2(self, another):
        if self.x == another.x:
            step = 1 if self.y < another.y else -1
            y = self.y
            while y != another.y:
                yield Point(self.x, y)
                y += step
            yield Point(self.x, y)

        elif self.y == another.y:
            step = 1 if self.x < another.x else -1
            x = self.x
            while x != another.x:
                yield Point(x, self.y)
                x += step
            yield Point(x, self.y)
        else:
            d_x = self.x - another.x
            d_y = self.y - another.y
            s_x = 1 if d_x < 0 else -1
            s_y = 1 if d_y < 0 else -1

            # if d_y:
            if abs(d_x) > abs(d_y):
                delta = 1. * d_x / d_y
                for i in range(0, abs(d_x) + 1):
                    yield Point(self.x + i * s_x, self.y + i * s_x / delta)

            # elif d_x:
            else:
                delta = 1. * d_y / d_x
                for i in range(0, abs(d_y) + 1):
                    yield Point(self.x + i * s_y / delta, self.y + i * s_y)


def generate_line_data():
    # root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line"
    root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"

    dst_root = os.path.join(root, "line_fish_eye")

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    s_img_root = os.path.join(dst_root, "image")
    s_line_object_root = os.path.join(dst_root, "line_object")
    s_line_cls_root = os.path.join(dst_root, "line_cls")
    s_line_order_root = os.path.join(dst_root, "line_order")
    for _dir in [s_img_root, s_line_object_root, s_line_cls_root, s_line_order_root]:
        os.mkdir(_dir)

    img_root = os.path.join(root, "png")
    label_root = os.path.join(root, "json")

    img_names = [name for name in os.listdir(img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    for img_name in img_names[:30]:
        img_path = os.path.join(img_root, img_name)
        label_path = os.path.join(label_root, img_name[:-3] + "json")

        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        map_size = (img_h, img_w)
        object_map = np.zeros(map_size, dtype=np.uint16)
        order_map = np.zeros(map_size, dtype=np.uint16)
        cls_map = np.zeros(map_size, dtype=np.uint16)

        with open(label_path, "r") as fp:
            label_infos = json.load(fp)

        img_show = copy.deepcopy(img)
        for object_id, label_info in enumerate(label_infos['objects']):
            if "lines_and_labels" not in label_info.keys():
                continue

            line = label_info["lines_and_labels"][0]
            # line = [[1213, 40], [1221, 0]]
            # line = [[714, 852], [741, 852]]
            cls = label_info["lines_and_labels"][1]
            # if cls != "door_wall_line":
            # if cls != "convex_wall_line_easy":
            #     continue

            print("cls", cls)
            x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
            x1 = int(min(max(0, x1), img_w - 1))
            x2 = int(min(max(0, x2), img_w - 1))
            y1 = int(min(max(0, y1), img_h - 1))
            y2 = int(min(max(0, y2), img_h - 1))

            # img show
            color = color_list[class_dict[cls]]
            cv2.line(img_show, (x1, y1), (x2, y2), color, 3)
            # cv2.line(img_show, (x1, y1), (x2, y2), (0, 255, 0), 3, lineType=LINE_8)

            # draw object map
            # cv2.line(object_map, (x1, y1), (x2, y2), (object_id + 1,))

            # draw cls map
            cls_id = class_dict[cls] + 1
            # cv2.line(cls_map, (x1, y1), (x2, y2), (cls_id,))

            # draw order map
            pointA = Point(x1, y1)
            pointB = Point(x2, y2)
            # points = []
            for point_order, point in enumerate(pointA.line2(pointB)):
                p_x = int(min(max(0, point.x), img_w - 1))
                p_y = int(min(max(0, point.y), img_h - 1))
                order_map[p_y, p_x] = point_order

                object_map[p_y, p_x] = object_id + 1
                cls_map[p_y, p_x] = cls_id


            # print(line)
            # break
                # points.append(point)
            # print(points[0], points[-1], len(points))
        s_base_name = img_name.split(".")[0]
        np.save(os.path.join(s_line_object_root, s_base_name + ".npy"), object_map)
        np.save(os.path.join(s_line_cls_root, s_base_name + ".npy"), cls_map)
        np.save(os.path.join(s_line_order_root, s_base_name + ".npy"), order_map)

        print("object num:", len(label_infos['objects']))

        cv2.imwrite(os.path.join(s_img_root, s_base_name + ".jpg"), img)

        # plt.imshow(img_show)
        # plt.show()


if __name__ == "__main__":
    print("Start")
    generate_line_data()
    print("end")