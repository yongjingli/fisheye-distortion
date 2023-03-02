import cv2
import json
import yaml
import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy.interpolate


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


def parse_calib(calib_path):
    calib_infos = dict()
    calib_info = yaml.load(open(calib_path), Loader=yaml.FullLoader)
    if 'camera_front_left_rectify' not in calib_info.keys():
        return False

    calib_cam = calib_info['camera_front_left_rectify']
    cam_K = np.array(calib_cam['intrinsics']).reshape(3, 3)

    baseline = calib_cam['baseline']
    dsitort = calib_info['camera_front_left']["distortion_coeffs"]

    calib_infos["cam_K"] = cam_K
    calib_infos["baseline"] = baseline
    calib_infos["dsitort"] = np.array(dsitort)
    return calib_infos


def get_img_line_map(img_path, label_path, show=False):
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
        cls = label_info["lines_and_labels"][1]
        # if cls != "door_wall_line":
        # if cls != "convex_wall_line_easy":
        #     continue

        # print("cls", cls)
        x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        x1 = int(min(max(0, x1), img_w - 1))
        x2 = int(min(max(0, x2), img_w - 1))
        y1 = int(min(max(0, y1), img_h - 1))
        y2 = int(min(max(0, y2), img_h - 1))

        # img show
        color = color_list[class_dict[cls]]
        if show:
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

            if show:
                cv2.circle(img, (x1, y1), 1, color, 1)
                # show order
                if point_order % 100 == 0:
                    img_show = cv2.putText(img_show, str(order_map[p_y, p_x]), (p_x, p_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    infos = {"img": img,
             "img_show": img_show,
             "order_map": order_map,
             "object_map": object_map,
             "cls_map": cls_map, }

    return infos


# mode: 'linear', 'nearest'
def distort_image(img: np.ndarray, cam_intr: np.ndarray, dist_coeff: np.ndarray,
                  mode: str = 'nearest', crop_output: bool = True,
                  crop_type: str = "corner") -> np.ndarray:
    """Apply fisheye distortion to an image

    Args:
        img (numpy.ndarray): BGR image. Shape: (H, W, 3)
        cam_intr (numpy.ndarray): The camera intrinsics matrix, in pixels: [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                            Shape: (3, 3)
        dist_coeff (numpy.ndarray): The fisheye distortion coefficients, for OpenCV fisheye module.
                            Shape: (1, 4)
        mode (DistortMode): For distortion, whether to use nearest neighbour or linear interpolation.
                            RGB images = linear, Mask/Surface Normals/Depth = nearest
        crop_output (bool): Whether to crop the output distorted image into a rectangle. The 4 corners of the input
                            image will be mapped to 4 corners of the distorted image for cropping.
        crop_type (str): How to crop.
            "corner": We crop to the corner points of the original image, maintaining FOV at the top edge of image.
            "middle": We take the widest points along the middle of the image (height and width). There will be black
                      pixels on the corners. To counter this, original image has to be higher FOV than the desired output.

    Returns:
        numpy.ndarray: The distorted image, same resolution as input image. Unmapped pixels will be black in color.
    """
    assert cam_intr.shape == (3, 3)
    assert dist_coeff.shape == (4,)

    imshape = img.shape
    if len(imshape) == 3:
        h, w, chan = imshape
    elif len(imshape) == 2:
        h, w = imshape
        chan = 1
    else:
        raise RuntimeError(f'Image has unsupported shape: {imshape}. Valid shapes: (H, W), (H, W, N)')

    imdtype = img.dtype

    # Get array of pixel co-ords
    xs = np.arange(w)
    ys = np.arange(h)
    xv, yv = np.meshgrid(xs, ys)
    img_pts = np.stack((xv, yv), axis=2)  # shape (H, W, 2)
    img_pts = img_pts.reshape((-1, 1, 2)).astype(np.float32)  # shape: (N, 1, 2)

    # Get the mapping from distorted pixels to undistorted pixels
    undistorted_px = cv2.fisheye.undistortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    # undistorted_px = cv2.fisheye.distortPoints(img_pts, cam_intr, dist_coeff)  # shape: (N, 1, 2)
    undistorted_px = cv2.convertPointsToHomogeneous(undistorted_px)  # Shape: (N, 1, 3)
    undistorted_px = np.tensordot(undistorted_px, cam_intr, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
    undistorted_px = cv2.convertPointsFromHomogeneous(undistorted_px)  # Shape: (N, 1, 2)
    undistorted_px = undistorted_px.reshape((h, w, 2))  # Shape: (H, W, 2)
    undistorted_px = np.flip(undistorted_px, axis=2)  # flip x, y coordinates of the points as cv2 is height first

    # Map RGB values from input img using distorted pixel co-ordinates
    if chan == 1:
        img = np.expand_dims(img, 2)
    # try:
    if chan == 1:
        fill_value = 0
    else:
        fill_value = 0

    interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode,
                                                               bounds_error=False, fill_value=fill_value)
                     for channel in range(chan)]

    # interpolators = [scipy.interpolate.RegularGridInterpolator((ys, xs), img[:, :, channel], method=mode,
    #                                                            bounds_error=False, fill_value=0)
    #                  for channel in range(chan)]
    img_dist = np.dstack([interpolator(undistorted_px) for interpolator in interpolators])

    # except:
    #     print("bounds_error happen skip")
    #     return None

    if imdtype == np.uint8:
        # RGB Image
        img_dist = img_dist.round().clip(0, 255).astype(np.uint8)
    elif imdtype == np.uint16:
        # Mask
        img_dist = img_dist.round().clip(0, 65535).astype(np.uint16)
    elif imdtype == np.float16 or imdtype == np.float32 or imdtype == np.float64:
        img_dist = img_dist.astype(imdtype)
    else:
        raise RuntimeError(f'Unsupported dtype for image: {imdtype}')

    if crop_output:
        # Crop rectangle from resulting distorted image
        # Get mapping from undistorted to distorted
        distorted_px = cv2.convertPointsToHomogeneous(img_pts)  # Shape: (N, 1, 3)
        cam_intr_inv = np.linalg.inv(cam_intr)
        distorted_px = np.tensordot(distorted_px, cam_intr_inv, axes=(2, 1))  # To camera coordinates, Shape: (N, 1, 3)
        distorted_px = cv2.convertPointsFromHomogeneous(distorted_px)  # Shape: (N, 1, 2)
        distorted_px = cv2.fisheye.distortPoints(distorted_px, cam_intr, dist_coeff)  # shape: (N, 1, 2)
        distorted_px = distorted_px.reshape((h, w, 2))
        if crop_type == "corner":
            # Get the corners of original image. Round values up/down accordingly to avoid invalid pixel selection.
            top_left = np.ceil(distorted_px[0, 0, :]).astype(np.int)
            bottom_right = np.floor(distorted_px[(h - 1), (w - 1), :]).astype(np.int)
            img_dist = img_dist[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]

            height_min = top_left[1]
            width_min = top_left[0]

        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError
    else:
        width_min = 0
        height_min = 0

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist, width_min, height_min


def get_distort_img_line_map(img, object_map, cls_map, order_map,
                             calib_infos, mode, crop_output, crop_type):
    dist_img, width_min, height_min = distort_image(img, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                             crop_output, crop_type)

    dist_object_map, width_min, height_min = distort_image(object_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                                    crop_output, crop_type)

    dist_cls_map, width_min, height_min = distort_image(cls_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                                 crop_output, crop_type)

    dist_order_map, width_min, height_min = distort_image(order_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                                   crop_output, crop_type)

    new_cam_k = copy.deepcopy(calib_infos["cam_K"])
    new_cam_k[0, 2] = new_cam_k[0, 2] - width_min
    new_cam_k[1, 2] = new_cam_k[1, 2] - height_min

    infos = {"dist_img": dist_img,
             "dist_object_map": dist_object_map,
             "dist_cls_map": dist_cls_map,
             "dist_order_map": dist_order_map,
             "cam_K": calib_infos["cam_K"],
             "new_cam_k": new_cam_k,
             "dsitort": calib_infos["dsitort"],
             "baseline": calib_infos["baseline"]
             }
    return infos


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

        # object_line[:, :] = object_line[:, ::-1]
        # curve_lines.append([object_line, cls_id])
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

        # cv2.circle(img, (x1, y1), 1, color, 1)
        cv2.line(img, (x1, y1), (x2, y2), color, 3)
        pre_point = cur_point
        # show order
        # if i % 100 == 0:
        #     img = cv2.putText(img, str(i), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)

    txt_i = len(points) // 2
    txt_x = int(points[txt_i][0])
    txt_y = int(points[txt_i][1])
    # img = cv2.putText(img, cls_name, (txt_y, txt_x), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)

    return img


def get_cls_names():
    cls_names = dict()
    for k, v in class_dict.items():
        cls_names.update({v: k})
    return cls_names


def parse_distort_img_line_map(img, object_map, cls_map, order_map):
    img_show = copy.deepcopy(img)
    # parse line
    object_lines = parse_object_map(object_map, img)
    cls_lines = parse_cls_map(cls_map, object_lines)
    order_lines = parse_order_map(order_map, object_lines)
    curve_lines = get_curve_lines(object_lines, cls_lines, order_lines)

    cls_names = get_cls_names()
    for curve_id, curve_line in enumerate(curve_lines):
        points, cls = curve_line
        color = color_list[cls]

        cls_name = cls_names[cls]
        img_show = draw_curce_line_on_img(img_show, points, cls_name, color)

    return img_show
