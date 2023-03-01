import numpy as np
import cv2
import os
import shutil
import scipy.interpolate
import copy
import yaml
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


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
        fill_value = 125

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
        elif crop_type == "middle":
            # Get the widest point of original image, then get the corners from that.
            width_min = np.ceil(distorted_px[int(h / 2), 0, 0]).astype(np.int32)
            width_max = np.ceil(distorted_px[int(h / 2), -1, 0]).astype(np.int32)
            height_min = np.ceil(distorted_px[0, int(w / 2), 1]).astype(np.int32)
            height_max = np.ceil(distorted_px[-1, int(w / 2), 1]).astype(np.int32)
            img_dist = img_dist[height_min:height_max, width_min:width_max]
        else:
            raise ValueError

    if chan == 1:
        img_dist = img_dist[:, :, 0]

    return img_dist


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



def distort_line_data():
    # root = "/home/liyongjing/Egolee/programs/fisheye-distortion/local_files/data/line"
    # root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"
    # dst_root = (root, "line_fish_eye")

    root = "/mnt/nas01/algorithm_data/training_data/edge_line/indoor_edge_line_1219"

    dst_root = os.path.join(root, "line_fish_eye")

    s_img_root = os.path.join(dst_root, "image")
    # s_img_root = os.path.join(dst_root, "png")
    s_line_object_root = os.path.join(dst_root, "line_object")
    s_line_cls_root = os.path.join(dst_root, "line_cls")
    s_line_order_root = os.path.join(dst_root, "line_order")

    img_names = [name for name in os.listdir(s_img_root)
                 if name.split(".")[-1] in ["jpg", "png"]]

    dst_distort_root = os.path.join(root, "line_fish_eye_distort")
    if os.path.exists(dst_distort_root):
        shutil.rmtree(dst_distort_root)
    os.mkdir(dst_distort_root)

    s_distort_img_root = os.path.join(dst_distort_root, "image")
    s_distort_line_object_root = os.path.join(dst_distort_root, "line_object")
    s_distort_line_cls_root = os.path.join(dst_distort_root, "line_cls")
    s_distort_line_order_root = os.path.join(dst_distort_root, "line_order")
    for _dir in [s_distort_img_root, s_distort_line_object_root,
                 s_distort_line_cls_root, s_distort_line_order_root]:
        os.mkdir(_dir)

    for img_name in img_names:
        img_path = os.path.join(s_img_root, img_name)
        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)

        s_base_name = img_name.split(".")[0]
        object_map_path = os.path.join(s_line_object_root, s_base_name + ".npy")
        cls_map_path = os.path.join(s_line_cls_root, s_base_name + ".npy")
        order_map_path = os.path.join(s_line_order_root, s_base_name + ".npy")

        object_map = np.load(object_map_path)
        cls_map = np.load(cls_map_path)
        order_map = np.load(order_map_path)

        calib_path = os.path.join(root, "yml", s_base_name + ".yml")
        calib_infos = parse_calib(calib_path)
        if calib_infos == False:
            continue

        # calib_infos["dsitort"] = np.array([-0.163775, 0.0192114, 0.000170357, 0.000205845])
        # calib_infos["dsitort"] = np.array([-0.70, 0.3, 0.1, 0.2])
        # calib_infos["dsitort"] = np.array([-0.29, 0.22, -0.07, 0.18])
        # calib_infos["dsitort"] = np.array([-0.2, -0.21, 0.41, -0.4])
        # calib_infos["dsitort"] = np.array([-0.38, -0.36, 0.01, 0.45])
        # calib_infos["dsitort"] = np.array([-0.47, -0.29, 0.47, 0.01])
        generate_times = 1
        for _ in tqdm(range(generate_times)):
            if generate_times > 1:
                k0 = random.uniform(-0.8, 0.8)
                k1 = random.uniform(-0.5, 0.5)
                k2 = random.uniform(-0.8, 0.8)
                k3 = random.uniform(-0.5, 0.5)
            else:
                # k0, k1, k2, k3 = [-0.6, -0.35, 0.47, 0.1]    # 测试比较好的一组参数
                # k0, k1, k2, k3 = [-0.62, -0.35, -0.07, -0.07]    # 测试比较好的一组参数
                # k0, k1, k2, k3 = [0.53, -0.42, 0.31, -0.43]    # 测试不好的一组参数，真值会有问题
                k0, k1, k2, k3 = [-1/3, 1/5, -1/7, 1/9]   # fthea # 测试不好的一组参数，真值会有问题
                # k0, k1, k2, k3 = [0, 0, 0, 0]   # fthea # 测试不好的一组参数，真值会有问题
                # k0, k1, k2, k3 = [1, 0, 0, 0]   # fthea # 测试不好的一组参数，真值会有问题
                # k0, k1, k2, k3 = [1/3, 2/15, 17/315, 62/2835]   # fthea # 测试不好的一组参数，真值会有问题
                # k0, k1, k2, k3 = [1/3, 2/15, 17/315, 62/2835]   # fthea # 测试不好的一组参数，真值会有问题

            distort = np.array([k0, k1, k2, k3])
            calib_infos["dsitort"] = distort
            dist_info = "_".join([str(round(distort[0], 2)),
                                  str(round(distort[1], 2)),
                                  str(round(distort[2], 2)),
                                  str(round(distort[3], 2)), ])

            s_base_name = img_name.split(".")[0]
            s_base_name = s_base_name + dist_info

            mode = 'nearest'  # 'linear', 'nearest'
            crop_output = True
            crop_type = "corner"   # "corner", 'middle'
            dist_img = distort_image(img, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                          crop_output, crop_type)

            if dist_img is None:
                continue

            if dist_img.shape[0] < 100 or dist_img.shape[1] < 100:
                print("filter img, too small")
                continue

            if 1:
                dist_object_map = distort_image(object_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                              crop_output, crop_type)

                dist_cls_map = distort_image(cls_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                              crop_output, crop_type)

                dist_order_map = distort_image(order_map, calib_infos["cam_K"], calib_infos["dsitort"], mode,
                              crop_output, crop_type)



                np.save(os.path.join(s_distort_line_object_root, s_base_name + ".npy"), dist_object_map)
                np.save(os.path.join(s_distort_line_cls_root, s_base_name + ".npy"), dist_cls_map)
                np.save(os.path.join(s_distort_line_order_root, s_base_name + ".npy"), dist_order_map)

            cv2.imwrite(os.path.join(s_distort_img_root, s_base_name + ".jpg"), dist_img)

            # plt.subplot(2, 1, 1)
            # plt.imshow(img)
            # plt.subplot(2, 1, 2)
            # plt.imshow(dist_img)
            # plt.show()
            # exit(1)


if __name__ == "__main__":
    print("Start")
    distort_line_data()
    print("end")






