import cv2
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def prep_img_tensor(tensor):
    return tensor[0].permute(1, 2, 0).detach().cpu().numpy() \
           * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])


def prep_feat_tensor(tensor):
    return tensor[0].permute(1, 2, 0).detach().cpu().numpy()


def _get_roi(img):
    binary = np.mean(img, axis=2, keepdims=True)
    binary = cv2.resize(binary, (75, 75))
    binary = binary - binary.min()
    binary = binary / binary.max()
    binary = 1 - binary
    _, binary = cv2.threshold(binary, 0.3, 1, cv2.THRESH_BINARY)

    binary_padding = 37
    binary = cv2.copyMakeBorder(binary,
                                binary_padding,
                                binary_padding,
                                binary_padding,
                                binary_padding, 0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8))
    binary = np.array(binary, dtype=np.uint8)[binary_padding:-binary_padding, binary_padding:-binary_padding]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    try:
        best_contour = contours[0]
        if len(contours) > 1:
            for i in range(1, len(contours)):
                if len(contours[i]) > len(best_contour):
                    best_contour = contours[i]
        roi = np.zeros((75, 75))
        roi = cv2.fillPoly(roi, pts=[best_contour], color=(1, 1, 1))
        roi = cv2.resize(roi, img.shape[:2])
        return roi
    except:
        roi = np.zeros((75, 75))
        roi = cv2.resize(roi, img.shape[:2])
        return roi


def _vis_corr(dist, img1, img2, multiplier=2, sketchy=True):
    size = int(np.sqrt(dist.shape[0]))
    img_shape = 256 * multiplier
    offset = img_shape // size

    img1 = cv2.resize(img1, (512, 512))
    img2 = cv2.resize(img2, (512, 512))
    if sketchy:
        roi = _get_roi(img2)
    else:
        roi = np.zeros(img1.shape[:2])
        roi[offset // 2::offset * 2, offset // 2::offset * 2] = 1

    canv = np.zeros((img_shape * 2, img_shape, 3))
    canv[:img_shape, :, :] = img1
    canv[img_shape:, :, :] = img2

    color_counter = 0
    for j in range(dist.shape[0]):
        x2, y2 = j % size, j // size
        x2 = x2 * offset + offset // 2
        y2 = y2 * offset + offset // 2

        if roi[y2, x2] > 0.75:
            color_counter += 1

    colors = np.array(sns.color_palette("hls", color_counter))

    color_counter = 0
    for j in range(dist.shape[0]):
        i = np.argmin(dist[:, j])
        x1, y1 = i % size, i // size
        x2, y2 = j % size, j // size

        x1 = x1 * offset + offset // 2
        x2 = x2 * offset + offset // 2
        y1 = y1 * offset + offset // 2
        y2 = y2 * offset + offset // 2

        if roi[y2, x2] > 0.75:
            cv2.circle(canv, (x1, y1), 3 * multiplier, color=colors[color_counter], thickness=-1)
            cv2.circle(canv, (x2, y2 + size * offset), 3 * multiplier, color=colors[color_counter], thickness=-1)
            cv2.line(canv, (x1, y1), (x2, y2 + size * offset), color=colors[color_counter],
                     thickness=1 * multiplier, lineType=cv2.LINE_AA)
            color_counter += 1

        else:
            continue

    return canv


def _vis_map(maps):
    length = len(maps)
    size = maps[0].shape[0]
    for i in range(len(maps)):
        maps[i] = maps[i].reshape(size * size, -1)
    maps = np.concatenate(maps, axis=0)
    # maps = maps / (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-6)

    pca = PCA(n_components=3)
    maps = pca.fit_transform(maps)
    trans = QuantileTransformer()
    maps = trans.fit_transform(maps)

    maps = np.split(maps, length)
    maps = [m.reshape(size, size, 3) for m in maps]

    return maps


def gen_graph(mem):
    fig, axes = plt.subplots(11, len(mem["image1"]),
                             figsize=(len(mem["image1"]) * 3, 12 * 3),
                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,]})
    for idx in range(len(mem["image1"])):
        axes[0, idx].imshow(mem["image1"][idx])
        axes[1, idx].imshow(mem["image2"][idx])

        axes[2, idx].imshow(mem["warp_image12"][idx])
        axes[3, idx].imshow(mem["warp_image21"][idx])

        if mem["weight3_1"][idx] is not None:
            img_size = mem["warp_image12"][idx].shape[0]
            axes[4, idx].imshow(F.interpolate(mem["weight3_1"][idx].unsqueeze(0).unsqueeze(0),
                                              (img_size, img_size), mode="bilinear")[0, 0])
            axes[5, idx].imshow(F.interpolate(mem["weight3_2"][idx].unsqueeze(0).unsqueeze(0),
                                              (img_size, img_size), mode="bilinear")[0, 0])

        if mem["dist"][idx] is not None:
            corr = _vis_corr(mem["dist"][idx],
                            mem["image1"][idx], mem["image2"][idx])
            axes[6, idx].imshow(corr)

        map1, map2 = _vis_map([mem["res2_1"][idx], mem["res2_2"][idx]])
        axes[7, idx].imshow(map1)
        axes[8, idx].imshow(map2)

        map1, map2 = _vis_map([mem["res3_1"][idx], mem["res3_2"][idx]])
        axes[9, idx].imshow(map1)
        axes[10, idx].imshow(map2)

    return fig
