import cv2
import numpy as np
import os.path as osp

def iou(m1, m2):
    return np.logical_and(m1, m2).sum() / np.logical_or(m1, m2).sum()

def compute_ious(img_name2mask, dir_path, aggregate=True):
    ious = dict()
    for img_name, mask in img_name2mask.items():
        true_mask = cv2.imread(
            osp.join(dir_path, img_name.replace(".jpg", ".png")),
            cv2.IMREAD_GRAYSCALE
            )
        if mask.dtype != bool:
            mask = mask > 0
        ious[img_name] = iou(mask, true_mask)
    if aggregate:
        return np.mean(list(ious.values()))
    return ious
        