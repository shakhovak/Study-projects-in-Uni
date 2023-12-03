import cv2
import glob
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image_hsv = cv2.GaussianBlur(image_hsv, (11, 11), 1)

    light_orange = (1, 190, 150)
    dark_orange = (30, 255, 255)
    light_white = (60, 0, 200)
    dark_white = (145, 145, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10), anchor=(1, 1))

    mask0 = cv2.inRange(image_hsv, light_orange, dark_orange)
    mask0 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel)
    mask1 = cv2.inRange(image_hsv, light_white, dark_white)
    mask = mask0 + mask1

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10), anchor=(1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25), anchor=(1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)

    return mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = "train" if args.is_train else "test"

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask

    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
