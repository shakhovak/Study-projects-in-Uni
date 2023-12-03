import cv2
import json
import numpy as np
import os.path as osp

def json2numpy(path: str, src: str, dst: str):
    with open(path, 'r') as f:
        data = json.loads(f.read())
    for _, v in data.items():
        regions = v['regions']
        file_name = v['filename']
        img = cv2.imread(osp.join(src, file_name))
        mask = np.zeros((img.shape[0], img.shape[1]))
        for _, region in regions.items():
            x = region['shape_attributes']['all_points_x']
            y = region['shape_attributes']['all_points_y']
            cnt = np.array(list(zip(x, y)), dtype=int)
            cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
        cv2.imwrite(osp.join(dst, file_name.replace('.jpg', '.png')), mask)
