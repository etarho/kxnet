# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import cv2
import datetime
import numpy as np
import json
import os.path as osp
from tqdm import trange
from skimage import measure
from pycocotools import mask

# +
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]
YELLOW = [255, 255, 0]
MAGENTA = [255, 0, 255]
CYAN = [0, 255, 255]

info = {
    "description": "HPDataset 023",
    "url": "",
    "version": "1.0",
    "year": 2020,
    "contributor": "Kosuke Ishikawa <etarho.py@gmail.com>",
    "date_created": "{}".format(datetime.date.today()),
}

LICENSES = [{"url": "", "id": 1, "name": "BMPE"}]

CATEGORIES = [
    {
        "id": 1,
        "name": "fixed_edge",
        "supercategory": "forceps",
    },
    {
        "id": 2,
        "name": "shaft",
        "supercategory": "forceps",
    },
    {
        "id": 3,
        "name": "opening_edge",
        "supercategory": "forceps",
    },
]


# -

class Img2Mask:
    def __init__(self, im, threshold=100):
        """
        Convert to mask
        """
        self.img = im.copy()
        self.img[im >= threshold] = 255
        self.img[im < threshold] = 0

    def __call__(self):
        img_ = self.img.copy()
        for color in [YELLOW, MAGENTA, CYAN]:
            img_[np.where((self.img == color).all(axis=2))] = BLACK

        return img_

    def to_npmask(self):
        img_ = np.zeros([img.shape[0], img.shape[1]])
        for i, color in enumerate([RED, GREEN, BLUE]):
            img_[np.where((self.img == color).all(axis=2))] = i + 1

        return img_


def mask2coco(npmasks, img_id, coco_dict):
    """
    Convert ndarray masks to COCO json file
    """

    images = {
        "license": 1,
        "coco_url": "",
        "date_captured": "",
        "flickr_rul": "",
        "id": img_id,
        "height": npmasks[0].shape[0],
        "width": npmasks[0].shape[1],
        "file_name": "img_{:05d}.jpg".format(img_id),
    }

    coco_dict["images"].append(images)

    for category, npmask in enumerate(npmasks):
        fortran_ground_truth_binary_mask = np.asfortranarray(npmask)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(npmask, 0.5)

        annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": category + 1,
            "id": 1000000 + (category + 1) * 100000 + img_id,
        }

        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)

        coco_dict["annotations"].append(annotation)


if __name__ == "__main__":

    coco_train = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    coco_val = {
        "info": info,
        "licenses": LICENSES,
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    ds_num = int(input("Dataset: #"))
    img_dir = "../../Database/ds_{:03d}/masks".format(ds_num)
    num_img_dict = {"train": 10000, "val": 2000}
    for phase in ["train", "val"]:
        for i in trange(num_img_dict[phase]):
            img = cv2.imread(osp.join(img_dir, phase, "img_{:05d}.jpg".format(i + 1)))
            img2mask = Img2Mask(img, threshold=100)
            mask_ = img2mask()
            npmasks = (mask_ / 255).astype(np.uint8)
            npmasks = [npmasks[:, :, 0], npmasks[:, :, 1], npmasks[:, :, 2]]

            if phase == "train":
                mask2coco(npmasks=npmasks, img_id=i + 1, coco_dict=coco_train)
            else:
                mask2coco(npmasks=npmasks, img_id=i + 1, coco_dict=coco_val)

    #             cv2.imwrite(img_dir + '/{}/mask_{}.png'.format(phase, i+1), mask_)

    with open(
        "../../Database/ds_{:03d}/anno_train.json".format(ds_num), "w"
    ) as f_train:
        json.dump(coco_train, f_train, indent=4)
    with open("../../Database/ds_{:03d}/anno_val.json".format(ds_num), "w") as f_val:
        json.dump(coco_val, f_val, indent=4)

print("Finished!!")
