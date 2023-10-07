import utils
from pycocotools import coco
import numpy as np
import cv2
import json
from tqdm import tqdm
import sys
import os

def segmentationDetection(image: 'np.ndarray', file):

    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours of blobs
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    for contour in contours:
        segmentation = []
        for i in range(len(contour)):
            segmentation.append(int(contour[i][0][0]))
            segmentation.append(int(contour[i][0][1]))
        segmentations.append([segmentation])
    return segmentations

def contourDetector(file):
    image = cv2.imread("/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/patches_masks/patch_3.jpg",
                       cv2.IMREAD_GRAYSCALE)

    # Threshold the image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of blobs
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)


def custom_sort(string):
    string = os.path.basename(string)
    # Use regular expressions to extract the integer index
    import re
    match = re.search(r'\d+', string)
    if match:
        index = int(match.group())
        return index
    else:
        return 0  # Return 0 if no integer index is found

def polygon_area(x_s, y_s):
    n = len(x_s)
    area = 0

    for i in range(n):
        x1 = x_s[i]
        y1 = y_s[i]
        x2 = x_s[(i + 1) % n]
        y2 = y_s[(i + 1) % n]
        # Next vertex (wraps around for the last vertex)
        area += (x1 * y2 - x2 * y1)

    return round(0.5 * abs(area),2)

def maskToCoco(dirPath: 'str', savePath: 'str'):

    info = {}
    licenses = []
    images = []
    annotations = []
    categories = []

    info = {"year": "2023", "version": "10", "description": "Exported from roboflow.com", "contributor": "",
            "url": "https://public.roboflow.com/object-detection/undefined",
            "date_created": "2023-08-21T17:48:20+00:00"}
    licenses = [{"id": 1, "url": "https://creativecommons.org/licenses/by/4.0/", "name": "CC BY 4.0"}]

    categories.append({"id": 0, "name": "holes", "supercategory": "none"})

    u_paths = utils.get_file_paths(dirPath + "/*.jpg")
    l_paths = utils.get_file_paths(dirPath + "/*.png")
    l_paths = sorted(l_paths, key=custom_sort)
    u_paths = sorted(u_paths, key=custom_sort)
    print(l_paths)
    l_images = [cv2.imread(l_path, cv2.IMREAD_GRAYSCALE) for l_path in l_paths]
    l_images = [utils.filterImage(image, 1) for image in l_images]

    annot_id = 0
    const = 0

    id = 0
    id += const

    for i in tqdm(range(len(l_images))):

        l_image = l_images[i]
        images.append({"id": id, "license": 1, "file_name": u_paths[i], "height": len(l_image),
                       "width": len(l_image[0]),
                       "date_captured": "2023-09-14T17:48:20+00:00"})
        if np.all(l_image == 0):
            id += 1
            continue

        img_segmentations = segmentationDetection(l_image, l_paths[i])
        #print(contourDetector("df"))
        #if (os.path.basename(l_paths[i]) == "patch_3.jpg"): print(img_segmentations)
        for segmentation in img_segmentations:
            if (len(segmentation[0]) < 6): continue
            x_s = [segmentation[0][i] for i in range(0, len(segmentation[0]), 2)]
            y_s = [segmentation[0][i] for i in range(1, len(segmentation[0]), 2)]
            xmin = int(min(x_s))
            ymin = int(min(y_s))
            xmax = int(max(x_s))
            ymax = int(max(y_s))
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = polygon_area(x_s, y_s)
            annot = {"id": annot_id, "image_id": id,
                     "category_id": 0, "bbox": bbox,
                     "area": area,
                     "segmentation": segmentation,
                     "iscrowd": 0}
            annot_id += 1
            annotations.append(annot)
        id += 1

    new_coco = {"info": info, "licenses": licenses, "categories": categories, "images": images,
                "annotations": annotations}
    with open(savePath, "w") as json_file:
        json.dump(new_coco, json_file)

maskToCoco("/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/data/all_images/", "/home/antonio123/workspace/Github_projects/StatisticalImageAnalysis/annotations/all_images_anns.json")


