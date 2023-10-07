from pycocotools import coco
import utils
import json
from tqdm import tqdm
import sys
from matplotlib.patches import Polygon
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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

def findImageId(coco_data: 'dict', desired_file_name):
    # Find the image_id with the specified file_name
    image_id = None
    for image in coco_data['images']:
        if image['file_name'] == desired_file_name:
            image_id = image['id']
            break
    return image_id

def cocoMerger(json_file1, json_file2, output_json):
    data1 = coco.COCO(json_file1).dataset
    data2 = coco.COCO(json_file2).dataset
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

    images = data1["images"] + data2["images"]
    annotations = data1["annotations"] + data2["annotations"]

    new_coco = {"info":info,"licenses":licenses,"categories":categories, "images":images, "annotations":annotations}

    with open(output_json, "w") as json_file:
        json.dump(new_coco, json_file)

def visualizeAnnot(seg):
    x = seg[::2]
    y = seg[1::2]
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((200, 200, 3)))  # Blank canvas
    plt.axis('off')
    poly = [(x[j], y[j]) for j in range(len(x))]
    polygon = Polygon(poly, edgecolor='r', facecolor='none', linewidth=2)
    plt.gca().add_patch(polygon)

def main():
    dir_json: 'str'
    save_json: 'str'
    images_path: ''
    patch_height: 'int'
    patch_width: 'int'
    if len(sys.argv) > 5:
        dir_json = sys.argv[1]
        save_json = sys.argv[2]
        images_path = sys.argv[3]
        patch_height = int(sys.argv[4])
        patch_width = int(sys.argv[5])
    else:
        print("Some of the parameters are missing")
        exit(-1)
    print(patch_width, patch_height)
    data = coco.COCO(dir_json)
    info = {}
    licenses = []
    images = []
    annotations = []
    categories = []

    info = {"year": "2023", "version": "10", "description": "Exported from roboflow.com", "contributor": "",
             "url": "https://public.roboflow.com/object-detection/undefined",
             "date_created": "2023-08-21T17:48:20+00:00"}
    licenses = [{"id":1,"url":"https://creativecommons.org/licenses/by/4.0/","name":"CC BY 4.0"}]

    categories.append({"id":0, "name":"holes", "supercategory":"none"})
    id = 0
    threshold = 12
    annot_id = 0
    files = utils.get_file_paths(images_path)
    files.sort()
    const = 0
    #print(files)
    id += const
    for i in tqdm(range(len(files))):
        image_id = findImageId(data.dataset, os.path.basename(files[i]))
        ann_ids = data.getAnnIds(imgIds=image_id, catIds=1)
        print(ann_ids)
        image = cv2.imread(files[i], cv2.IMREAD_GRAYSCALE)
        for y_start in range(0, 2000, patch_height):
            for x_start in range(0, 2400, patch_width):
                patch = image[y_start:y_start + patch_height, x_start:x_start + patch_width]
                if np.all(patch < threshold):
                #    print(True)
                    continue
                images.append({"id":id, "license":1, "file_name":"patch_" + str(id) + ".jpg", "height": patch_height, "width":patch_width,
                               "date_captured":"2023-08-21T17:48:20+00:00"})
                for ann_id in ann_ids:
                    segmentation = data.anns[ann_id]["segmentation"][0]
                    x_s = [segmentation[i] for i in range(0,len(segmentation), 2)]
                    #print(x_s)
                    y_s = [segmentation[i] for i in range(1,len(segmentation), 2)]
                    #print(y_s)
                    new_segmentation = []
                    for i in range(len(x_s)):
                        #print(x_s[i], y_s[i], x_start, y_start)
                        if (x_start <= x_s[i] < (x_start + patch_height)) and (y_start <= y_s[i] < (y_start + patch_height)):
                            #print(x_s[i], y_s[i], x_start, y_start)
                            new_segmentation.append(round(x_s[i] - x_start, 2))
                            new_segmentation.append(round(y_s[i] - y_start, 2))
                    if len(new_segmentation) == 0:continue
                    segmentation = new_segmentation
                    x_s = [segmentation[i] for i in range(0, len(segmentation), 2)]
                    y_s = [segmentation[i] for i in range(1, len(segmentation), 2)]
                    xmin = int(min(x_s))
                    ymin = int(min(y_s))
                    xmax = int(max(x_s))
                    ymax = int(max(y_s))
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                    area = polygon_area(x_s, y_s)
                    annot = {"id":annot_id, "image_id":id,
                             "category_id":0, "bbox":bbox,
                             "area":area,
                             "segmentation":[segmentation],
                             "iscrowd":0}
                    annot_id += 1

                    annotations.append(annot)

                id += 1
    new_coco = {"info":info,"licenses":licenses,"categories":categories, "images":images, "annotations":annotations}
    with open(save_json, "w") as json_file:
        json.dump(new_coco, json_file)

if __name__ == "__main__":
        main()

