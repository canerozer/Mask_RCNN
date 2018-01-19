import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
import time


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = os.path.join("/media/dontgetdown/model_partition/videos")

# Put the frames of videos to
#
frame_folder_names = os.listdir(IMAGE_DIR)
video_directories = []
video_names = []
for folder_name in frame_folder_names:
    assert os.path.isdir(os.path.join(IMAGE_DIR, folder_name))==True , "The image directory should only contain folders"
    video_names.append(folder_name)
    video_directories.append(IMAGE_DIR+"/"+folder_name)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def coco_to_voc_bbox_converter(y1, x1, y2, x2, roi_score):
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h, roi_score

# Start testifying images for every frame in a particular folder_name.
# When enumerator hits the batch size number, the model will begin detection.
video_counter = 0
for video_id, video_dir in enumerate(video_directories):
    print("Video in Process: {}/{}".format(video_id+1, len(video_directories)))
    image_list = []
    image_ids = os.listdir(os.path.join(IMAGE_DIR, video_dir))
    image_counter = 0
    sorted_image_ids = sorted(image_ids, key=lambda x: x[:-4])
    for d, image_id in enumerate(sorted_image_ids):
        print (image_id)
        if(image_id[-4:]==".jpg"):
            print(skimage.io.imread(os.path.join(video_dir, sorted_image_ids)))
            #image_list.append(skimage.io.imread(os.path.join(video_dir, sorted_image_ids)))
        if len(image_list)==config.BATCH_SIZE:
            print("Processed Frame ID: {}/{}".format(d+1, len(image_ids)))
            results = model.detect(image_list, verbose=1)
            r = results[0]
            image_list.clear()
            with open(MODEL_DIR+"/"+video_names[video_id], 'a+') as f:
                for score_id, scores in enumerate(r['scores']):
                    y1, x1, y2, x2 = r['rois'][score_id]
                    obj_score = r['scores'][score_id]
                    predicted_class_id = r['class_ids'][score_id]
                    x, y, w, h, score = coco_to_voc_bbox_converter(y1, x1, y2, x2, obj_score)
                    f.write(str(d+1)+", "+str(x)+", "+str(y)+", "+str(w)+", "+str(h)+", "+str(score)+", "+str(class_names[predicted_class_id])+"\n")
            print("")



# Visualize results
#print (image_id)
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                            class_names, r['scores'])
