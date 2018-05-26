"""
Custom sample usages: \
python3 xtracting_rpn_results_2.py \
--test-dataset-dir="/media/dontgetdown/model_partition/VOT_Subset/" \
--image-extension=".jpg"
"""

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import argparse
import coco
import utils
import model as modellib
import visualize
import time
import keras.backend as K
import tensorflow as tf

# Test some videos
parser = argparse.ArgumentParser(description='Test some videos.')
parser.add_argument('--test-dataset-dir', metavar='TD', type=str,
                    default="/media/dontgetdown/model_partition/OTB/",
                    help='enter the test directory')
parser.add_argument('--image-extension', metavar='CDC', type=str,
                    default=".jpg", help="type the codec of images")

args = parser.parse_args()

gpu_options = tf.GPUOptions(allow_growth=True)
# gpu_options = tf.GPUOptions(allocator_type = 'BFC')
config_keras = tf.ConfigProto(gpu_options=gpu_options)
K.set_session(tf.Session(config=config_keras))


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(args.test_dataset_dir)


frame_folder_names = os.listdir(IMAGE_DIR)
video_directories = []
video_names = []
for folder_name in frame_folder_names:
    assert os.path.isdir(os.path.join(IMAGE_DIR, folder_name)), (
        "The image directory should only contain folders")
    video_names.append(folder_name)
    video_directories.append(IMAGE_DIR+"/"+folder_name)

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80 + 1
    RPN_ANCHOR_STRIDE = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=LOGS_DIR, config=config)

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


def coco_to_voc_bbox_converter(y1, x1, y2, x2):
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((3, w, h), dtype=np.uint8)
    ret[0, :, :] = im
    ret[1, :, :] = im
    ret[2, :, :] = im
    return ret


# Start testifying images for every frame in a particular folder_name.
# When enumerator hits the batch size number, the model will begin detection.
video_counter = 0

# Number of clipped refined anchors to be extracted per frame is limited to 50.
limit = 50
for video_id, video_dir in enumerate(video_directories):
    print("Video in Process: {}/{}".format(video_id+1, len(video_directories)))
    print("Video Name: {}".format(video_dir))

    image_list = []
    image_ids = os.listdir(os.path.join(IMAGE_DIR, video_dir))
    image_counter = 0

    # Sort the images in folder and retrieve only jpg images
    sorted_image_ids = sorted(image_ids, key=lambda x: x[:-4])
    sorted_image_ids = list(filter(lambda x: args.image_extension in x,
                                   sorted_image_ids))

    for d, image_id in enumerate(sorted_image_ids):
        print (image_id)
        if(image_id[-4:] == args.image_extension):
            image = skimage.io.imread(os.path.join(video_dir, image_id))
            dims = image.shape

            # If image is BW, convert it to RGB for handling exception.
            if len(image.shape) == 2:
                image = to_rgb1(image)

            image_list.append(image)

            # Get the scale and padding parameters by using resize_image.
            _, _, scale, pad, _ = utils.resize_image(image,
                                                    min_dim=config.IMAGE_MIN_DIM,
                                                    max_dim=config.IMAGE_MAX_DIM,
                                                    min_scale=config.IMAGE_MIN_SCALE,
                                                    mode="square")

            # Roughly calculate padding across different axises.
            aver_pad_y = (pad[0][0] + pad[0][1])/2
            aver_pad_x = (pad[1][0] + pad[1][1])/2

        if len(image_list) == config.BATCH_SIZE:
            print("Processed Frame ID: {}/{}".format(d+1,
                                                     len(sorted_image_ids)))

            # Code taken from the iPython file, to retrieve the top anchors.
            time_start = time.time()

            pillar = model.keras_model.get_layer("ROI").output
            results = model.run_graph(image_list, [
                ("rpn_class", model.keras_model.get_layer("rpn_class").output),
                ("proposals", model.keras_model.get_layer("ROI").output),
                ("refined_anchors_clipped",
                    model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
            ])

            # Updating to the recent version of refined_anchors_clipped,
            # Normalized coordinates will be resized to 1024 x 1024
            r = results["refined_anchors_clipped"][0, :limit]
            r = r * np.array([1024, 1024, 1024, 1024])

            scores = ((np.sort(results['rpn_class'][:, :, 1]
                               .flatten()))[::-1])[:limit]

            print(time.time() - time_start)
            
            # A little bit of math for converting image dimensions 
            # from 1024 x 1024 to dim[0] x dim[1]

            r = (r - np.array((aver_pad_y, aver_pad_x,
                               aver_pad_y, aver_pad_x)))/scale

            # Clears the image list after evaluation
            image_list.clear()

            with open(LOGS_DIR+"/"+video_names[video_id]+"_rpn", 'a+') as f:
                for prop_id, proposals in enumerate(r):
                    y1, x1, y2, x2 = proposals
                    x, y, w, h = coco_to_voc_bbox_converter(y1, x1, y2, x2)
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if x + w > dims[1]:
                        w = dims[1] - x
                    if y + h > dims[0]:
                        h = dims[0] - y
                    things_to_write = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        format(image_id, '.8s'), prop_id+1,
                        format(x, '.2f'), format(y, '.2f'),
                        format(w, '.2f'), format(h, '.2f'),
                        format(scores[prop_id], '.8f'))
                    f.write(things_to_write)
            r = None
            print("")
