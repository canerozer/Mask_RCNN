"""
Custom usage: python3 evaluation.py --mode="inference" --test-dataset-dir="Datasets/VOT2016/"\
--model-dir="logs/
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
from config import Config
from utils import particle_array_const


# Test some videos
parser = argparse.ArgumentParser(description='Test some videos.')
parser.add_argument('test_dataset_dir', metavar='TD', type=str,
                    default="/home/mspr/Datasets/test_dataset",
                    help='enter the test directory')
parser.add_argument("--mode", required=True,
                    metavar="<mode>",
                    help="Indicate the model mode as 'inference' or"
                         "'extension' or 'tavot'.")
parser.add_argument('--model-dir', metavar='MD', type=str,
                    default=None,
                    help='enter the test directory')
parser.add_argument('--particles-dir', metavar='PD', type=str,
                    default=None,
                    help='folder directory for importing particle filter'
                         ' proposals when the mode is set to extension')
parser.add_argument("--segment", default=False, type=utils.str2bool,
                    metavar="<segment>",
                    help="Save segmentation results for each detection")
parser.add_argument("--tau", default=0.3, type=float,
                    metavar="<tau>",
                    help="IoU thr of LF block")

args = parser.parse_args()


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
assert args.model_dir is not None
COCO_MODEL_PATH = args.model_dir

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(args.test_dataset_dir)


video_directories = []
video_names = []
frame_folder_names = sorted(os.listdir(IMAGE_DIR))
for folder_name in frame_folder_names:
    assert os.path.isdir(os.path.join(IMAGE_DIR, folder_name)), (
    "The image directory should only contain folders")
    video_names.append(folder_name)
    video_directories.append(os.path.join(IMAGE_DIR, folder_name))

# Directory for particles should be entered
#when the mode is extension.
PARTICLE_DIR = args.particles_dir
if args.mode == "extension":
    assert PARTICLE_DIR is not None, "Present the directory for particles."

    # Check for if particles for all videos are present.
    particles_videoname = sorted(os.listdir(PARTICLE_DIR))
    videonames = ([x.split("_", maxsplit=3)[0] for x in particles_videoname])

    assert frame_folder_names == videonames, "Some particle files or videos "\
                                            "are missing"

    particles_full_path = [os.path.join(PARTICLE_DIR, x) for x in
                           particles_videoname]
    # Importing text files to construct numpy arrays


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    global args
    NAME = "coco_evaluation"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 80 + 1
    #NUM_CLASSES = 1 + 1

    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_NMS_THRESHOLD = 0.7
    FILTER_BACKGROUND = True

    if args.mode == "extension":
        #POST_PS_ROIS_INFERENCE = 400
        POST_PS_ROIS_INFERENCE = 1000
        #DETECTION_MAX_INSTANCES = 400
        DETECTION_MAX_INSTANCES = 1000
        IOU_THR = args.tau
    elif args.mode == "inference":
        POST_PS_ROIS_INFERENCE = 1000
        DETECTION_MAX_INSTANCES = 1000
        
    INIT_BN_BACKBONE = True    
    INIT_GN_BACKBONE = False
    INIT_BN_HEAD = True    
    INIT_GN_HEAD = False

    #USE_BOTTOM_UP_AUG = False # Bottom-up augumentation setting
    #LATERAL_SHORTCUTS = False # Green and red dash connections                               
    #FC_MASK_FUSION = False # Mask fusion setting

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode=args.mode, model_dir=LOGS_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', '	bird',
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
               'teddy bear', 'hair drier', 'toothbrush', 'face']


def coco_to_voc_bbox_converter(y1, x1, y2, x2, roi_score):
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h, roi_score

def voc_to_coco_bbox_converter(arr):
    x, y, w, h = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    x2 = x + w
    y2 = y + h
    new_arr = np.array((y, x, y2, x2)).transpose((1, 2, 0))
    return new_arr

def to_rgb1(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((3, w, h), dtype=np.uint8)
    ret[0, :, :] = im
    ret[1, :, :] = im
    ret[2, :, :] = im
    return ret

def softmax(arr):
    maximum_nr = np.max(arr, axis=1)
    arr = arr - maximum_nr[:, np.newaxis]
    probs = np.exp(arr)/(np.sum(np.exp(arr), axis=1)[:, np.newaxis])
    return probs

# Start testifying images for every frame in a particular folder_name.
# When enumerator hits the batch size number, the model will begin detection.
video_counter = 0

for video_id, video_dir in enumerate(video_directories):
    video_name = video_names[video_id]
    if args.mode == "extension":
        particles = particle_array_const(particles_full_path[video_id],
                                         os.path.join(video_dir, os.listdir(video_dir)[0]),
                                         config=config)
    with open(LOGS_DIR+"/"+"TAVOT_tau"+str(args.tau)+
               "/"+video_name+"_tavot_LF_nms07", 'a+') as f:
        #f.write("fn\tx\ty\tw\th\tobj_score\tlbl\tc1\tconf_1\t\tc2\tconf_2\t\tc3\tconf_3\t\tc4\tconf_4\t\tc5\tconf_5\n")
        print("Video in Process: {}/{}".format(video_id+1, len(video_directories)))
        print("Video name: {}".format(video_dir))
        image_list = []
        print(IMAGE_DIR)
        print("")
        print(video_dir)

        image_counter = 0

        image_ids = os.listdir(video_dir)
        sorted_image_ids = sorted(image_ids, key=lambda x: x[:-4])
        sorted_image_ids = list(filter(lambda x:  x[-4:] == ".jpg", sorted_image_ids))

        for d, image_id in enumerate(sorted_image_ids):
            print (image_id)
            if(image_id[-4:]==".jpg"):
                #print(skimage.io.imread(os.path.join(video_dir, image_id)))
                image = skimage.io.imread(os.path.join(video_dir, image_id))
                if len(image.shape) == 2:
                    image = to_rgb1(image)
                image_list.append(image)

            if len(image_list)==config.BATCH_SIZE:
                print("Processed Frame ID: {}/{}".format(d+1, len(sorted_image_ids)))
                if args.mode == "extension":
                    results = model.detect(image_list, verbose=1, particles=particles[d])
                elif args.mode == "inference":
                    results = model.detect(image_list, verbose=1)
                elif args.mode == "tavot":
                    results = model.detect(image_list, verbose=1, prev_output=prev_output)
                r = results[0]
                image_list.clear()
                for score_id, scores in enumerate(r['scores']):
                    y1, x1, y2, x2 = r['rois'][score_id]
                    obj_score = r['scores'][score_id]
                    predicted_class_id = r['class_ids'][score_id]
                    # Max top-5 probabilities
                    probs = -np.sort(-r['logits'][score_id])[:5]

                    # Arguments of top-5
                    p_c_ids = np.argsort(-r['logits'][score_id])[:5]

                    # When # of classes are less than 5
                    if config.NUM_CLASSES < 5:
                        diff = 5 - config.NUM_CLASSES
                        predicted_class_id = np.pad(predicted_class_id, (0,diff), 'constant', constant_values=0)
                        probs = np.pad(probs, (0,diff), 'constant', constant_values=0)
                        p_c_ids = np.pad(p_c_ids, (0,diff), 'constant', constant_values=0)

                        # When backgrounds are not filtered, the objectness score still needs to remain consistent.

                    if predicted_class_id == 0:
                        score = probs[1]
                        predicted_class_id = p_c_ids[1]
                    x, y, w, h, score = coco_to_voc_bbox_converter(y1, x1, y2, x2, obj_score)

                    things_to_write = "{}\t\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n"\
                                      .format(image_id[:-4], x, y, w, h, format(score, '.8f'),
                                      predicted_class_id, p_c_ids[0],
                                    format(probs[0], '.8f'), p_c_ids[1],
                                    format(probs[1], '.8f'), p_c_ids[2],
                                    format(probs[2], '.8f'), p_c_ids[3],
                                    format(probs[3], '.8f'), p_c_ids[4],
                                    format(probs[4], '.8f'))
                    f.write(things_to_write)
                    if args.segment:
                        dr = r['masks'][:,:,score_id] * 255
                        out_fn = image_id[:-4] + "_mask_" +\
                                 str(score_id) + ".png"
                        out_dirpath = os.path.join(LOGS_DIR, video_name, out_fn)
                        utils.mkdir_ifnotfound(os.path.dirname(out_dirpath))
                        skimage.io.imsave(out_dirpath, dr)

                print("")
