import visualize
import argparse
import os

parser = argparse.ArgumentParser(description='Draw bounding boxes on videos.')
parser.add_argument('--test-dataset-dir', metavar='TD', type=str,
                    default="/home/mspr/Datasets/test_dataset",
                    help='enter the test dataset directory')
parser.add_argument('--test-bbox-results', metavar='TB', type=str,
                    help='enter the directory containing files for \
                    final bounding boxes for each video')
parser.add_argument('--output-type', metavar='T', type=str,
                    help='define the type of output as ''rpn'' for region proposals\
                    and ''mask'' for head outputs')

args = parser.parse_args()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory for importing bounding boxes for each video
BBOX_DIR = args.test_bbox_results

# Directory for videos as video frames
VIDEO_DIR = os.path.join(args.test_dataset_dir)

video_names = os.listdir(VIDEO_DIR)
for video_name in video_names:
    print(video_name)
