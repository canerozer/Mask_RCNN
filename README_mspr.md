## To Do List After Cloning This Repository

#### 1) Make a directory name called "logs"
#### 2) All the scripts have to be run inside the project folder.

## Python Scripts Available to Run:
### 1) evaluation.py

--> Extracts the bounding box coordinates of the objects using the head of the
network in a text file. The text file also contains the objectness score and class label
of the objects.

Takes --test-dataset-dir argument as the input, which provides
the videos as video frames in separate folders. It will output the results to the

Sample terminal command: python3 evaluation.py --test-dataset-dir /path/to/dataset

Sample dataset folder:
dataset/
|-> test_video_1/
  |-> 1.jpg
  |-> 2.jpg
  .
|-> test_video_2/
  |-> 1.jpg
  |-> 2.jpg
  .
.

Sample output:
logs/
|-> test_video_1_mask
|-> test_video_2_mask
.

For each head output, the order of output arguments are such that:
$frame_id, $left_top_x, $left_top_y, $width, $height, $objectness_score, $class_id

#### 2) evaluation.py *
--> Extracts the RPN output of the network given the images to a text file. For
each frame, RPN will create 50 proposals with its bounding box coordinate
information and objectness score. Using the same dataset folder structure, the
output is going to be as below:

Sample output:
logs/
|-> test_video_1_rpn
|-> test_video_2_rpn
.

Inside each text file, the outputs are ordered such that:
$proposal_id, $left_top_x, $left_top_y, $width, $height, $objectness_score
Since it is guaranteed that RPN will output 50 proposals for each frame, I did
not provide any specific information for the frame number.

* --> The program may terminate itself due to a CUDA memory issue.

#### 3)
