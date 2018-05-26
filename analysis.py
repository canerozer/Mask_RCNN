import os
import numpy as np
import cv2
import threading
import time


class Analysis:
    def __init__(self, datasets_dir, ground_truth_filename, preds_dir):
        self.datasets_dir = datasets_dir
        self.ground_truth_filename = ground_truth_filename
        self.preds_dir = preds_dir

    def false_alarm_plot(self):
        pass

    def video_player(self):
        """
        Shows the image frames of different videos in a specific directory.
        """

        ic_list = self.image_reader()
        gt_list = self.gt_reader()
        pred_list = self.pred_reader()

        for n, ic in enumerate(ic_list):
            for d, frame in enumerate(ic):
                time_begin = time.time()
                print("Currently read frame number: {}".format(d))
                image = cv2.imread(frame)

                try:
                    gt_pt1, gt_pt2 = self.rect_point_gen(gt_list[n][d])
                except:
                    pass

                # print(pred_list[n][0])
                for k, pred in enumerate(pred_list[n]):
                    if d == pred[0]:
                        pred_pt1, pred_pt2 = tuple(pred[1:3]), tuple(pred[3:5])
                        cv2.rectangle(image, pred_pt1, pred_pt2, color=(0,0,255), thickness=2)

                cv2.rectangle(image, gt_pt1, gt_pt2, color=(0,255,0), thickness=2)
                cv2.imshow('display', image)
                cv2.waitKey(1)
                delay = float(time.time() - time_begin)
                print("FPS: {}".format(1/(delay)))
        
    def image_reader(self):
        ic_list = []
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        for video in video_names:
            frame_names = os.listdir(os.path.join(
                self.datasets_dir, video))
            frame_names = filter(lambda x: x[-4:] == ".jpg",
                frame_names)
            frame_names = sorted(frame_names, key=
                lambda x: int(x[:-4]))
            frame_names = [os.path.join(self.datasets_dir, video, 
                            frame_name) for frame_name in frame_names]

            ic_list.append(frame_names)
        return ic_list
            
        # filter(, content)
        # io.imread_collection()
    
    def gt_reader(self):
        ground_truths = []
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        for video in video_names:
            gt_path = os.path.join(self.datasets_dir, video, self.ground_truth_filename)
            with open(gt_path, 'r') as f:
                ground_truths.append(f.read().split())
        self.ground_truths = ground_truths
        return ground_truths

    def rect_point_gen(self, str_coords):
        x, y, w, h = map(float, str_coords.split(","))
        return (int(x), int(y)), (int(x+w), int(y+h))
    
    def pred_reader(self):
        video_names = os.listdir(self.preds_dir)
        video_names = sorted(video_names)
        preds_all = []
        for n, video in enumerate(video_names):
            pred_sample_path =  os.path.join(self.preds_dir, video)
            video_preds = []
            with open(pred_sample_path, 'r') as f:
                temp_input_lines = f.read().split("\n")[:-1]
            for line in temp_input_lines:
                fr_id, x, y, w, h, _, _ = map(float, line.split("\t"))
                video_preds.append(np.array([fr_id, x, y, x+w, y+h], dtype=np.int16))
            preds_all.append(video_preds)
        return preds_all
        
    def iou(self):
        pass


if __name__ == "__main__":
    datasets_dir = "Datasets/VOT2018_LT_Subset/"
    ground_truth_file_name = "groundtruth.txt"
    preds_dir = "logs/Evaluations/MASK_VOT2018_Subsets/"+\
                "MASK_VOT2018_final_th0.7_detnms_th0.4/"
    analise = Analysis(datasets_dir, ground_truth_file_name,
                        preds_dir)
    # analise.video_player()
    analise.video_player()
