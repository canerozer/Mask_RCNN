import os
import numpy as np
import cv2
import threading
import time
import logging


class Analysis:
    def __init__(self, datasets_dir, ground_truth_filename, preds_dir=None, dataset_name=None):
        self.datasets_dir = datasets_dir
        self.ground_truth_filename = ground_truth_filename
        self.preds_dir = preds_dir
        self.dataset_name = dataset_name
        logging.basicConfig(filename='analysis_{}.log'.format(self.dataset_name),level=logging.DEBUG,
                            format='%(levelname)s:%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p',
                            filemode='w')

    def video_player(self, show_pred_boxes=True, show_gt_boxes=True):
        """
        Shows the image frames of different videos in a specific directory.
        Uses the ground truth and predicted bounding boxes and shows them in the figure.
        """

        ic_list = self.image_reader()

        if show_gt_boxes:
            gt_list = self.gt_reader()
        if show_pred_boxes:
            pred_list = self.pred_reader()

        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        for n, ic in enumerate(ic_list):
            for d, frame in enumerate(ic):
                time_begin = time.time()
                print("Currently read frame number: {}".format(d))
                image = cv2.imread(frame)

                if show_gt_boxes:
                    for k, gt in enumerate(gt_list[n]):
                        if d == gt[0]:
                            try:
                                gt_pt1, gt_pt2 = tuple(gt[1:3]), tuple(gt[3:5])
                                cv2.rectangle(image, gt_pt1, gt_pt2, color=(0,255,0), thickness=2)
                            except:
                                pass

                if show_pred_boxes:
                    assert self.preds_dir != None
                    for k, pred in enumerate(pred_list[n]):
                        if d == pred[0]:
                            pred_pt1, pred_pt2 = tuple(pred[1:3]), tuple(pred[3:5])
                            cv2.rectangle(image, pred_pt1, pred_pt2, color=(0,0,255), thickness=2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(image, video_names[n], fontFace=font, fontScale=0.5)
                cv2.putText(image, video_names[n], (0, 20), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('display', image)
                cv2.waitKey(1)
                delay = float(time.time() - time_begin)
                print("FPS: {}".format(1/(delay)))
                
                if(delay < 0.15):
                    time.sleep(0.15 - delay)
    
    def false_alarm_plot(self):
        pass

    def retreiver(self, duration, preds_all, gt_all, video_id, dict_ref):
        for frame_id in range(duration):
            if (frame_id%100 == 0) or (frame_id+1 == duration):
                print("{}/{} is complete.".format(frame_id+1, duration))
            preds = self.retreive_preds(preds_all, video_id, frame_id+1)
            gt = gt_all[video_id][frame_id]
            iou_frame_all_pred = []
            for pred in preds:
                try:
                    iou_for_pred = (self.iou(pred[1:5], gt[1:5]), pred[5])
                except:
                    iou_for_pred = (None, pred[5])
                iou_frame_all_pred.append(iou_for_pred)
            # iou_frame_all_pred = sorted(iou_frame_all_pred, reverse=True)
            dict_ref.append(iou_frame_all_pred)

    def tester(self, iou_thresholds=[0.25, 0.5, 0.75]):
        """
        
        """
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)
        frame_lengths = [len(length) for length in self.image_reader()]
        self.dict_iou = {video_name: [] for video_name in video_names}
        
        preds_all = self.pred_reader()
        gt_all = self.gt_reader()

        if self.dataset_name == "VOT2018_LT_Subset" or self.dataset_name == "VOT2018_LT":
            target_classes = {'bicycle': 1, 'car9': 3}

        elif self.dataset_name == "VOT2016_Subset":
            target_classes = {'ball1': 33, 'ball2': 33, 'basketball': 1, 'birds1': kus,
                    'birds2': kus, 'bianket': 1, 'bmx': 1, 'bolt1': 1, 'bolt2'}

        #############################################################################
        ### TEST 1 
        ### How many GT entries are defined as nan?
        ### E.g: When the object is not present in the scene.
        #############################################################################
        logging.info("#############################################################################")
        logging.info("TEST 1: How many GT entries were annotated as nan?")
        isnan_list = []
        for video_id, video_name in enumerate(video_names):
            nan_counter = 0
            frame_ids = []
            for frame_id in range(frame_lengths[video_id]):
                if np.isnan(gt_all[video_id][frame_id][1]):
                    frame_ids.append(frame_id)
                    nan_counter+=1
            isnan_list.append(frame_ids)
            logging.info("{}/{} frames were annotated as 'nan' in video: {}.".format(nan_counter,\
                    frame_lengths[video_id], video_name))
        logging.info("TEST 1 complete.")
        logging.info("#############################################################################")

        logging.info("#############################################################################")
        logging.info("TEST 2: For the frames where there is no GT defined, how many of "+\
                "them have at least 1 prediction?")

        #############################################################################
        ### TEST 2 
        ### Among the frames in videos where the corresponding GT entries are not 
        ### defined, are there any predictions? If so, what is the count of them as
        ### frames?
        ###     If there are, does the prediction bbox exceed the IoU threshold?
        ###     If it is over the threshold, does the label of objects inside
        ###          bounding boxes have the same 
        ### E.g: When the object is not present in the scene.
        #############################################################################
        
        pred_histogram_video = []
        for video_id, video_name in enumerate(video_names):
            preds_when_nan_valid = [0]*81
            for pred_id, pred in enumerate(preds_all[video_id]):
                for nan_frame_id in isnan_list[video_id]:
                    if pred[0] == nan_frame_id:
                        preds_when_nan_valid[pred[5]]+=1
            pred_histogram_video.append(preds_when_nan_valid)
            logging.info("Video Name: {} {}".format(video_name, preds_when_nan_valid))

        logging.info("TEST 2 complete.")
        logging.info("#############################################################################")

        threads = []

        for video_id, video_name in enumerate(video_names):
            dict_ref = self.dict_iou[video_name]
            t = threading.Thread(target=self.retreiver, args=(frame_lengths[video_id],
                            preds_all, gt_all, video_id, dict_ref))
            threads.append(t)
            logging.debug("IoU calculation process for '{}' has begun.".format(video_name))

            # for frame_id in range(frame_lengths[video_id]):
            #     preds = self.retreive_preds(preds_all, video_id, frame_id+1)
            #     gt = gt_all[video_id][frame_id]
            #     iou_frame_all_pred = []
            #     for pred in preds:
            #         try:
            #             iou_for_pred = (self.iou(pred[1:5], gt[1:5]), pred[5])
            #         except:
            #             iou_for_pred = (None, pred[5])
            #         iou_frame_all_pred.append(iou_for_pred)
            #     iou_frame_all_pred = sorted(iou_frame_all_pred, reverse=True)
            #     self.dict_iou[video_name].append(iou_frame_all_pred)

        for d, thread in enumerate(threads):
            thread.start()

        for thread in threads:
            thread.join()

        logging.debug("All threads have been successfully suspended.")

        #############################################################################
        ### TEST 3 
        ### By using the IoU thresholds and IoU rates, this part will separate if our 
        ### bounding boxes are positive or negative.
        #############################################################################

        logging.info("#############################################################################")
        logging.info("TEST 3: Separation of bounding boxes as positive or negative based"+\
                     "on their IoU's.")

        for video_id, video_name in enumerate(video_names):
            # different_iou_video_based_conf = []
            for iou_threshold in iou_thresholds:
                iou_based_conf = []
                counter_single_pred = 0
                counter_multi_pred = 0
                tp_single = 0
                fn_single_1 = 0
                fn_single_2 = 0
                fp_single = 0
                tn_single = 0
                tp_multi = 0
                fp_multi = 0
                fn_multi = 0
                tn_multi = 0
                for d, preds in enumerate(self.dict_iou[video_name]):
                    conf_matrix_items = []
                    if len(preds)==1:
                        counter_single_pred += 1
                        if preds[0][0] == None:
                            conf_matrix_items.append((d, "False", "Positive", 1))
                            fn_single_1 += 1
                            # logging.info((d, "False", "Negative", 1))
                        elif preds[0][0]>iou_threshold:
                            if preds[0][1]==target_classes[video_name]:
                                conf_matrix_items.append((d, "True", "Positive"))
                                tp_single += 1
                                # logging.info((d, "True", "Positive"))
                            else:
                                conf_matrix_items.append((d, "False", "Positive", 1))
                                fp_single += 1
                                # logging.info((d, "False", "Positive", 1))
                        elif preds[0][0]<iou_threshold:
                            if preds[0][1]==target_classes[video_name]:
                                conf_matrix_items.append((d, "True", "Negative", 1))
                                tn_single += 1
                                # logging.info((d, "True", "Negative", 1))
                            else:
                                conf_matrix_items.append((d, "False", "Negative", 2))
                                fn_single_2 += 1
                                # logging.info((d, "False", "Negative", 2))
                    
                    elif len(preds)>1:
                        idx = []
                        counter_multi_pred+=1
                        for n, pred in enumerate(preds):
                            if pred[0] == None:
                                conf_matrix_items.append((d, "False", "Positive", 1))
                                fn_multi += 1
                                # logging.info((d, "False", "Negative", 1))
                            # Obtain the indices which might be true positive bboxes
                            elif pred[0]>iou_threshold:
                                if pred[1]==target_classes[video_name]:
                                    idx.append(n)
                        
                        for n, pred in enumerate(preds):
                            counter=0
                            if any(k==n for k in idx):
                                if counter==0:
                                    conf_matrix_items.append((d, "True", "Positive"))
                                    tp_multi += 1
                                    # logging.info((d, "True", "Positive"))
                                    counter+=1
                                else:
                                    conf_matrix_items.append((d, "True", "Negative", 2))
                                    tn_multi += 1
                                    # logging.info((d, "True", "Negative", 2))
                            else:
                                conf_matrix_items.append((d, "False", "Positive", 2))
                                fp_multi += 1
                                # logging.info((d, "False", "Positive", 2))

                    iou_based_conf.append(conf_matrix_items)
                logging.info("VName: {} IOU: {} True Positives: {}".format(video_name, iou_threshold, tp_multi+tp_single))
                logging.info("VName: {} IOU: {} True Negatives: {}".format(video_name, iou_threshold, tn_multi+tn_single))
                logging.info("VName: {} IOU: {} False Positive: {}".format(video_name, iou_threshold, fp_multi+fp_single))
                logging.info("VName: {} IOU: {} False Negatives: {}".format(video_name, iou_threshold, fn_single_1+fn_single_2+fn_multi))

                # different_iou_video_based_conf.append(iou_based_conf)
                
            logging.info("Number of multiple predictions for {}: {}".format(video_name, counter_multi_pred))
            logging.info("Number of single predictions for {}: {}".format(video_name, counter_single_pred))
            logging.info("Total number of frames for {} is: {}".format(video_name, frame_lengths[video_id]))
        logging.info("TEST 3 complete.")
        logging.info("#############################################################################")


    def iou(self, bbox1, bbox2):
        """
        Calculates the IoU of 2 bounding boxes.

        Parameters:   bbox1, bbox2: list or numpy array of bounding box coordinates.
        The input should contain the top-left corner's x and y coordinates and 
        width and height of the bounding boxes.
        
        Assertations: width and height informations of bbox1 and bbox2 should be 
        larger than 0.
        
        Returns:      iou: A floating point decimal representing the IoU ratio, which
        is the division of bounding box areas of intersection to their union.
        """
        x1, y1, x1_t, y1_t = bbox1
        w1 = x1_t - x1
        h1 = y1_t - y1
        x2, y2, x2_t, y2_t = bbox2
        w2 = x2_t - x2
        h2 = y2_t - y2

        assert w1 and w2 > 0
        assert w1 and h2 > 0
        
        iou = 0
        if (((x1>x2 and x1<x2+w2) or (x1+w1>x2 and x1+w1<x2+w2) or 
            (x2>x1 and x2<x1+w1) or (x2+w2>x1 and x2+w2<x1+w1)) and 
            ((y1>y2 and y1<y2+h2) or (y1+h1>y2 and y1+h1<y2+h2) or
            (y2>y1 and y2<y1+h1) or (y2+h2>y1 and y2+h2<y1+h1))):
            iou_xmin = float(max(x1, x2))
            iou_xmax = float(min(x1+w1, x2+w2))
            iou_ymin = float(max(y1, y2))
            iou_ymax = float(min(y1+h1, y2+h2))
            intersection_area = (iou_ymax - iou_ymin)*(iou_xmax - iou_xmin)
            total_area = float(w1)*float(h1) + float(w2)*float(h2) - intersection_area
            iou = intersection_area/total_area
        return iou

    def retreive_preds(self, preds, video_id, frame_id):
        output = []
        for pred_frame in preds[video_id]:
            if pred_frame[0] == frame_id:
                output.append(pred_frame)
        return output


    def image_reader(self):
        ic_list = []
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        if self.dataset_name != "MSPR_Dataset":
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
        else:
            for video in video_names:
                frame_names = os.listdir(os.path.join(
                    self.datasets_dir, video, "Images"))
                frame_names = filter(lambda x: x[-4:] == ".jpg",
                    frame_names)
                frame_names = sorted(frame_names, key=
                    lambda x: int(x[:-4]))
                frame_names = [os.path.join(self.datasets_dir, video, "Images",
                                frame_name) for frame_name in frame_names]
                ic_list.append(frame_names)
        return ic_list
                
    def gt_reader(self):
        ground_truths = []
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        for video in video_names:
            video_gts = []
            if self.dataset_name == "VOT2018_LT_Subset" or self.dataset_name == "VOT2018_LT":

                gt_path = os.path.join(self.datasets_dir, video, self.ground_truth_filename)
                with open(gt_path, 'r') as f:
                    temp_input_lines = f.read().split("\n")[:-1]

                logging.debug("Parsing ground truths of {} as VOT2018 format.".format(video))
                for d, line in enumerate(temp_input_lines):
                    try:
                        x, y, w, h = map(float, line.split(","))
                    except:
                        logging.error("GT for {} in {} dataset couldn't have been parsed.".format(
                                        video, self.dataset_name))
                    video_gts.append(np.array([d+1, x, y, x+w, y+h], dtype=np.float32))

            elif self.dataset_name == "VOT2016_Subset":

                gt_path = os.path.join(self.datasets_dir, video, self.ground_truth_filename)
                with open(gt_path, 'r') as f:
                    temp_input_lines = f.read().split("\n")[:-1]

                logging.debug("Parsing ground truths of {} as VOT2016 format.".format(video))
                for d, line in enumerate(temp_input_lines):
                    try:
                        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split(","))
                        x, y, w, h = self.vot16_to_18(x1, y1, x2, y2, x3, y3, x4, y4)
                    except:
                        logging.error("GT for {} in {} dataset couldn't have been parsed.".format(video, self.dataset_name))
                    video_gts.append(np.array([d+1, x, y, x+w, y+h], dtype=np.float32))

            elif self.dataset_name == "MSPR_Dataset":

                gt_path = os.path.join(self.datasets_dir, video, "Images", self.ground_truth_filename)
                with open(gt_path, 'r') as f:
                    temp_input_lines = f.read().split("\n")[:-1]

                logging.debug("Parsing ground truths of {} as MSPR Dataset format.".format(video))
                for line in temp_input_lines:
                    try:
                        d, x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split(","))
                        x, y, w, h = self.vot16_to_18(x1, y1, x2, y2, x3, y3, x4, y4)
                    except:
                        logging.error("GT for {} in {} dataset couldn't have been parsed.".format(video, self.dataset_name))
                    video_gts.append(np.array([d, x, y, x+w, y+h], dtype=np.float32))
                
            else:
                logging.warning("{} not found.".format(self.dataset_name))

                
            ground_truths.append(video_gts)
        return ground_truths
    
    def pred_reader(self):
        preds_all = []
        video_names = os.listdir(self.preds_dir)
        video_names = sorted(video_names)

        for video in video_names:
            logging.debug("Parsing predictions of {}".format(video))
            pred_sample_path =  os.path.join(self.preds_dir, video)
            video_preds = []
            with open(pred_sample_path, 'r') as f:
                temp_input_lines = f.read().split("\n")[:-1]
            for line in temp_input_lines:
                fr_id, x, y, w, h, _, label = map(float, line.split("\t"))
                video_preds.append(np.array([fr_id, x, y, x+w, y+h, label], dtype=np.int16))
            preds_all.append(video_preds)
        return preds_all
        
    def vot16_to_18(self, x1, y1, x2, y2, x3, y3, x4, y4):
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        w = xmax - xmin
        h = ymax - ymin
        return xmin, ymin, w, h

if __name__ == "__main__":
    # test-case 1
    # dataset_name = "VOT2016_Subset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2016_final_th0/"

    # test-case 2
    # dataset_name = "VOT2018_LT_Subset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2018_Subsets/"+\
    #         "MASK_VOT2018_final_th0.01_detnms_th0.4/"

    # test-case 3
    # dataset_name = "VOT2018_LT"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2018_final_th0.001/"

    # ozan case
    dataset_name = "MSPR_Dataset"
    datasets_dir = "Datasets/"+dataset_name+"/"
    ground_truth_file_name = "groundtruth.txt"

    analise = Analysis(datasets_dir, ground_truth_file_name,
                        dataset_name=dataset_name)
    analise.video_player(show_pred_boxes=False)
    # analise.tester()
