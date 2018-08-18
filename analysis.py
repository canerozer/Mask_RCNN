import os
import numpy as np
import cv2
import threading
import time
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools


class Analysis:
    def __init__(self, datasets_dir, ground_truth_filename, number_top,
                 log_analysis, debug=True, record_nones=False,
                 heed_singles=False, heed_multiples=False, all_tp=False,
                 preds_dir=None, dataset_name=None,
                 counter_or_objectness='counter'):

        self.datasets_dir = datasets_dir
        self.ground_truth_filename = ground_truth_filename
        self.number_top = number_top
        self.log_analysis = log_analysis
        self.debug = debug
        self.record_nones = record_nones
        self.heed_singles = heed_singles
        self.heed_multiples = heed_multiples
        self.preds_dir = preds_dir
        self.dataset_name = dataset_name
        self.all_tp = all_tp
        self.counter_or_objectness = counter_or_objectness

        if not os.path.isdir(self.log_analysis):
            os.mkdir(self.log_analysis)

        if self.debug:
            logging.basicConfig(filename=log_analysis + 'analysis_{}.log'.format(
                                self.dataset_name), level=logging.DEBUG,
                                format='%(levelname)s:%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S%p', filemode='w')
            logging.debug("Analysis Started")

        assert (self.counter_or_objectness == 'counter' or
                self.counter_or_objectness == 'objectness')



    def video_player(self, show_pred_boxes=True, show_gt_boxes=True):
        """
        Shows the image frames of different videos in a specific directory.
        Uses the ground truth and predicted bounding boxes and shows them in
        the figure.
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
                                cv2.rectangle(image, gt_pt1, gt_pt2, color=(
                                    0, 255, 0), thickness=2)
                            except:
                                pass

                if show_pred_boxes:
                    # assert self.preds_dir is not None
                    for k, pred in enumerate(pred_list[n]):
                        if d == pred[0][0]:
                            pred_pt1 = tuple((pred[0][1], pred[0][2]))
                            pred_pt2 = tuple((pred[0][3], pred[0][4]))
                            cv2.rectangle(image, pred_pt1, pred_pt2,
                                          color=(0, 0, 255), thickness=2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, video_names[n], (0, 20), font, 0.8,
                            (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Fr: {}".format(d), (0, 35), font, 0.4,
                            (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow('display', image)
                cv2.waitKey(1)
                delay = float(time.time() - time_begin)
                print("FPS: {}".format(1 / delay))

                if(delay < 0.15):
                    time.sleep(0.15 - delay)

    def retreiver(self, duration, preds_all, gt_all, video_id, dict_ref):
        for frame_id in range(duration):

            if (frame_id % 100 == 0) or (frame_id + 1 == duration):
                print("{}/{} is complete.".format(frame_id + 1, duration))

            framewise_preds = self.ret_preds(preds_all, video_id, frame_id + 1)
            gt = gt_all[video_id][frame_id]
            iou_frame_all_pred = []
            for pred in framewise_preds:
                try:
                    iou_for_pred = (self.iou(pred[1:5], gt[1:5]), *pred[5:])
                except:
                    iou_for_pred = (None, *pred[5:])
                iou_frame_all_pred.append(iou_for_pred)
            # iou_frame_all_pred = sorted(iou_frame_all_pred, reverse=True)
            dict_ref.append(iou_frame_all_pred)

    def ret_preds(self, preds, video_id, frame_id):
        output = []
        for pred_frame in preds[video_id]:
            if pred_frame[0][0] == frame_id:
                output.append(list(pred_frame[0]))
        return output

    def test1(self):
        """
        Testing for determining how many of the GT entries are described as nan.
        E.g: When the object is not present in the scene.
        Simply, the first coordinate x will be checked as nan or not.
        """
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)
        frame_lengths = [len(length) for length in self.image_reader()]

        gt_all = self.gt_reader()

                #############################################################################
        ### TEST 1 
        ### How many GT entries were annotated as nan?
        ### E.g: When the object is not present in the scene.
        ### Simply, the first coordinate x will be checked as nan or not.
        #############################################################################
        logging.info("#############################################################################")
        logging.info("TEST 1: How many GT entries were annotated as nan?")
        self.isnan_list = []
        for video_id, video_name in enumerate(video_names):
            nan_counter = 0
            frame_ids = []
            for frame_id in range(frame_lengths[video_id]):
                if np.isnan(gt_all[video_id][frame_id][1]):
                    frame_ids.append(frame_id)
                    nan_counter+=1
            self.isnan_list.append(frame_ids)
            logging.info("{}/{} frames were annotated as 'nan' in video: {}.".format(nan_counter,\
                    frame_lengths[video_id], video_name))
        logging.info("TEST 1 complete.")
        logging.info("#############################################################################")

    def test2(self):
        """
        Finds how many of the predictions does not have any corresponding GT entries.
        This definition will write to file on a class basis using isnan_list from self.test1().
        In this case, self.test1() has to be called to create the self.isnan_list attrib.
        """

        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)
        
        preds_all = self.pred_reader()

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
            for pred in preds_all[video_id]:
                for nan_frame_id in self.isnan_list[video_id]:
                    if pred[0][0] == nan_frame_id:
                        preds_when_nan_valid[pred[0][5]]+=1
            pred_histogram_video.append(preds_when_nan_valid)
            logging.info("Video Name: {} {}".format(video_name, preds_when_nan_valid))

        logging.info("TEST 2 complete.")
        logging.info("#############################################################################")


    def test3(self, analyse=[0.5], top_what_pred=1):
        """
        
        """
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)
        frame_lengths = [len(length) for length in self.image_reader()]
        self.dict_iou = {video_name: [] for video_name in video_names}
        
        preds_all = self.pred_reader()
        gt_all = self.gt_reader()

        self.analyse = analyse

        if self.dataset_name == "VOT2018_LT_Subset" or self.dataset_name == "VOT2018_LT":
            self.target_classes = {'bicycle': 1, 'car9': 3}

        elif self.dataset_name == "VOT2016_Subset" or self.dataset_name == "VOT2016_Subset_Subset":
            self.target_classes = {'ball1': 33, 'ball2': 33, 'basketball': 1, 'birds1': 15,
                    'birds2': 15, 'blanket': 1, 'bmx': 1, 'bolt1': 1, 'bolt2': 1, 'book': 74,
                    'car1': 3, 'car2': 3, 'fernando': 16, 'girl': 1, 'graduate': 1, 'gymnastics1': 1, 
                    'gymnastics2': 1, 'gymnastics3': 1, 'gymnastics4': 1, 'handball1': 1, 'handball2': 1,
                    'iceskater1': 1, 'iceskater2': 1, 'motocross1': 4, 'motocross2': 4, 'nature': 15,
                    'pedestrian1': 1, 'pedestrian2': 1, 'racing': 3, 'road': 4, 'sheep': 19, 'singer1': 1,
                    'singer2': 1, 'soccer2': 1, 'traffic': 1, 'tunnel': 3, 'wiper': 3, 'matrix': 81,
                    'shaking': 81, 'singer3': 81, 'soccer1': 81, 'soldier': 81}

        threads = []

        for video_id, video_name in enumerate(video_names):
            dict_ref = self.dict_iou[video_name]
            t = threading.Thread(target=self.retreiver, args=(frame_lengths[video_id],
                            preds_all, gt_all, video_id, dict_ref))
            threads.append(t)
            logging.debug("IoU calculation process for '{}' has begun.".format(video_name))

            # Alternative code is stated below but it will not create multiple processes.

            # for frame_id in range(frame_lengths[video_id]):
            #     framewise_preds = self.ret_preds(preds_all, video_id, frame_id+1)
            #     gt = gt_all[video_id][frame_id]
            #     iou_frame_all_pred = []
            #     for pred in framewise_preds:
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
        logging.info("TEST 3: Assigning bboxes (+) or (-) based on if the prediction is "+\
                     "True or False depending on the IoU being higher than a specified threshold.")

        self.different_iou_video_based_conf = {video_name: np.zeros((len(analyse), 12)) for video_name in video_names}
        self.all_videos_temporal_stats = {video_name: np.zeros((frame_lengths[video_id], 12)) for video_id, video_name in enumerate(video_names)}

        for video_id, video_name in enumerate(video_names):
            for thr_id, iou_thr in enumerate(analyse):
                counter_single_pred = 0
                counter_multi_pred = 0

                miss_detection = 0
                fp_single_type_none = 0
                tp_single = 0
                fn_single = 0
                fp_single = 0
                tn_single = 0
                fp_multi_type_none = 0
                tp_multi = 0
                fn_multi = 0
                tn_multi = 0
                fp_multi_1 = 0
                fp_multi_2 = 0

                for d, framewise_preds in enumerate(self.dict_iou[video_name]):
                    miss_detection_temp = 0
                    tp_single_temp = 0
                    fn_single_temp = 0
                    fp_single_temp = 0
                    tn_single_temp = 0
                    tp_multi_temp = 0
                    fn_multi_temp = 0
                    tn_multi_temp = 0
                    fp_multi_1_temp = 0
                    fp_multi_2_temp = 0

                    # When there is no prediction
                    if len(framewise_preds) is 0:
                        miss_detection_temp += 1
                        miss_detection += 1

                    # When there is single prediction
                    elif len(framewise_preds)==1 and self.heed_singles:
                        counter_single_pred += 1
                        if (framewise_preds[0][0] == None and self.record_nones):
                            fp_single_type_none += 1

                        elif framewise_preds[0][0]>=iou_thr:
                            if framewise_preds[0][2*top_what_pred-1]==self.target_classes[video_name]:
                                if self.counter_or_objectness is 'counter':
                                    tp_single += 1
                                    tp_single_temp += 1
                                elif self.counter_or_objectness is 'objectness':
                                    tp_single += framewise_preds[0][2*top_what_pred]
                                    tp_single_temp += framewise_preds[0][2*top_what_pred]

                            else:
                                fn_single += 1
                                fn_single_temp += 1

                        elif framewise_preds[0][0]<iou_thr:
                            if framewise_preds[0][2*top_what_pred-1]==self.target_classes[video_name]:
                                fp_single += 1
                                fp_single_temp += 1

                            else:
                                tn_single += 1
                                tn_single_temp += 1

                    # When there are multiple predictions
                    elif len(framewise_preds)>1 and self.heed_multiples:
                        idx = []
                        for n, pred in enumerate(framewise_preds):
                            counter_multi_pred+=1

                            if (pred[0] == None and self.record_nones):
                                fp_multi_type_none += 1

                            elif pred[0]<iou_thr:
                                if pred[2*top_what_pred-1]==self.target_classes[video_name]:
                                    fp_multi_1 += 1
                                    fp_multi_1_temp += 1
                                
                                else:
                                    tn_multi += 1
                                    tn_multi_temp += 1

                            # Obtain the indices which might be true positive bboxes
                            elif pred[0]>=iou_thr:
                                if pred[2*top_what_pred-1]==self.target_classes[video_name]:
                                    idx.append((n, pred[0]))

                                else:
                                    fn_multi += 1
                                    fn_multi_temp += 1
                                    
                        idx = sorted(idx, key=lambda x: x[1], reverse=True)
                        
                        counter = 0
                        for n, pred in enumerate(framewise_preds):
                            if any(k==n for k, iou in idx):
                                if counter==0:
                                    if self.counter_or_objectness is 'counter':
                                        tp_multi += 1
                                        tp_multi_temp += 1
                                    elif self.counter_or_objectness is 'objectness':
                                        tp_multi += framewise_preds[0][2*top_what_pred]
                                        tp_multi_temp += framewise_preds[0][2*top_what_pred]
                                    counter+=1
                                else:
                                    fp_multi_2 += 1
                                    fp_multi_2_temp += 1

                    self.all_videos_temporal_stats[video_name][d] = d, tp_multi_temp, fp_multi_1_temp,\
                            fp_multi_2_temp, fn_multi_temp, tn_multi_temp, miss_detection_temp,\
                            tp_single_temp, fn_single_temp, fp_single_temp, tn_single_temp,\
                            tp_multi_temp+tp_single_temp

                if self.record_nones:
                    logging.info("VName: {}\tIOU: {}\tSingle Predictions FP from none GT: {}".format(video_name, iou_thr, fp_single_type_none))
                    logging.info("VName: {}\tIOU: {}\tMulti Predictions FP from none GT: {}".format(video_name, iou_thr, fp_multi_type_none))

                if self.heed_singles:
                    logging.info("VName: {}\tIOU: {}\tSingle Predictions TP: {}".format(video_name, iou_thr, tp_single))
                    logging.info("VName: {}\tIOU: {}\tSingle Predictions FP: {}".format(video_name, iou_thr, fp_single))
                    logging.info("VName: {}\tIOU: {}\tSingle Predictions FN: {}".format(video_name, iou_thr, fn_single))
                    logging.info("VName: {}\tIOU: {}\tSingle Predictions TN: {}".format(video_name, iou_thr, tn_single))

                logging.info("VName: {}\tIOU: {}\tMulti Predictions TP: {}".format(video_name, iou_thr, tp_multi))
                logging.info("VName: {}\tIOU: {}\tMulti Predictions FP Type I: {}".format(video_name, iou_thr, fp_multi_1))
                logging.info("VName: {}\tIOU: {}\tMulti Predictions FP Type II: {}".format(video_name, iou_thr, fp_multi_2))
                logging.info("VName: {}\tIOU: {}\tMulti Predictions FN: {}".format(video_name, iou_thr, fn_multi))
                logging.info("VName: {}\tIOU: {}\tMulti Predictions TN: {}".format(video_name, iou_thr, tn_multi))
                self.different_iou_video_based_conf[video_name][thr_id] = iou_thr, fp_single_type_none, fp_multi_type_none,\
                                 tp_single, fp_single, fn_single, tn_single, tp_multi, fp_multi_1, fp_multi_2, fn_multi, tn_multi
            logging.info("\t\t\t\tNumber of multiple predictions for {}: {}".format(video_name, counter_multi_pred))
            logging.info("\t\t\t\tNumber of single predictions for {}: {}".format(video_name, counter_single_pred))
            logging.info("\t\t\t\tTotal number of frames for {} is: {}".format(video_name, frame_lengths[video_id]))

        logging.info("TEST 3 complete.")
        logging.info("#############################################################################")
    

    def iouth_count_graph(self):

        for video_name in self.different_iou_video_based_conf.keys():
            iou_thr = self.different_iou_video_based_conf[video_name][:, 0]
            features = self.different_iou_video_based_conf[video_name][:, 1:]
            fig = plt.figure()
            if self.record_nones:
                plt.plot(iou_thr, features[:, 0], label="FP Single w/ GT None")
                plt.plot(iou_thr, features[:, 1], label="FP Multiple w/ GT None")

            if self.heed_singles:
                plt.plot(iou_thr, features[:, 2], label="TP Single")
                plt.plot(iou_thr, features[:, 3], label="FP Single")
                plt.plot(iou_thr, features[:, 4], label="FN Single")
                plt.plot(iou_thr, features[:, 5], label="TN Single")

            if self.heed_multiples:
                plt.plot(iou_thr, features[:, 6], label="TP Multiple")
                plt.plot(iou_thr, features[:, 7], label="FP Multiple Type I")
                plt.plot(iou_thr, features[:, 8], label="FP Multiple Type II")
                plt.plot(iou_thr, features[:, 9], label="FN Multiple")
                plt.plot(iou_thr, features[:, 10], label="TN Multiple")

            if self.all_tp:
                plt.plot(iou_thr, features[:, 2]+features[:, 6], label="TP Single+Multi")

            if self.record_nones or self.heed_singles or self.heed_multiples or self.all_tp:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
                plt.suptitle("Test Statistics for {}".format(video_name))
                plt.ylabel('Number of predictions satisfying the condition')
                plt.xlabel('IoU Threshold')
                plt.savefig(self.log_analysis+video_name+".jpg")


    def frame_count_stats_pdf_graph(self):

        assert len(self.analyse)==1

        for video_name in self.all_videos_temporal_stats.keys():
            frame_ids = self.all_videos_temporal_stats[video_name][:, 0]
            tp_multi = self.all_videos_temporal_stats[video_name][:, 1]
            fp_multi_1 = self.all_videos_temporal_stats[video_name][:, 2]
            fp_multi_2 = self.all_videos_temporal_stats[video_name][:, 3]
            fn_multi = self.all_videos_temporal_stats[video_name][:, 4]
            tn_multi = self.all_videos_temporal_stats[video_name][:, 5]
            miss_det = self.all_videos_temporal_stats[video_name][:, 6]
            tp_single = self.all_videos_temporal_stats[video_name][:, 7]
            fn_single = self.all_videos_temporal_stats[video_name][:, 8]
            fp_single = self.all_videos_temporal_stats[video_name][:, 9]
            tn_single = self.all_videos_temporal_stats[video_name][:, 10]
            tp_all = self.all_videos_temporal_stats[video_name][:, 11]
            
            if self.heed_singles:
                label = "TP Single"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_tpsingle_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, tp_single, label, title, save_dir)

                label = "FN Single"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_fnsingle_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, fn_single, label, title, save_dir)

                label = "FP Single"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_fpsingle_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, fp_single, label, title, save_dir)

                label = "TN Single"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_tnsingle_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, tn_single, label, title, save_dir)

            if self.heed_multiples:            
                label = "TP Multiple"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_tpmulti_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, tp_multi, label, title, save_dir)

                label = "FP Type I Multiple"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_fpt1multi_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, fp_multi_1, label, title, save_dir)

                label = "FP Type II Multiple"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_fpt2multi_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, fp_multi_2, label, title, save_dir)

                label = "FN Multiple"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_fnmulti_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, fn_multi, label, title, save_dir)

                label = "TN Multiple"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_tnmulti_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, tn_multi, label, title, save_dir)

            label = "Miss Detection"
            title = label + " for {}".format(video_name)
            save_dir = self.log_analysis+video_name+\
                "_temporal_pdf_md_iou{}".format(self.analyse[0])+".jpg"
            self.figure_function(video_name, frame_ids, miss_det, label, title, save_dir)
            
            if self.all_tp:
                label = "TP All"
                title = label + " for {}".format(video_name)
                save_dir = self.log_analysis+video_name+\
                    "_temporal_pdf_tpall_iou{}".format(self.analyse[0])+".jpg"
                self.figure_function(video_name, frame_ids, tp_all, label, title, save_dir)

    
    def figure_function(self, video_name, frame_ids, data_points, label, title, save_dir):
        plt.figure()
        plt.plot(frame_ids, data_points, label=label)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.suptitle(title)
        if self.counter_or_objectness is 'counter':
            plt.ylabel('Number of predictions satisfying the condition (counter)')
        elif self.counter_or_objectness is 'objectness':
            plt.ylabel('Number of predictions satisfying the condition (obj. score)')
        plt.xlabel('Frame #')
        plt.savefig(save_dir)
        plt.close()

    def conf_matrix(self, top_what_pred=1, one_gt_plot=True):
        ## TO DO ##
        ## 1) Write the images onto a file.
        ## 2) Save the numbers into a CSV file.
        """
        Constructs the confusion matrix given the 
        """
        video_names = os.listdir(self.datasets_dir)
        video_names = sorted(video_names)

        for video_id, video_name in enumerate(video_names):
            cm = np.zeros((81, 81), dtype=np.int32)
            gt_index = self.target_classes[video_name]
            for preds in self.dict_iou[video_name]:
                for pred in preds:
                    pred_index = pred[2*top_what_pred-1]
                    cm[gt_index][pred_index] += 1
            if one_gt_plot:
                cm = cm[gt_index, :][np.newaxis]
            # self.plot_confusion_matrix(cm, one_gt_plot, true_lbl=gt_index)
            self.plot_confusion_matrix(cm, one_gt_plot)

    def plot_confusion_matrix(self, cm, one_gt_plot,
                            true_lbl=None, 
                            classes=None,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues,
                            numbers=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            # print('Confusion matrix, without normalization')
            pass

        plt.figure()
        plt.yticks([])
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if classes is not None:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

        if numbers:
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        if true_lbl is not None:
            plt.ylabel('True label {}'.format(true_lbl))
        else:
            plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

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

            elif self.dataset_name == "VOT2016_Subset" or self.dataset_name == "VOT2016_face_Subset"\
                or self.dataset_name == "VOT2016":

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
            
            # Handling the first line seen in result txt files with top-5 probs. 
            if self.number_top == 5:
                temp_input_lines = temp_input_lines[1:]

            for id, line in enumerate(temp_input_lines):
                if self.number_top == 1:
                    fr_id, x, y, w, h, _, label = map(float, line.split("\t"))
                    video_preds.append(np.array([fr_id, x, y, x+w, y+h, label], dtype=np.int16))
                elif self.number_top == 5:
                    try:
                        fr_id, x, y, w, h, _, _, label1, prob1, label2, prob2, label3, prob3, label4,\
                                            prob4, label5, prob5 = map(float, line.split("\t"))
                    except:
                        raise(AssertionError("video name: {} line number: {}".format(video, id+2)))
                        
                    video_preds.append(np.array([(fr_id, x, y, x+w, y+h, label1, prob1, label2, 
                                        prob2, label3, prob3, label4, prob4, label5, prob5)], 
                                        dtype=[('', 'i4'),('', 'i4'),('', 'i4'),('', 'i4'),('', 'i4'),
                                        ('', 'i4'),('', 'f4'),('', 'i4'),('', 'f4'),('', 'i4'),
                                        ('', 'f4'),('', 'i4'),('', 'f4'),('', 'i4'),('', 'f4')]))
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
    # test-case -1
    images_dir = "VOT2016/"
    dataset_name = "VOT2016_face_Subset"
    datasets_dir = "Datasets/"+dataset_name+"/"+images_dir+"/"
    ground_truth_file_name = "groundtruth.txt"
    preds_dir = "logs/Evaluations/MASK_VOT2018_Subsets/stage1-10of30/"
    # preds_dir = "logs/Evaluations/MASK_VOT2016FACE_184imgtrain/"
    number_top = 5
    log_analysis = "analysistop2/"
    debug = True

    # test-case 0
    # dataset_name = "VOT2016_Subset_Subset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2016_Subset_final_th0.01_top5prob/"
    # number_top = 5
    # log_analysis = "analysis_temp/"
    # debug = True

    # test-case 1
    # dataset_name = "VOT2016_Subset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2016_final_th0/"
    # number_top = 1

    # test-case 2
    # dataset_name = "VOT2018_LT_Subset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2018_Subsets/"+\
    #         "MASK_VOT2018_final_th0.01_detnms_th0.4/"
    # number_top = 1

    # test-case 3
    # dataset_name = "VOT2018_LT"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # preds_dir = "logs/Evaluations/MASK_VOT2018_final_th0.001/"
    # number_top = 1

    # ozan case
    # dataset_name = "MSPR_Dataset"
    # datasets_dir = "Datasets/"+dataset_name+"/"
    # ground_truth_file_name = "groundtruth.txt"
    # number_top = 1

    analyse = Analysis(datasets_dir, ground_truth_file_name, number_top, log_analysis,
                         debug=debug, preds_dir=preds_dir, dataset_name=dataset_name,
                         heed_singles=True, heed_multiples=True, all_tp=True,
                         counter_or_objectness='counter')
    analyse.video_player()

    # analyse.test1()
    # analyse.test2()

    # thr = list(np.arange(0.001, 1.001, 0.001))
    # analyse.test3(analyse=thr)
    # analyse.iouth_count_graph()
    
    # for top_x in range(1, 6):
    #     log_analysis = "logs/analysis_face/analysistop{}/".format(top_x)
    #     analyse = Analysis(datasets_dir, ground_truth_file_name, number_top, log_analysis,
    #                     debug=debug, preds_dir=preds_dir, dataset_name=dataset_name,
    #                     heed_singles=True, heed_multiples=True, all_tp=True,
    #                     counter_or_objectness='objectness')
    #                     
    #     analyse.test3(analyse=[0.5], top_what_pred=top_x)
    #     analyse.frame_count_stats_pdf_graph()

    # analyse.test3(analyse=[0.9])
    # analyse.conf_matrix()
