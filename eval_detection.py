import json
# import urllib2

import numpy as np
import pandas as pd
# from joblib import Parallel, delayed

# from utils import get_blocked_videos
from utils import interpolated_prec_rec
from utils import segment_iou_time

class ANETdetection(object):

    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    GROUND_TRUTH_FIELDS = ['database']
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']
    PREDICTION_FIELDS = ['results']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        # if self.check_status:
        #     self.blocked_videos = get_blocked_videos()
        # else:
        #     self.blocked_videos = list()
        self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        activity_index_sum={}
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid in data['database'].keys():
            v= data['database'][videoid]
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    activity_index_sum[ann['label']] = 1
                    cidx += 1
                else:                
                    activity_index_sum[ann['label']] = activity_index_sum[ann['label']]+1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['moment']))
                # t_end_lst.append(float(ann['moment']))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                    #  't-end': t_end_lst,
                                     'label': label_lst})
        # print(activity_index)
        # print(activity_index_sum)
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid  in data['results'].keys():
            v=data['results'][videoid]
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['moment']))
                # t_end_lst.append(float(result['moment']))
                label_lst.append(label)
                score_lst.append(result['score'])
                # score_lst.append(1)
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                #    't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        # results = Parallel(n_jobs=len(self.activity_index))(
        #             delayed(compute_average_precision_detection)(
        #                 ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
        #                 prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
        #                 tiou_thresholds=self.tiou_thresholds,
        #             ) for label_name, cidx in self.activity_index.items())


        results=[]
        for label_name, cidx in self.activity_index.items():
            results.append(compute_average_precision_detection(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        tiou_thresholds=self.tiou_thresholds,
                    ))

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        # self.mAP = np.array(weight_ap)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on CCTV-Pipe detection task.')
            print ('\tAverage-mAP: {}'.format(self.average_mAP))
            # print([round(x,5) for x in self.mAP])

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou_time(this_pred[['t-start' ]].values,
                               this_gt[['t-start' ]].values)

        # print(this_gt)
        # print(this_pred)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] > tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap



import sys
import os
import os.path

# input_dir = sys.argv[1]
# output_dir = sys.argv[2]

# submit_dir = os.path.join(input_dir, 'res')
# truth_dir = os.path.join(input_dir, 'ref')

# if not os.path.isdir(submit_dir):
#     print ("%s doesn't exist" % submit_dir)

# if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)


submit_dir='./'
truth_dir='./'
output_dir='./'

if __name__ == "__main__":

    from eval_detection import ANETdetection

    anet_detection = ANETdetection(
        ground_truth_filename=os.path.join(truth_dir,"cctv_pipe.json"),
        prediction_filename=os.path.join(submit_dir,'test_result.json'),
        subset='testing', tiou_thresholds=np.linspace(5, 15, 3), verbose=True, check_status=False)
    anet_detection.evaluate()

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')


    output_file.write("Avg.mAP"+": {:.3f}".format(np.mean(anet_detection.mAP)*100)+"\n")
    output_file.write("mAP@5"+": {:.3f}".format(anet_detection.mAP[0]*100)+"\n")
    output_file.write("mAP@10"+": {:.3f}".format(anet_detection.mAP[1]*100)+"\n")
    output_file.write("mAP@15"+": {:.3f}".format(anet_detection.mAP[2]*100))


    output_file.close()