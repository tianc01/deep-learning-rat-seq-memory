"""
This file implements supervised classification for sequence memory replay neural decoding

Content
-------
The package mainly contains:

  get_trials_data           get all trials data
  get_train_test_params     set up all the universal/global parameters
  get_raw_train_data        get training data
  train                     train the model 
  predict                   predict odors across the trial time and save the results in a pickle file

and visualization functions:
  plot_mean_and_individual_prediction_by_odor     plot mean and individual predictions in one plot (grouped by odor)
  plot_mean_confband_prediction_by_odor           plot mean predictions with confidence band (grouped by odor)
  plot_prediction_by_odor_per_trial               plot individual predictions (grouped by odor)
"""

import pickle, json
import numpy as np
from pdb import set_trace as st
import matplotlib
from matplotlib import colors
# matplotlib.use("TkAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import time
from itertools import cycle
from scipy import interp

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold

from lenet_trainer import LeNetTrainer
from lr_trainer import LRTrainer
from lr_lasso_trainer import LRLassoTrainer
from rat_data import RatData
from config import Config
from trial import Trial

class DecodeData():
    def __init__(self, json_file_name):
        self.config = Config(json.load(open(json_file_name, 'r')))

        # self.prediction_name: universal name prefix for saving results
        self.prediction_name = 'results/prediction_{}_session{}_{}_train_from{}to{}_test_from{}to{}_decode_stride_{}_correct_{}_inseq_{}'.format(self.config.rat_name,
            self.config.session, self.config.train.trainer, self.config.train.start_sec, self.config.train.end_sec, 
            self.config.test.start_sec, self.config.test.end_sec, self.config.test.stride_size, self.config.correct_trial, self.config.in_seq_trial)

        # set up
        self.get_trials_data()
        self.get_train_test_params()

        # train and predict
        if self.config.steps.predict:
            self.get_raw_train_data()
            self.train()
            self.predict()

        if self.config.steps.cv_evaluate:
            self.get_raw_train_data()
            self.cv_evaluate_roc()

        # plot predictions for individual trials and their average in one plot
        if self.config.steps.plot_mean_and_individual:
            self.plot_mean_and_individual_prediction_by_odor()

        # plot average predictions with confidence band
        if self.config.steps.plot_mean_and_confband:
            self.plot_mean_confband_prediction_by_odor()

        # plot prediction for each trial and save them separately
        if self.config.steps.plot_per_trial:
            self.plot_prediction_by_odor_per_trial()

    def get_trials_data(self):
        '''Get all trials data by either loading saved data (.pkl) or extracting 
        trials data by creating RatData Instance
        '''

        # load trials data from pickle file directly
        if self.config.steps.load_trials:
            start_time = time.time()
            print('Loading trials...')
            trials = pickle.load(open(self.config.test.trials_file, 'rb'))
            self.trials = self.get_selected_idx(trials, correct = self.config.correct_trial, in_sequence = self.config.in_seq_trial)
            print("Loading trials --- %s seconds ---" % (time.time() - start_time))
        
        # extract trials data by creating instance of class RatData
        if self.config.steps.extract_rd_data:
            rd = RatData(json_file_name)
            print('Extracting trials...')
            trials = rd.extract_trials()
            rd.close_conn()
            self.trials = self.get_selected_idx(trials, correct = self.config.correct_trial, in_sequence = self.config.in_seq_trial)


    def get_selected_idx(self, trials, correct = True, in_sequence = True):
        ''' Select trials according to rat's performance (correct or not), and InSeq/OutSeq
        INPUTS
        ------
        trials: list
            list of Trial instances
        correct: bool
            rat is correct or not
        in_sequence: bool
            odor is in sequence or not

        OUTPUTS
        -------
        selected_trials: list
            list of Trial instances
        '''
        selected_trials = []
        correct_condition = np.ones(len(trials), dtype = np.uint8)
        in_seq_condition = np.ones(len(trials), dtype = np.uint8)
        for i, trial in enumerate(trials):
            if correct and not trial.performance:
                correct_condition[i] = 0
            if in_sequence and not trial.in_seq:
                in_seq_condition[i] = 0
            if correct_condition[i] == 1 and in_seq_condition[i] == 1:
                selected_trials.append(trial)
                print('Trial {} is selected'.format(trial.trial_idx))
        print('{} trials in total'.format(len(selected_trials)))
        return selected_trials

    def get_neuron_lfp_col_names(self):
        neuron_names = []
        tetrode_names = []
        for tetrode in self.config.tetrodes:
            neurons = eval('self.config.units.{}'.format(tetrode))
            if neurons:
                tetrode_names.append(tetrode+'_'+self.config.LFP_name)
            for neuron in neurons:
                neuron_names.append(neuron)
        return neuron_names, tetrode_names

    def get_train_test_params(self):
        self.train_start_idx = int(round((self.config.train.start_sec - self.config.test.start_sec)/self.config.bin_width))
        self.train_num_idx = int(round(self.config.train.window_size/self.config.bin_width))

        assert self.config.test.stride_size >= self.config.bin_width
        # assert self.config.test.stride_size % self.config.bin_width == 0
        self.idx_diff = int(round(self.config.test.stride_size/self.config.bin_width))
        self.num_test_times = int((self.config.test.end_sec - self.config.test.start_sec)/self.config.bin_width)
        self.test_data_start_idx = range(self.num_test_times)[0::self.idx_diff]
        self.test_times = [float("{0:.2f}".format(i*self.config.bin_width + self.config.test.start_sec)) for i in self.test_data_start_idx]

        self.neuron_names, self.tetrode_names = self.get_neuron_lfp_col_names()
       
    def one_hot_to_letter(self, one_hot):
        return self.config.odor_letters[np.argmax(one_hot)]
        
    def get_raw_train_data(self):
        '''
        self.X_train: nested list
            np.array(self.X_train): ndarray[number of trials, number of neurons, number of time bins]
            Example, if training window is from 0.15s to 0.4s, with 0.01s bin width, the number of time bins = 25
        self.y_train: nested list
            np.array(self.y_train): ndarray[number of trials, number of classes]
        '''
        print('Extracing raw training data...')
        self.X_train = []
        self.y_train = []
        for trial in self.trials:
            cur_X_train = []
            for neuron_spikes in trial.spikes['all']:
                cur_X_train.append(neuron_spikes[self.train_start_idx:self.train_start_idx+self.train_num_idx])
            self.X_train.append(cur_X_train)
            self.y_train.append(trial.odor)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)


    def train(self):
        self.trainer = eval('{}()'.format(self.config.train.trainer))
        if self.config.train.params:
            params = {}
            for param in dir(self.config.train.params): 
                if not param.startswith('__'):
                    exec('params["{}"] = self.config.train.params.{}'.format(param, param))
            fitted_coef = self.trainer.train(self.X_train, self.y_train, None, None, **params)
        else:
            fitted_coef = self.trainer.train(self.X_train, self.y_train)

    def cv_evaluate_roc(self):
        params = {}
        for param in dir(self.config.train.params): 
            if not param.startswith('__'):
                exec('params["{}"] = self.config.train.params.{}'.format(param, param))

        cv = StratifiedKFold(n_splits=self.config.cross_val.num_folds)
        roc_dict_keys = list(range(len(self.config.odor_letters)))+['micro','macro']
        tprs = dict()
        aucs = dict()
        for key in roc_dict_keys:
            tprs[key] = []
            aucs[key] = []
        mean_fpr = np.linspace(0, 1, 100)

        k = 0
        for train, test in cv.split(self.X_train[:,:,0], self.y_train[:,0]):
            cur_trainer = eval('{}()'.format(self.config.train.trainer))
            print('')
            print('Training Fold {}...'.format(k+1))
            fitted_coef = cur_trainer.train(self.X_train[train,:], self.y_train[train,:], self.X_train[test,:], self.y_train[test,:], **params)
            y_score = cur_trainer.predict(self.X_train[test,:])            
            y_test = self.y_train[test,:]

            # Compute ROC curve and area the curve
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(len(self.config.odor_letters)):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(self.config.odor_letters))]))
            # Then interpolate all ROC curves at this points
            macro_mean_tpr = np.zeros_like(all_fpr)
            for i in range(len(self.config.odor_letters)):
                macro_mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            macro_mean_tpr /= len(self.config.odor_letters)

            fpr["macro"] = all_fpr
            tpr["macro"] = macro_mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            for key in roc_dict_keys:
                tprs[key].append(interp(mean_fpr, fpr[key], tpr[key]))
                tprs[key][-1][0] = 0.0
                aucs[key].append(roc_auc[key])

            k += 1
        
        mean_tpr = dict()
        mean_auc = dict()
        std_auc = dict()
        for key in roc_dict_keys:
            mean_tpr[key] = np.mean(tprs[key], axis=0)
            mean_tpr[key][-1] = 1.0
            mean_auc[key] = auc(mean_fpr, mean_tpr[key])
            std_auc[key] = np.std(aucs[key])

        print(params)
        print('CV mean ROC micro-average (AUC = {:0.2f} $\pm$ {:0.2f})'.format(mean_auc["micro"], std_auc["micro"]))
        print('CV mean ROC macro-average (AUC = {:0.2f} $\pm$ {:0.2f})'.format(mean_auc["macro"], std_auc["macro"]))

        # # Plot all ROC curves
        # fig = plt.figure()
        # plt.plot(mean_fpr, mean_tpr["micro"],
        #          label='CV mean ROC micro-average (AUC = {:0.2f} $\pm$ {:0.2f})'
        #                ''.format(mean_auc["micro"], std_auc["micro"]),
        #          color='deeppink', linestyle=':', linewidth=4)

        # plt.plot(mean_fpr, mean_tpr["macro"],
        #          label='CV mean ROC macro-average (AUC = {:0.2f} $\pm$ {:0.2f})'
        #                ''.format(mean_auc["macro"], std_auc["macro"]),
        #          color='navy', linestyle=':', linewidth=4)

        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkolivegreen', 'darkorchid'])
        # for i, color in zip(range(len(self.config.odor_letters)), colors):
        #     plt.plot(mean_fpr, mean_tpr[i], color=color, lw=2, 
        #         label='CV mean ROC odor {} (AUC = {:0.2f} $\pm$ {:0.2f})'
        #                ''.format(self.config.odor_letters[i], mean_auc[i], std_auc[i]))

        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Cross validated average Receiver operating characteristic for multi-class')
        # plt.legend(loc="lower right")
        # fig.savefig('results/cv_mean_roc_{}.png'.format(self.config.train.trainer))
        # plt.close(fig)

    def predict(self):
        if self.config.train.load_trainer:
            self.trainer = eval('{}()'.format(self.config.train.trainer))
        test_results = {}
        for odor in self.config.odor_letters:
            test_results[odor] = []
        for trial in self.trials:
            test_data = []
            for start_idx in self.test_data_start_idx:
                cur_test_data = []
                for neuron_spikes in trial.spikes['all']:
                    cur_test_data.append(neuron_spikes[start_idx:start_idx+self.train_num_idx])
                test_data.append(cur_test_data)
            print('Predicting Trial {}...'.format(trial.trial_idx))
            prediction = self.trainer.predict(test_data)
            odor = self.one_hot_to_letter(trial.odor)
            test_results[odor].append(prediction)
        self.save_prediction(test_results)

    def save_prediction(self, test_results):
        save_path = '{}.pkl'.format(self.prediction_name)
        print('Saving prediction: {}'.format(save_path))
        with open(save_path,'wb') as f:
            pickle.dump(test_results,f)

    def load_prediction(self):
        load_path = '{}.pkl'.format(self.prediction_name)
        print('Loading prediction: {}'.format(load_path))
        test_results = pickle.load(open(load_path,'rb'))
        return test_results

    def plot_mean_and_individual_prediction_by_odor(self):
        ''' Plot mean and individual predictions in one plot (grouped by odor)
        '''
        test_results = self.load_prediction()
        fig, ax = plt.subplots(nrows = len(self.config.odor_letters), ncols = len(self.config.odor_letters),figsize = (20,20)) # 100: 10000 pixel
        ax_idx = 0
        for pred_odor_idx in range(len(self.config.odor_letters)):
            for true_odor in self.config.odor_letters:
                probs_over_time = []
                cur_ax = fig.axes[ax_idx]
                for trial in test_results[true_odor]:
                    probs_over_time.append(trial[:,pred_odor_idx])
                    cur_ax.plot(self.test_times, trial[:,pred_odor_idx], color = 'gray', linewidth = 0.2)

                ave_probs_over_time = np.mean(np.array(probs_over_time), axis = 0)
                cur_ax.plot(self.test_times, ave_probs_over_time, color = 'black', linewidth=3)
                cur_ax.set_ylim(0,1)
                cur_ax.set_title('True {} Predict {}'.format(true_odor, 
                    self.config.odor_letters[pred_odor_idx]), fontsize=20)
                ax_idx += 1
        fig.suptitle('Model {} train from {} to {} decoding stride {} ({} session{})'.format(self.config.train.trainer, self.config.train.start_sec, 
            self.config.train.end_sec, self.config.test.stride_size,  self.config.rat_name, self.config.session), fontsize=30)
        im_name = '{}_mean_indv.png'.format(self.prediction_name)
        fig.savefig(im_name)
        plt.close(fig)

    def plot_mean_confband_prediction_by_odor(self):
        ''' Plot mean predictions with confidence band (grouped by odor)
        '''
        test_results = self.load_prediction()
        fig, ax = plt.subplots(nrows = len(self.config.odor_letters), ncols = len(self.config.odor_letters),figsize = (20,20)) # 100: 10000 pixel
        ax_idx = 0
        for pred_odor_idx in range(len(self.config.odor_letters)):
            for true_odor in self.config.odor_letters:
                probs_over_time = []
                for trial in test_results[true_odor]:
                    probs_over_time.append(trial[:,pred_odor_idx])
                ave_probs_over_time = np.mean(np.array(probs_over_time), axis = 0)
                error_probs_over_time = 1.96*np.std(np.array(probs_over_time), axis = 0)
                cur_ax = fig.axes[ax_idx]
                cur_ax.plot(self.test_times, ave_probs_over_time)
                cur_ax.fill_between(self.test_times, ave_probs_over_time-error_probs_over_time, 
                    ave_probs_over_time+error_probs_over_time,linewidth=0, facecolor='lightgray')
                cur_ax.set_ylim(0,1)
                cur_ax.set_title('True {} Predict {}'.format(true_odor, 
                    self.config.odor_letters[pred_odor_idx]), fontsize=20)
                ax_idx += 1
        fig.suptitle('Model {} train from {} to {} ({} session{})'.format(self.config.train.trainer, self.config.train.start_sec, 
            self.config.train.end_sec,  self.config.rat_name, self.config.session), fontsize=30)
        im_name = '{}_mean_conf.png'.format(self.prediction_name)
        fig.savefig(im_name)
        plt.close(fig) 

    def plot_prediction_by_odor_per_trial(self):
        ''' Plot predictions for each trial (grouped by odor)
        '''
        test_results = self.load_prediction()
        save_path = '{}'.format(self.prediction_name)
        if not os.path.exists(save_path): os.makedirs(save_path)
        for true_odor in self.config.odor_letters:
            for trial_idx, trial in enumerate(test_results[true_odor]):
                fig, ax = plt.subplots(nrows = len(self.config.odor_letters), ncols = 1,figsize = (10,20)) # 100: 10000 pixel
                ax_idx = 0
                for pred_odor_idx in range(len(self.config.odor_letters)):
                    probs_over_time = trial[:,pred_odor_idx]
                    cur_ax = fig.axes[ax_idx]
                    cur_ax.plot(self.test_times, probs_over_time)
                    cur_ax.set_ylim(0,1)
                    cur_ax.set_title('Odor {} Trial {} Predict {}'.format(true_odor, trial_idx,
                        self.config.odor_letters[pred_odor_idx]), fontsize=20)
                    ax_idx += 1
                fig.suptitle('Model {} train from {} to {} ({} session{})'.format(self.config.train.trainer, self.config.train.start_sec, 
                    self.config.train.end_sec,  self.config.rat_name, self.config.session), fontsize=20)
                im_name = '{}/odor_{}_trial_{}.png'.format(save_path, true_odor, trial_idx)
                fig.savefig(im_name)
                plt.close(fig)

# ---------------------------------------------
if __name__ == "__main__":
    # dd = DecodeData('data/superchris_session1_lr.json')
    dd = DecodeData('data/superchris_session1_lenet.json')