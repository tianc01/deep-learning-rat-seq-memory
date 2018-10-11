'''
This file extracts data associated with trials only. The data is saved as a .pkl file.
Note: The data in the SQL database table is for the entire session (about 50 minutes).

Content
-------
The package mainly contains:
  extract_trials          extract all trials data
'''

import json, pickle
import math
import numpy as np
import sqlite3
from pdb import set_trace as st

from config import Config
from trial import Trial
import peakutils
import os
import pprint
import time

class RatData:
    def __init__(self, json_file_name):
        self.config = Config(json.load(open(json_file_name, 'r')))

        # connect to SQL database
        self.conn = sqlite3.connect(self.config.db_name)

        # # fetch table information
        # query = "pragma table_info({});".format(self.config.tb_name)
        # print(self.conn.execute(query).fetchall())

        self.neuron_names, self.tetrode_names = self.get_neuron_lfp_col_names()

    def get_neuron_lfp_col_names(self):
        ''' Extract Neuron Names and Tetrode Names
        Note: Each neuron is linked to one of the tetrodes. 
        For example, neurons "T1_U1", "T1_U2", "T1_U3", "T1_U4" are linked to tetrode "T1".
        OUTPUTS
        -------
        neuron_names: list
            list of neuron names
        tetrode_names: list
            list of tetrode names. note: there are some tetrodes with no neuron linked. Only
            tetrode linked with neurons are selected.
        '''
        neuron_names = []
        tetrode_names = []
        for tetrode in self.config.tetrodes:
            neurons = eval('self.config.units.{}'.format(tetrode))
            if neurons:
                tetrode_names.append(tetrode+'_'+self.config.LFP_name)
            for neuron in neurons:
                neuron_names.append(neuron)
        pprint.pprint('Tetrodes: {}'.format(tetrode_names))
        pprint.pprint('Neurons: {}'.format(neuron_names))
        return neuron_names, tetrode_names

    def get_trial_times(self):
        """ Obtain index and key times for each trial from the SQL databse table
        OUTPUTS
        -------
        all_poke_idx: list
            SQL table row index of poke time for all trials
        poke_times: list
            poke time in second for all trials 
        all_withdraw_idx: list
            SQL table row index of withdraw time for all trials
        withdraw_times: list
            withdraw time in second for all trials 
        all_odor_idx: list
            SQL table row index of odor release time for all trials
        odor_times: list
            odor release time in second for all trials 
        """

        # fetch poke times for all trials from sql database table
        start_time = time.time()
        print('Fetching Poke Times...')
        query_poke = 'select df_idx, TimeBin from {} where PokeEvents = 1;'.format(self.config.tb_name)
        result_poke = self.conn.execute(query_poke).fetchall()
        all_poke_idx = [i[0] for i in result_poke]
        poke_times = [i[1] for i in result_poke]
        print("Total time --- %s seconds ---" % (time.time() - start_time))
        
        # fetch withdraw times for all trials from sql database table
        start_time = time.time()
        print('Fetching Withdraw Times...')
        query_withdraw = 'select df_idx, TimeBin from {} where PokeEvents = -1;'.format(self.config.tb_name)
        result_withdraw = self.conn.execute(query_withdraw).fetchall()
        all_withdraw_idx = [i[0] for i in result_withdraw]
        withdraw_times = [i[1] for i in result_withdraw]
        print("Total time --- %s seconds ---" % (time.time() - start_time))
        
        # fetch odor release times for all trials from sql database table
        start_time = time.time()
        print('Fetching Odor Times...')
        query_odor =  'select df_idx, TimeBin from {} where {}'.format(self.config.tb_name, self.config.odor_conditions)
        result_odor = self.conn.execute(query_odor).fetchall()
        all_odor_idx = [i[0] for i in result_odor]
        odor_times = [i[1] for i in result_odor]
        print("Total time --- %s seconds ---" % (time.time() - start_time))

        # assert Poke Time < Odor Release Time < Withdraw Time
        for i,j,k in zip(poke_times,odor_times,withdraw_times):
            assert i  < j and j < k

        return all_poke_idx, poke_times, all_withdraw_idx, withdraw_times, all_odor_idx, odor_times

    def get_behavior(self, idx):
        """ Obtain behaviorial data for one trial

        INPUTS
        ------
        idx: int
            SQL table row index of the odor time

        OUTPUTS
        -------
        odor: 'numpy.ndarray'
            example: array([1, 0, 0, 0, 0]) represents class 'A'
        position: 'numpy.ndarray'
            example: array([1, 0, 0, 0, 0]) represents the first position
        in_seq: int
            1 if InSeq. 0 if OutSeq.
        performance: int
            1 if rat is Correct. 0 if rat is Incorrect
        """
        query_behavior = 'select {} from {} where df_idx={}'.format(self.config.behavior_conditions, self.config.tb_name, idx)
        result_behavior = self.conn.execute(query_behavior).fetchall()
        odor = np.array([int(l) for l in result_behavior[0][self.config.behavior_idx.odor_start:self.config.behavior_idx.odor_end]])
        position = np.array([int(l) for l in result_behavior[0][self.config.behavior_idx.pos_start:self.config.behavior_idx.pos_end]])
        in_seq = int(result_behavior[0][self.config.behavior_idx.pos_end])
        performance = int(result_behavior[0][self.config.behavior_idx.pos_end+1])
        if performance != 1: performance = 0
        return odor, position, in_seq, performance

    def get_start_end_idx(self, poke_idx, withdraw_idx, bin_width, start_sec = 0, end_sec = None):
        """ Obtain the start and end row index for a trial.
        Note: a trial usually starts with rat's nose-poke. However, we would like to extract data
        a few seconds before and after a trial, and decode the neural activities.

        INPUTS
        ------
        poke_idx: int
            table row index for the start time (usually the poke time) of a trial
        withdraw_idx: int
            table row index for the withdraw time
        bin_width: float
            the width of time bin for each row (it's 0.001s in this table)
        start_sec: float
            the start time relative to nose-poke
        end_sec: float
            the end time relative to nose-poke

        OUTPUTS
        -------
        start_idx: int
            table row index for the relative start time
        end_idx: int
            table row index for the relative end time
        """
        start_idx = poke_idx + int(start_sec/bin_width)
        if end_sec is None:
            end_idx = withdraw_idx
        else:
            end_idx = start_idx + int((end_sec-start_sec)/bin_width)
        return start_idx, end_idx

    def get_spikes_lfp_data(self, start_idx, end_idx):
        """ Obtain the start and end row index for a trial.
        Note: a trial usually starts with rat's nose-poke. However, we would like to extract data
        a few seconds before and after a trial, and decode the neural activities.

        INPUTS
        ------
        start_idx: int
            table row index for the relative start time
        end_idx: int
            table row index for the relative end time

        OUTPUTS
        -------
        spikes_data: dict
            dict keys are neuron/unit names. 
        spikes_data['T23_U1']: list 
            list of number of spikes for every 0.001s from relative start time to relative end time
            len(spikes_data['T23_U1']) = number of time bins for one trial (bin width = 0.001s)
        lfp_data: dict
            dict keys are tetrode names.
        lfp_data['T23_LFP_Theta_HilbVals']: list
            list of local field potentials for every 0.001s from relative start time to relative end time
            len(lfp_data['T23_LFP_Theta_HilbVals']) = number of time bins for one trial (bin width = 0.001s)
        """
        query= 'select {}, {} from {} where df_idx >= {} and df_idx < {};'.format(', '.join(self.neuron_names), 
            ', '.join(self.tetrode_names), self.config.tb_name, start_idx, end_idx)
        result = self.conn.execute(query).fetchall()
        spikes_data = {}
        for idx, name in enumerate(self.neuron_names):
            spikes_data[name] = [l[idx] for l in result]
        lfp_data = {}
        for idx, name in enumerate(self.tetrode_names):
            lfp_data[name] = [l[idx+len(self.neuron_names)] for l in result]
        return spikes_data, lfp_data

    def get_test_data(self, test_data_start_idx, spikes_data, compress_data):
        """ Save neural spike train data from dictionary to a nested list.
        You have the option to compress the data (e.g., instead of saving number of spikes
        every 0.001s, saving spikes every 0.01s)

        INPUTS
        ------
        test_data_start_idx: list
            list of start index for each test data
        spikes_data: dict
            dict keys are neuron/unit names. 
        spikes_data['T23_U1']: list 
            list of number of spikes for every 0.001s from relative start time to relative end time
            len(spikes_data['T23_U1']) = number of time bins for one trial (bin width = 0.001s)
        compress_data: bool
            False if saving data every 0.001s
            True if saving data using wider time bin, e.g., every 0.01 s

        OUTPUTS
        -------
        test_spikes: nested list
            len(test_spikes) = number of neurons
        test_spikes[0]: list
            list of number of spikes from relative start time to end time for the first neuron
            len(test_spikes[0]) = number of time bins for one trial (bin width = self.config.bin_width)
        """
        test_spikes = []
        if compress_data:
            for neuron in self.neuron_names:
                test_spikes.append(self.get_compressed_data(test_data_start_idx, spikes_data[neuron]))
        else:
            for neuron in self.neuron_names:
                test_spikes.append(spikes_data[neuron])
        return test_spikes

    def get_compressed_data(self, test_data_start_idx, data):
        """ Save number of spikes using wider time bin

        INPUTS
        ------
        test_data_start_idx: list
            list of start index for each test data from relative start time to end time
        data: list
            list of number of spikes for one neuron 
            len(data) = number of time bins for one trial (bin width = self.config.original_bin_width = 0.001s)

        OUTPUTS
        -------
        new_data: list
            list of number of spikes for one neuron
            len(new_data) = number of time bins for one trial (bin width = self.config.bin_width)
        """
        new_data = [0.0] * int(len(data)/self.num_bins_per_spike_data)
        for new_data_idx, start_idx in enumerate(test_data_start_idx):
            if sum(data[start_idx:start_idx+self.num_bins_per_spike_data]) > 0:
                new_data[new_data_idx] = sum(data[start_idx:start_idx+self.num_bins_per_spike_data])
        return new_data

# ------------------------------------------------------------------------------------------------------------- #
#                             Extract all test spike sequences                                                  #
# ------------------------------------------------------------------------------------------------------------- #

    def extract_trials(self):
        trials = []
        all_poke_idx, poke_times, all_withdraw_idx, withdraw_times, all_odor_idx, odor_times = self.get_trial_times()
        
        # set up parameters
        self.total_bins_per_trial = int((self.config.test.end_sec + self.config.train.window_size- self.config.test.start_sec)/self.config.original_bin_width)
        self.num_bins_per_spike_data = int(self.config.bin_width/self.config.original_bin_width)
        test_data_start_idx = range(self.total_bins_per_trial)[0::self.num_bins_per_spike_data]

        # for each trial, create a Trial instance
        for i in range(len(odor_times)):
            # get behaviorial data for current trial
            odor, position, in_seq, performance = self.get_behavior(all_odor_idx[i])
            # create instance of class Trial
            cur_trial = Trial(i+1, poke_times[i], withdraw_times[i], odor_times[i], odor, position, in_seq, performance)

            # get start index and end index for a trial
            if self.config.onset_time:
                start_idx, end_idx = self.get_start_end_idx(all_odor_idx[i], all_withdraw_idx[i], self.config.original_bin_width,
                    self.config.test.start_sec, self.config.test.end_sec+self.config.train.window_size)
            else:
                start_idx, end_idx = self.get_start_end_idx(all_poke_idx[i], all_withdraw_idx[i], self.config.original_bin_width,
                    self.config.test.start_sec, self.config.test.end_sec+self.config.train.window_size)
            
            # get neural activity data (spike train and lfp) for current trial
            all_spikes_data, lfp_data = self.get_spikes_lfp_data(start_idx, end_idx)

            # save neural spike train data as a nested list
            cur_trial.spikes['all'] = self.get_test_data(test_data_start_idx, all_spikes_data, self.config.compress_data)

            trials.append(cur_trial)

        # save the trials data in a pickle file
        with open(self.config.test.trials_file,'wb') as f:
            pickle.dump(trials,f)

        return trials

    def close_conn(self):
        self.conn.close()

if __name__ == "__main__":
    rd = RatData('data/superchris_session1_lr.json')
    rd.extract_trials()
    rd.close_conn()