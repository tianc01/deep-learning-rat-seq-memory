'''
This file populates data from a .mat file to a SQLlite database

Table Column information:
-------
Timestamps: the upper limit of their associated timestamp (bin), i.e. each row contains 
    values associated measurements/events occuring between the previous and current row's timestamp value.
T?_LFP_Raw: Raw LFP trace (in voltage)

T?_LFP_Raw_HilbVals: Hilbert transformed LFP (phase measurements (range +/- pi))

T?_LFP_Theta: The Theta filtered LFP (in voltage)

T?_LFP_Theta_HilbVals: Hilbert transformed Theta LFP (phase measurements (range +/- pi))

T?_LFP_Beta: Beta filtered LFP (in voltage)

T?_LFP_Beta_HilbVals: Hilbert transformed Beta LFP (phase measurements (range +/- pi))

T?-U?: binary (0 or 1) indicators of whether there was a spike in that time bin or not for unit? of tetrode?

Odor?: (1 or 0) trial odor was presented or not

Position?: (1 or 0) position indicator

InSeqLog: 1 for InSeq trial, 0 for the rest (nontrials and OutSeq)

PerformanceLog: 1(correct), -1(incorrect), 0(null event)

PokeEvents: 1(rat initialy enters the port), -1(rat withdraws from the port), 0(null event)

XvalRatMazePosition/YvalRatMazePosition: a position value (probably in pixel) or a NaN (position was 
    recorded every 15ms on average, many unrecorded rows)
'''

from pdb import set_trace as st
import scipy.io
import pandas as pd
import numpy as np
import pprint
import pickle
from trial import Trial
import h5py
import sqlite3

class PopulateData():
    def __init__(self, matfile):
        print('Loading .mat file...')
        file = h5py.File(matfile,'r') 

        print('Loading column names...')
        self.obtain_column_names(file)

        print('Loading data and create pandas dataframe...')
        data = np.rollaxis(np.array(file['statMatrix']),axis = 1)
        self.df = pd.DataFrame(data, index = range(data.shape[0]), columns = self.col_names)

        # connect to sql database ratwork.db
        conn = sqlite3.connect("ratwork.db")
        # write a pandas dataframe to TABLE "superchris_session1" in ratwork.db
        self.df.to_sql("superchris_session1", conn, index=True, index_label = 'df_idx')
        # print table information
        query_one = "pragma table_info(superchris_session1);"
        print(conn.execute(query_one).fetchall())
        # print first 5 row of the table
        query_two = "select * from superchris_session1 limit 5;"
        print(conn.execute(query_two).fetchall())
        # close the connection
        conn.close()

    def obtain_column_names(self, file):
        self.col_names = []
        for col in file['statMatrixColIDs']:
            col_name_as_list = ['_' if chr(i) == '-' else chr(i) for i in file[col[0]][:]]
            col_name = ''.join(col_name_as_list)
            self.col_names.append(col_name)
        readme_file = open('mat_col_desc.txt','r')
        for line in readme_file.readlines():
            print(line.rstrip('\n'))
        print('Display columns')
        for i in self.col_names:
            print i

pd = PopulateData('data/SuperChris_WellTrainedSession.mat')