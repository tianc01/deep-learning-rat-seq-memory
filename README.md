# README #

This repository implements a CNN model in decoding neural activities for a sequence memory task.

### Instructions ###

* Download the .mat file which contains a table with each row representing every 0.001 second recording time (50 minutes in total).

* Run `to_sql_all_tetrodes_superchris_session1.py`, which writes all the data in `SuperChris_WellTrainedSession.mat` to a SQlite database table.

* Run `rat_data.py`, which extracts data related to trials only from the SQL database. A trial starts with rat's nose-poke, however, we would like to extract data a few seconds before and after a trial, and decode the neural activities during that period. Currently we are extracting -2s to 2s relative to the trial start time (poke time). The relative start and end time can be changed in `data/superchris_session1_cnn.json`: `"test"-->"start_sec"` and `"test"-->"end_sec"`.

* Run `decode_data.py`. It has the following functionalities:
    * Load/extract trials data
    * Train, predict and save the prediction
    * Plot prediction results
    
	You can modify `dd = DecodeData('data/*.json')` at the end of the file in `if __name__ == "__main__":`. If implementing CNN model, use 'data/superchris_session1_cnn.json'. If implementing logistic regression, use 'data/superchris_session1_lr.json'.
	
### Examples ###
An example for predicting memory replay
![An example for predicting memory replay](https://github.com/tianc01/deep-learning-rat-seq-memory/blob/master/fig/prediction.png)

### Files ###

* `data/SuperChris_WellTrainedSession.mat`: Raw data. About 3GB. You have to download it from the shared Google Drive. 

* `data/superchris_session1_cnn.json`: This JSON file will be imported to `rat_data.py` and `decode_data.py`. All the parameters in `rat_data.py` and `decode_data.py`should be changed and stored in this file. 

* `to_sql_all_tetrodes_superchris_session1.py`: This file populates raw data from the .mat file to a SQLlite database

* `rat_data.py`: This file extracts data associated with trials only. The data is saved as a .pkl file.

* `decode_data.py`: This file implements supervised classification for sequence memory replay neural decoding.

* `trial.py`: Define class Trial

* `config.py`: Configure all the parameters in a JSON file to a python dictionary

* `lr_trainer.py`: Logistic regression trainer

* `cnn_trainer.py`: CNN (LeNet) trainer

* `mat_col_desc.txt`: Table (.mat file) column description
