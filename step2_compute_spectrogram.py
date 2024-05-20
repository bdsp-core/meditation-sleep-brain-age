import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import spectrogram
from tqdm import tqdm
import sys
sys.path.insert(0, 'myfunctions')
from load_dataset import *
from segment_EEG import segment_EEG, myprint


if __name__ == '__main__':
    epoch_length = 30 # [s]
    amplitude_thres = 490 # [uV]
    line_freq = 60.  # [Hz]
    bandpass_freq = [0.5, 20.]  # [Hz]
    n_jobs = 8
    newFs = 200.
    # normal_only = True  # we are using channel wise epoch status

    # get list of files
    #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
    df_mastersheet = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')


    # for each recording
    bad_reasons = {}
    for si in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID.iloc[si]
        dataset = df_mastersheet.DataType.iloc[si]
        signal_path = df_mastersheet.signal_path.iloc[si]
        label_path = df_mastersheet.label_path.iloc[si]
        predicted_label_path = df_mastersheet.predicted_label_path.iloc[si]
        feature_path = df_mastersheet.feature_path.iloc[si]
            
        #feature_path = feature_path.replace('Sleep_data_mediation','/spectrograms/')
        #if os.path.exists(feature_path):
        #    continue
           
        #try:
        _load_dataset = eval(f'load_{dataset}_dataset')
        EEG, sleep_stages, EEG_channels, combined_EEG_channels, Fs, start_time = _load_dataset(signal_path, label_path)
        # check whether sleep_stage contains all 5 stages
        #unique_stages = set(sleep_stages[~np.isnan(sleep_stages)])
        #if len(unique_stages)<=2:
            #raise ValueError(f'{sid}: #sleep stage <= 2 ({str(unique_stages)})')
            
        # segment EEG
        epochs, sleep_stages, epoch_start_idx, epoch_status, specs, freq = segment_EEG(EEG, sleep_stages, epoch_length, epoch_length, Fs, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, amplitude_thres=amplitude_thres, n_jobs=n_jobs, make_artifact_eeg_nan=False)
         #if len(specs) <= 4 * 3600/30:
             #raise ValueError(f'{sid}: too short')
                
        myprint(epoch_status, EEG_channels)
        mat = {
            'start_time':start_time.strftime('%Y-%m-%d %H:%M:%S'),
            # 'EEG_feature_names':feature_names,
            # 'EEG_features':features,
            'EEG_specs': specs,
            'EEG_frequency': freq,
            'channel_names': EEG_channels,
            'sleep_stages': sleep_stages,
            'epoch_start_idx': epoch_start_idx,
            'age': df_mastersheet.Age.iloc[si],
            'sex': df_mastersheet.Sex.iloc[si],
            'epoch_status': epoch_status,
            'Fs': Fs,
            'newFs': newFs,
             }
                
        if not pd.isna(predicted_label_path):
            df_predicted_label = pd.read_csv(predicted_label_path)
            predicted_sleep_stages = df_predicted_label['sleep stage corrected'].values
            T = len(sleep_stages)
            assert len(predicted_sleep_stages) >= T
            mat['predicted_sleep_stages'] = predicted_sleep_stages[:T]
                
        sio.savemat(feature_path, mat)
                
        #except Exception as ee:
             #msg = str(ee)
           # print(msg)
            #bad_reasons[si] = msg
            
    print(bad_reasons)

