from collections import defaultdict
import datetime
import pickle
import numpy as np
import os
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')
import sys
sys.path.insert(0, 'myfunctions')
from load_dataset import *
from segment_EEG import segment_EEG, myprint
from extract_features_parallel import extract_features


def get_artifact_indicator_lda(model, thres, specs, specs_db, freqs):
    """
    model:      the trained LDA model
    thres:      this is a dictionary {'1%': a number, '5%': a number, '10%': a number}
    specs:      spectrogram in power, numpy array, shape = (#epochs, #channel, #freq)
    specsi_db:  spectrogram in decibel, numpy array, shape = (#epochs, #channel, #freq)
    freqs:      frequency values, numpy array, shape = (#freq,)

    This function first compute the two features: total power, and 2nd order diff (squareness);
    then feed the features into the model,
    and then it outputs a dictionary {'1%': a boolean array, '5%': a boolean array, '10%': a boolean array}.
    Each boolean array has shape=(#epoch, #channel), where True means artifact.
    """
    # compute features
    
    # feature: total power
    shape = specs.shape
    specs = specs.reshape(-1, shape[-1])
    tp = specs.sum(axis=1) * (freqs[1] - freqs[0])
    tp_db = 10 * np.log10(tp)

    # feature: 2nd order diff for measuring the squareness of spectrum
    specs_db = specs_db.reshape(-1, shape[-1])
    specs_db_n = specs_db / specs_db.std(axis=1, keepdims=True)
    diff2 = np.abs(np.diff(np.diff(specs_db_n, axis=1), axis=1)).max(axis=1)
    diff2_log = np.log(diff2)

    X = np.c_[tp_db, diff2_log]
    yp = model.decision_function(X)

    res = {}
    for k, v in thres.items():
        res[k] = (yp >= v).reshape(shape[:2]).T

    return res


# example code to use this function
if __name__=='__main__':

    # first, load the artifact model
    with open('artifact_model_LDA.pickle', 'rb') as ff:
        res = pickle.load(ff)
    lda_model = res['model']
    lda_thres = {'1%':res['thres_FNR1%'], '5%':res['thres_FNR5%'], '10%':res['thres_FNR10%'], 'balanced':res['best_thres']}
    spec_db_avg_low_thres = res['spec_db_avg_low_thres']

    # then, get the spectrogram

    # get list of files
    #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
    df_mastersheet = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')
    
    # define output folder
    input_spectogram_dir = 'spectrogram_data'
    output_dir = 'artifact_indicator'
    os.makedirs(output_dir, exist_ok=True)

    stage_num2txt = {1:'N3', 2:'N2', 3:'N1', 4:'R', 5:'W'}
    #for each recording
    df_out = defaultdict(list)
    for si in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID.iloc[si]
        dataset = df_mastersheet.DataType.iloc[si]
        age = df_mastersheet.Age.iloc[si]
        sex = df_mastersheet.Sex.iloc[si]
        dov = df_mastersheet.DateOfVisit.iloc[si]
        signal_path = df_mastersheet.signal_path.iloc[si]
        label_path = df_mastersheet.label_path.iloc[si]
        predicted_label_path = df_mastersheet.predicted_label_path.iloc[si]
        in_feature_path = os.path.join(input_spectogram_dir,
                        os.path.basename(df_mastersheet.feature_path.iloc[si].replace('\\',os.sep)))
        out_feature_path = os.path.join(output_dir, os.path.basename(df_mastersheet.feature_path.iloc[si].replace('\\',os.sep)))
        if not os.path.exists(in_feature_path):
            continue

        mat = sio.loadmat(in_feature_path)
        Fs = 200.#mat['newFs'].item()
        freqs = mat['EEG_frequency'].flatten()
        specs = mat['EEG_specs']
        channel_names = mat['channel_names'].flatten()
        epoch_start_idx = mat['epoch_start_idx'].flatten()
        start_time = datetime.datetime.strptime(mat['start_time'].item(), '%Y-%m-%d %H:%M:%S')
        epoch_start_times_str = [(start_time+datetime.timedelta(seconds=x/Fs)).strftime('%Y-%m-%d %H:%M:%S') for x in epoch_start_idx]
        sleep_stages = mat['sleep_stages'].flatten()
        sleep_stages_txt = [stage_num2txt.get(x, 'Unknown') for x in sleep_stages]
        
        #print(specs.shape)
        #print(freqs.shape)
        #specs = specs.reshape(specs.shape[0], specs.shape[2], specs.shape[1])
        specs = specs.transpose(0,2,1)
        #print(specs.shape)
        #print(specs)

        specs_db = 10*np.log10(specs)

        # get artifact indicator from the model
        artifact_indicator = get_artifact_indicator_lda(lda_model, lda_thres, specs, specs_db, freqs)
        
        # combine artifact indicator from the simple criteria
        epoch_status = np.char.strip(mat['epoch_status'])
        #epoch_status = epoch_status.reshape(epoch_status.shape[1], epoch_status.shape[0])
        #print(epoch_status.shape)
        #print(artifact_indicator.shape)

        # also sometimes the total power is too low (TODO: combine this into simple criteria)
        low_power = specs_db.mean(axis=2).T<=spec_db_avg_low_thres
        #print(low_power.shape)
        #low_power = low_power.reshape(low_power.shape[1], low_power.shape[0])
        #print(low_power.shape)
        
        for k in artifact_indicator:
            artifact_indicator[k] |= ( (epoch_status!='normal') | low_power )
        sio.savemat(out_feature_path, artifact_indicator)
        
        df_out['SID'].extend([sid]*len(sleep_stages))
        df_out['DataType'].extend([dataset]*len(sleep_stages))
        df_out['Age'].extend([age]*len(sleep_stages))
        df_out['Sex'].extend([sex]*len(sleep_stages))
        df_out['DateOfVisit'].extend([dov]*len(sleep_stages))
        #df_out['EpochStartIdx'].extend(epoch_start_idx)
        df_out['EpochStartTime'].extend(epoch_start_times_str)
        df_out['SleepStage'].extend(sleep_stages_txt)
        
        for fnr in ['10%', 'balanced']:
            for chi, ch in enumerate(channel_names):
                df_out[f'IsArtifact_FNR{fnr}_{ch}'].extend(artifact_indicator[fnr][chi].astype(int))
        df_out['signal_filename'].extend([os.path.basename(signal_path)]*len(sleep_stages))
    
    df_out = pd.DataFrame(data=df_out)
    df_out.to_csv(os.path.join(output_dir, 'artifact_indicator.csv'), index=False)
    
