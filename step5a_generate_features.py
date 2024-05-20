import os
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import convolve
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


if __name__=='__main__':
    epoch_length = 30 # [s]
    line_freq = 60.  # [Hz]
    bandpass_freq = [0.5, 20.]  # [Hz]
    n_jobs = 4
    newFs = 200.
    stage_bins = [1,2,3,4,5]
    
    #fnr = '10%'
    fnr = 'balanced'
    
    # get list of files
    #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
    df_mastersheet = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')
    # define output folder
    artifact_dir = 'artifact_indicator'
    output_feature_dir = f'features_FNR{fnr}_filtered'
    os.makedirs(output_feature_dir, exist_ok=True)
    
    # load kernels
    with open('filtering_kernels_Dreem_to_MGH.pickle', 'rb') as ff:
        kernels = pickle.load(ff)
    
    # for each recording
    for si in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID.iloc[si]
        dataset = df_mastersheet.DataType.iloc[si]
        signal_path = df_mastersheet.signal_path.iloc[si]
        label_path = df_mastersheet.label_path.iloc[si]
        age = df_mastersheet.Age.iloc[si]
        sex = 1 if df_mastersheet.Sex.iloc[si]=='M' else 0
        
        artifact_path = os.path.join(artifact_dir, os.path.basename(df_mastersheet.feature_path.iloc[si]))
        out_feature_path = os.path.join(output_feature_dir, os.path.basename(df_mastersheet.feature_path.iloc[si]))
        
        try:
            if not os.path.exists(artifact_path):
                raise ValueError(f'Not found: {artifact_path}')
                    
            # get epoch_status from artifact indicator
            mat = sio.loadmat(artifact_path)
            artifact = mat[fnr]
            epoch_status = np.empty_like(artifact, dtype=object)
            epoch_status[:] = 'normal'
            epoch_status[artifact==1] = 'abrnormal'
            
            # load dataset
            _load_dataset = eval(f'load_{dataset}_dataset')
            EEG, sleep_stages, EEG_channels, combined_EEG_channels, Fs, start_time = _load_dataset(signal_path, label_path)

            # segment EEG
            epochs, sleep_stages, epoch_start_idx, epoch_status = segment_EEG(EEG, sleep_stages, epoch_length, epoch_length, Fs, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, n_jobs=n_jobs, epoch_status=epoch_status, compute_spec=False)
            Fs = newFs
            std = np.nanstd(epochs)
            
            # filter signal to make it like MGH signal
            for stage in stage_bins:
                kernel = kernels[[x for x in kernels.keys() if x[0][0]<=age and x[0][1]>age and x[1]==sex and x[2]==stage][0]]
                ids = sleep_stages==stage
                if np.any(ids):
                    epochs[ids] = convolve(np.concatenate([epochs[ids][...,1:], epochs[ids]], axis=-1), kernel.reshape(1,1,-1), mode='valid')
            epochs = epochs/np.nanstd(epochs)*std # scale to original std
            
            # extract brain age features
            features, feature_names = extract_features(
                epochs, Fs, EEG_channels, 2,
                2, 1, return_feature_names=True,
                combined_channel_names=combined_EEG_channels,
                n_jobs=n_jobs, verbose=True)
            
            myprint(epoch_status, EEG_channels)
            sio.savemat(out_feature_path, {
                #'start_time':start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'EEG_feature_names':feature_names,
                'EEG_features':features,
                #'EEG_specs':specs,
                #'EEG_frequency':freq,
                'channel_names':EEG_channels,
                'sleep_stages':sleep_stages,
                #'epoch_start_idx':epoch_start_idx,
                #'age':df_mastersheet.Age.iloc[si],
                #'sex':df_mastersheet.Sex.iloc[si],
                'epoch_status':epoch_status,
                #'Fs':Fs,
                })
                
        except Exception as ee:
            msg = str(ee)
            print(msg)
    
    # get stage-averaged features
    minimum_epochs_per_stage = 3
    stages = ['W','N1','N2','N3','R', 'NREM']
    stage2num = {'W':[5],'R':[4],'N1':[3],'N2':[2],'N3':[1], 'NREM':[1,2,3]}
    
    auto_sleep_stage = ''
    datasets = df_mastersheet.DataType.unique()
    for dataset in datasets:
        print(f'converting into stage-averaged features for {dataset}')
        
        df = df_mastersheet[df_mastersheet.DataType==dataset].reset_index(drop=True)
        ba_features = []
        ba_features_no_log = []
        artifact_ratios = []
        num_missing_stages = []
        for si in tqdm(range(len(df))):
            sid = df.SID.iloc[si]
            feature_path = os.path.join(output_feature_dir, os.path.basename(df.feature_path.iloc[si]))
            
            # if for any reason the feature file does not exist,
            # fill this row with nan
            if not os.path.exists(feature_path):
                D = len(ba_features[-1])  #TODO assume the previous feature file is found
                ba_features.append([np.nan] * D)
                ba_features_no_log.append([np.nan] * D)
                artifact_ratios.append(np.nan)
                num_missing_stages.append(np.nan)
                continue
                
            mat = sio.loadmat(feature_path)
            features = mat['EEG_features']
            if auto_sleep_stage and 'predicted_sleep_stages' in mat.keys():
                sleep_stages = mat['predicted_sleep_stages'].flatten()
            else:
                sleep_stages = mat['sleep_stages'].flatten()
            channel_names = np.char.strip(mat['channel_names'])
            feature_names = np.char.strip(mat['EEG_feature_names'])
            epoch_status = np.char.strip(mat['epoch_status'])
    
            #TODO assumes channel arrangement is [left, right, left, right,...]
            epoch_status2 = np.array([epoch_status[::2], epoch_status[1::2]])
            artifact_ratio = 1 - np.any(epoch_status2 == 'normal', axis=0).mean()
            #artifact_ratio = 1-np.mean(epoch_status=='normal')
                
            # log-transform brain age features
            features_no_log = np.array(features)
            features = np.sign(features) * np.log1p(np.abs(features))
            
            # average features across sleep stages
            X = []
            X_no_log = []
            num_missing_stage = 0
            for stage in stages:
                ids = np.in1d(sleep_stages, stage2num[stage])
                if ids.sum()>=minimum_epochs_per_stage:
                    X.append(np.nanmean(features[ids], axis=0))
                    X_no_log.append(np.nanmean(features_no_log[ids], axis=0))
                else:
                    X.append(np.zeros(features.shape[1]) + np.nan)
                    X_no_log.append(np.zeros(features.shape[1]) + np.nan)
                if stage!='NREM' and np.all(np.isnan(X[-1])): # only for 5 stages
                    num_missing_stage += 1
            X = np.concatenate(X)
            X_no_log = np.concatenate(X_no_log)
            
            ba_features.append(X)
            ba_features_no_log.append(X_no_log)
            artifact_ratios.append(artifact_ratio)
            num_missing_stages.append(num_missing_stage)
            
        df['ArtifactRatio'] = artifact_ratios
        df['NumMissingStage'] = num_missing_stages
        
        cols = np.concatenate([[x.strip() + '_' + stage for x in feature_names] for stage in stages])
        df_feat = pd.DataFrame(data=np.array(ba_features), columns=cols)
        df_feat = pd.concat([df[['SID', 'DataType', 'Age', 'Sex', 'DateOfBirth', 'DateOfVisit', 'ArtifactRatio', 'NumMissingStage']], df_feat], axis=1)
        df_feat.to_csv(os.path.join(output_feature_dir, f'combined_features_{dataset}_FNR{fnr}{auto_sleep_stage}.csv'), index=False)
        
        df_feat.loc[:,cols] = np.array(ba_features_no_log)
        df_feat.to_csv(os.path.join(output_feature_dir, f'combined_features_no_log_{dataset}_FNR{fnr}{auto_sleep_stage}.csv'), index=False)
        
