import os
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.impute import KNNImputer
from mne.io.edf.edf import _read_edf_header
from tqdm import tqdm
sys.path.insert(0, 'myfunctions')
from load_dataset import *
        
        
if __name__=='__main__':
    dataset = 'Dreem'#sys.argv[1].strip()
    fnr = 'balanced'
    
    output_ba_dir = f'BA_results_FNR{fnr}_filtered'
    os.makedirs(output_ba_dir, exist_ok=True)
        
    # get list of subjects
    #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
    df_mastersheet = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')
    df_mastersheet['Age'] = (df_mastersheet.DateOfVisit - df_mastersheet.DateOfBirth).dt.total_seconds()/86400/365
    df_mastersheet = df_mastersheet[df_mastersheet.DataType==dataset].reset_index(drop=True)

    # read features
    feature_dir = f'features_FNR{fnr}_filtered'
    df_feat = pd.read_csv(os.path.join(feature_dir, f'combined_features_{dataset}_FNR{fnr}.csv'))
    assert np.all(df_feat.SID==df_mastersheet.SID)
    
    ### previous brain age
    
    # load brain age model
    brain_age_dir = 'brain_age_model_dreem'
    mat = sio.loadmat(os.path.join(brain_age_dir, 'training_features.mat'))
    feature_names = np.char.strip(mat['feature_names'])
    KNN_K = 10  # number of patients without any missing stage to average
    with open(os.path.join(brain_age_dir, 'feature_normalizer.pickle'), 'rb') as f:
        feature_mean, feature_std = pickle.load(f)
    with open(os.path.join(brain_age_dir, 'BA_model.pickle'),'rb') as ff:
        brain_age_coef, brain_age_intercept = pickle.load(ff)
    df_BA_adj = pd.read_csv(os.path.join(brain_age_dir, 'BA_adjustment_bias.csv'))

    # set intercept to test data average
    # because we are interested in BAI (BA-CA), i.e. we want to eliminate the effect of CA,
    # setting intercept to test data average will make BAI only reflect dot(feature, coef),
    # which is exactly what we want
    brain_age_intercept = df_feat.Age.mean()
    
    X = df_feat[feature_names].values

    # pre-process
    X = (X-feature_mean)/feature_std
    #X = (X - np.nanmean(X, axis=0)) / np.nanstd(X,axis=0)
    if np.any(np.isnan(X)):
        X = KNNImputer(n_neighbors=KNN_K).fit_transform(X)

    # compute brain age
    BA = np.logaddexp(np.dot(X, brain_age_coef)+brain_age_intercept, 0)

    # adjust BA
    BA_adjs = np.zeros(len(df_mastersheet))+np.nan
    for si in range(len(df_mastersheet)):
        CA = df_mastersheet.Age.iloc[si]
        idx = np.where((df_BA_adj.CA_min<=CA)&(df_BA_adj.CA_max>CA))[0][0]
        adj = df_BA_adj.bias.iloc[idx]
        BA_adjs[si] = adj
    
    # we want to make sure that the adjustment does not change mean BA
    BA_adjs = BA_adjs - BA_adjs.mean()
    BA_old = BA + BA_adjs
    
    ### robust brain age
    
    sys.path.insert(0, brain_age_dir)
    with open(os.path.join(brain_age_dir, 'stable_BA_model.pickle'), 'rb') as ff:
        model, feature_names = pickle.load(ff)
        
    # read features
    df_feat = pd.read_csv(os.path.join(feature_dir, f'combined_features_no_log_{dataset}_FNR{fnr}.csv'))
    assert np.all(df_feat.SID==df_mastersheet.SID)
    # read spindle features
    df_feat_sp = pd.read_csv(os.path.join(feature_dir, f'spindle_features_N2_channel_avg_{dataset}.csv'))
    assert np.all(df_feat.SID==df_feat_sp.SID)
    df_feat = pd.concat([df_feat, df_feat_sp.iloc[:,3:]], axis=1)
    
    # match the feature names in model
    df_feat = df_feat.rename(columns={x:x.replace('/','_') for x in df_feat.columns if '/' in x})
    
    stages = ['W','N1','N2','N3','R']
    _get_channels = eval(f'get_{dataset}_channels')
    ch_names, pair_ch_ids, combined_ch_names = _get_channels()
    for c1, c2 in zip(pair_ch_ids, combined_ch_names):
        for stage in stages:
            df_feat[f'kurtosis_{stage}_{c2}'] = (df_feat[f'kurtosis_{ch_names[c1[0]]}_{stage}']+df_feat[f'kurtosis_{ch_names[c1[1]]}_{stage}'])/2
    
    # compute brain age
    # model contains all preprocessing and adjustment steps
    X = df_feat[feature_names].values
    #plt.plot([1,2,3,4,5],model.steps[0][-1].mean_,c='k',marker='o',label='MGH PSG feature mean');plt.plot([1,2,3,4,5],np.nanmean(X,axis=0),c='r',marker='o',label='Dreem v3 after filtering feature mean');plt.legend(frameon=False);plt.xticks([1,2,3,4,5]);plt.ylim([0,400]);seaborn.despine();plt.tight_layout();plt.show()
    
    # !!! Dreem spindle/SO measures seem incorrect -- big deviation from MGH
    # related to signal quality, or F-O montage?
    # so set to MGH mean to eliminate its effect
    fid = feature_names.index('COUPL_OVERLAP_F')
    X[:,fid] = model.steps[0][-1].mean_[fid]

    # set intercept to test data average
    model.steps[2][-1].intercept_ = df_feat.Age.mean()
    
    BA_new = model.predict(X, y=df_feat.Age.values)
    
    df_feat['BA'] = BA_old
    df_feat['robustBA'] = BA_new
    
    
    ## other stuff
    
    # add note
    bad_reasons = [[] for i in range(len(df_feat))]
    bad_ids = np.where(df_feat.NumMissingStage>=2)[0]
    for i in bad_ids:
        bad_reasons[i].append('>=2 missing sleep stages')
        
    artifact_dir = 'artifact_indicator'
    lengths_hour = []
    start_times = []
    for i in range(len(df_feat)):
        signal_path = df_mastersheet.signal_path.iloc[i]
        artifact_path = os.path.join(artifact_dir, os.path.basename(df_mastersheet.feature_path.iloc[i]))
        
        #TODO not use _read_edf_header, use load_dataset
        start_times.append( _read_edf_header(signal_path, [], [])[0]['meas_date'].replace(tzinfo=None) )
        
        artifact_indicator = sio.loadmat(artifact_path, variable_names=[fnr])[fnr]
        lengths_hour.append( artifact_indicator.shape[1]*30/3600 )
    df_feat['StartTime'] = start_times
    df_feat['LengthsHour'] = lengths_hour
    bad_ids = np.where((df_feat.StartTime.dt.hour>=4)&(df_feat.StartTime.dt.hour<=20))[0]
    for i in bad_ids:
        bad_reasons[i].append(f'Unusual sleep time')
    bad_ids = np.where(df_feat.LengthsHour>=10)[0]
    for i in bad_ids:
        bad_reasons[i].append('>=10 hours of recording')
    bad_ids = np.where(df_feat.LengthsHour<=3)[0]
    for i in bad_ids:
        bad_reasons[i].append('<=3 hours of recording')
    
    #df_feat.loc[bad_ids, 'BA'] = np.nan
    for i in range(len(df_feat)):
        df_feat.loc[i, 'Note'] = ', '.join(bad_reasons[i])
    df_feat.loc[df_feat.Note=='', 'Note'] = np.nan
    
    # shift to match CA mean -- no
    #ids = pd.isna(df_feat.Note)
    #df_feat['BA'] = df_feat.BA - df_feat.BA[ids].mean()+df_feat.Age[ids].mean() #-model.steps[2][-1].intercept_
    #df_feat['robustBA'] = df_feat.robustBA - df_feat.robustBA[ids].mean()+df_feat.Age[ids].mean()
    
    # get BAI
    df_feat['BAI'] = df_feat.BA - df_feat.Age
    df_feat['robustBAI'] = df_feat.robustBA - df_feat.Age
    df_feat['StudyTime(HaoqiManual)'] = df_mastersheet['StudyTime(HaoqiManual)'].values

    df_feat = df_feat[['SID', 'DateOfVisit', 'DataType', 'StudyTime(HaoqiManual)', 'Age', 'Sex', 'BA', 'BAI', 
    'robustBA', 'robustBAI', 'NumMissingStage', 'StartTime', 'LengthsHour', 'ArtifactRatio', 'Note']]
    df_feat.to_csv(os.path.join(output_ba_dir, f'BA_{dataset}_FNR{fnr}.csv'), index=False)

