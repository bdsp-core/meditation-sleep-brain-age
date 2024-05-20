import glob
import os
import pickle
import sys
import subprocess
import numpy as np
import scipy.io as sio
from scipy.signal import convolve
import pandas as pd
import pyedflib
from tqdm import tqdm
sys.path.insert(0, 'myfunctions')
from load_dataset import *
from segment_EEG import segment_EEG


def convert_to_edf_xml(signal_path, label_path, artifact_indicator, dataset, output_path, use_predicted_sleep_stage=False, kernels=None):
    """
    """
    _load_dataset = eval(f'load_{dataset}_dataset')
    EEG, sleep_stages, EEG_channels, combined_EEG_channels, Fs, start_time = _load_dataset(signal_path, label_path)#, really_load=False)
    
    window_time = 30
    window_size = int(round(window_time*Fs))
    line_freq = 60.
    bandpass_freq = [0.5,20]
    n_jobs = 3
    newFs = 200
    epochs, sleep_stages, epoch_start_idx, epoch_status = segment_EEG(EEG, sleep_stages, window_time, window_time, Fs, newFs, notch_freq=line_freq, bandpass_freq=bandpass_freq, n_jobs=n_jobs, compute_spec=False)
    Fs = newFs
    
    if kernels is not None:
        std = np.nanstd(epochs)
        # filter signal to make it like MGH signal
        for stage in stage_bins:
            kernel = kernels[stage]
            ids = sleep_stages==stage
            if np.any(ids):
                epochs[ids] = convolve(np.concatenate([epochs[ids][...,1:], epochs[ids]], axis=-1), kernel.reshape(1,1,-1), mode='valid')
        epochs = epochs/np.nanstd(epochs)*std # scale to original std
        
    EEG = epochs.transpose(1,0,2).reshape(epochs.shape[1],-1)
        
    # read epoch_status from feature_path, do not use non-normal epochs
    sleep_stages[np.isnan(sleep_stages)] = -1
    sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:1, 2:2, 1:3}
    
    # because every channel can have different artifacts,
    # we are doing this for each channel
    edf_paths = []
    xml_paths = []
    for i in range(len(EEG_channels)):
        ch = EEG_channels[i]
        
        edf_path = output_path+f'_{ch}.edf'
        #"""
        channel_info = [
            {'label': ch,
             'dimension': 'uV',
             'sample_rate': Fs,
             'physical_max': 32767,
             'physical_min': -32768,
             'digital_max': 32767,
             'digital_min': -32768,
             'transducer': 'E',
             'prefilter': ''}]
        with pyedflib.EdfWriter(edf_path, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as ff:
            ff.setSignalHeaders(channel_info)
            ff.writeSamples(EEG[[i]])
        #"""

        xml_path = output_path+f'_{ch}.xml'
        sleep_stages2 = np.array(sleep_stages)
        sleep_stages2[artifact_indicator[i]==1] = -1 # make artifact epochs to Movement stage
        with open(xml_path, 'w') as ff:
            ff.write('<CMPStudyConfig>\n')
            ff.write('<EpochLength>30</EpochLength>\n')
            ff.write('<SleepStages>\n')
            for ss in sleep_stages2:
                ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
            ff.write('</SleepStages>\n')
            ff.write('</CMPStudyConfig>')
        
        edf_paths.append(edf_path)
        xml_paths.append(xml_path)
            
    return edf_paths, xml_paths, EEG_channels


def run_luna(output_dir, sids, edf_paths, xml_paths, channels, stage, cfreq=13.5, cycles=12):
    # create the list file
    list_path = os.path.join(output_dir, 'luna.lst')
    df = pd.DataFrame(data={'sid': sids, 'edf':edf_paths, 'xml':xml_paths})
    df = df[['sid', 'edf', 'xml']]
    df.to_csv(list_path, sep='\t', index=False, header=False)
    
    # generate R code to convert luna output .db to .mat
    db_path = os.path.join(output_dir, 'luna_output.db')
    if os.path.exists(db_path):os.remove(db_path)
    xls_path = os.path.join(output_dir, 'luna_output.xlsx')
    if os.path.exists(xls_path):os.remove(xls_path)
    r_code = """library(luna)
library(xlsx)
k<-ldb("%s")
d1<-lx(k,"SPINDLES", "CH_F")
d2<-lx(k,"SPINDLES", "CH")
d3 <- merge(d1,d2,by=c("ID","CH"))
write.xlsx(d3, "%s") 
"""%(db_path, xls_path)
    r_code_path = os.path.join(output_dir, 'convert_luna_output_db2mat.R')
    with open(r_code_path, 'w') as ff:
        ff.write(r_code)
    # maximum spindle duration 10s to account for pediatric subjects
    #with open(os.devnull, 'w') as FNULL:
    subprocess.check_call(['luna', list_path, '-o', db_path, '-s',
        'MASK ifnot=%s & RE & SPINDLES sig=%s max=10 fc=%g cycles=%d so mag=1.5'%(stage, ','.join(channels), cfreq, cycles)],)
    #    stdout=FNULL, stderr=subprocess.STDOUT)
    
    # if no N2 stages, this will throw error
    subprocess.check_call(['Rscript', r_code_path])#, stdout=FNULL, stderr=subprocess.STDOUT)
    df = pd.read_excel(xls_path)
    
    # delete intermediate files
    if os.path.exists(xls_path):
        os.remove(xls_path)
    if os.path.exists(r_code_path):
        os.remove(r_code_path)
    if os.path.exists(db_path):
        os.remove(db_path)
        
    return df
    

if __name__=='__main__':
    epoch_length = 30 # [s]
    line_freq = 60.  # [Hz]
    bandpass_freq = [0.5, 20.]  # [Hz]
    n_jobs = 4
    newFs = 200.
    cfreq = 13.5  # [Hz]
    cycles = 12
    stage = 'N2'
    stage_bins = [1,2,3,4,5]
    
    #fnr = '10%'
    fnr = 'balanced'
    
    # get list of files
    #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
    df_mastersheet = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')
    #from mne.io.edf.edf import _read_edf_header
    #set([tuple(sorted(_read_edf_header(df_mastersheet.signal_path.iloc[i],[],[])[0]['ch_names'])) for i in range(len(df_mastersheet))])
    
    artifact_dir = 'artifact_indicator'
    #sleep_stage_dir = 'spectrogram_data'
    
    # define output folder
    output_feature_dir = f'features_FNR{fnr}_filtered'
    os.makedirs(output_feature_dir, exist_ok=True)
    
    # # generate edf_paths and xml_paths
    # this is usually not a dropbox folder since this will be big
    # must be an absolute path without space!!!
    #TODO this is path on Haoqi's computer
    #TODO to modify, create an empty folder on your computer, it's only for temporal use, ok to delete after running this script
    tmp_dir = '/data/dropbox_dementia_detection_ElissaYe_spindle_results/edf_xml'
    os.makedirs(tmp_dir, exist_ok=True)
    
    # load kernels
    with open('filtering_kernels_Dreem_to_MGH.pickle', 'rb') as ff:
        kernels = pickle.load(ff)
        
    # make a unique key
    df_mastersheet['SID2'] = [os.path.basename(df_mastersheet.signal_path.iloc[i]).replace('.edf','') for i in range(len(df_mastersheet))]
    assert len(df_mastersheet.SID2.unique())==len(df_mastersheet)
    
    # generate edf and xmls files and paths required by Luna
    edf_paths = []
    xml_paths = []
    datatypes = []
    channels = []
    sids = []
    for i in tqdm(range(len(df_mastersheet))):
        sid = df_mastersheet.SID2.iloc[i]
        datatype = df_mastersheet.DataType.iloc[i]
        signal_path = df_mastersheet.signal_path.iloc[i]
        label_path  = df_mastersheet.label_path.iloc[i]
        artifact_path = os.path.join(artifact_dir, os.path.basename(df_mastersheet.feature_path.iloc[i]))
        #sleep_stage_path = os.path.join(sleep_stage_dir, os.path.basename(df_mastersheet.feature_path.iloc[i]))
        age = df_mastersheet.Age.iloc[i]
        sex = 1 if df_mastersheet.Sex.iloc[i]=='M' else 0
        
        kernels_ = {ss:kernels[[x for x in kernels.keys() if x[0][0]<=age and x[0][1]>age and x[1]==sex and x[2]==ss][0]] for ss in stage_bins}
        
        #if use_predicted_sleep_stage:
        #    predicted_label_path = df_mastersheet.predicted_label_path.iloc[i]
        #    if not pd.isna(predicted_label_path):
        #        label_path = predicted_label_path
        
        # the reason to check multiple files is that we are dealing with each channel separately,
        # each channel has a file
        #existing_files = glob.glob(os.path.join(tmp_dir, sid+'*.edf'))
        #if len(existing_files)>0:
        #    edf_paths_ = existing_files
        #    xml_paths_ = [x.replace('.edf','.xml') for x in edf_paths_]
        #    channels_ = [os.path.basename(x).replace(sid+'_','').replace('.edf','') for x in edf_paths_]
        #else:
        #sleep_stages = sio.loadmat(sleep_stage_path, variable_names=['sleep_stages'])['sleep_stages'].flatten()
        artifact_indicator = sio.loadmat(artifact_path, variable_names=[fnr])[fnr]
        #try:
        edf_paths_, xml_paths_, channels_ = convert_to_edf_xml(
            signal_path, label_path, artifact_indicator,
            datatype, os.path.join(tmp_dir, sid), kernels=kernels_)
        #except Exception as ee:
        #    # if anything goes wrong, ignore this file
        #    msg = '[%s]: %s'%(sid, str(ee))
        #    print(msg)
        #    continue
        for f in edf_paths_+xml_paths_:
            assert os.path.exists(f)
        edf_paths.extend(edf_paths_)
        xml_paths.extend(xml_paths_)
        channels.extend(channels_)
        sids.extend([sid]*len(edf_paths_))
        datatypes.extend([datatype]*len(edf_paths_))

    edf_paths = np.array(edf_paths)
    xml_paths = np.array(xml_paths)
    channels = np.array(channels)
    sids = np.array(sids)
    datatypes = np.array(datatypes)
    print(len(edf_paths))

    ## detect spindle and spindle-slow oscillation coupling
    
    unique_channels = set(channels)
    dfs = []
    for unique_channel in unique_channels:
        ids = np.where(channels==unique_channel)[0]
        for id_ in tqdm(ids):
            try:
                df_res = run_luna(
                    tmp_dir,
                    sids[[id_]],
                    edf_paths[[id_]],
                    xml_paths[[id_]],
                    [unique_channel], stage,
                    cfreq=cfreq, cycles=cycles)
                dfs.append(df_res)
            except Exception as ee:
                # if anything goes wrong, ignore this file
                print(f'{unique_channel}, {sids[id_]}: {str(ee)}')
            
    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    
    sid2datatype = {df_mastersheet.SID2.iloc[i]:df_mastersheet.DataType.iloc[i] for i in range(len(df_mastersheet))}
    dfs.insert(2,'DataType',[sid2datatype[dfs.ID.iloc[i]] for i in range(len(dfs))])
    
    dfs.to_csv(os.path.join(output_feature_dir, f'luna_output_{stage}.csv'), index=False)
    
    # convert to one subject per row
    
    unique_datatypes = dfs.DataType.unique()
    for datatype in unique_datatypes:
        _get_channels = eval(f'get_{datatype}_channels')
        ch_names, pair_ch_ids, combined_ch_names = _get_channels()
        
        df = dfs[dfs.DataType==datatype].reset_index(drop=True)
        
        sids = sorted(set(df.ID))
        vals = []
        for sid in sids:
            val = []
            for ids in pair_ch_ids:
                val.append(np.mean(np.r_[
                    df[(df.ID==sid)&(df.CH==ch_names[ids[0]])].iloc[:,5:].values,
                    df[(df.ID==sid)&(df.CH==ch_names[ids[1]])].iloc[:,5:].values,
                ], axis=0))
            vals.append( np.concatenate(val) )
    
        df2 = pd.DataFrame(
            data=np.array(vals), columns=np.concatenate([[x+'_'+ch for x in df.columns[5:]] for ch in combined_ch_names]))
        df2['SID2'] = sids
        
        df3 = df_mastersheet[['SID2']].merge(df2, on='SID2', how='inner')
        df3 = df_mastersheet[['SID2', 'SID', 'DataType','signal_path']].merge(df3, on='SID2', how='left')
        df3 = df3.drop(columns='SID2')
        df3.to_csv(os.path.join(output_feature_dir, f'spindle_features_{stage}_channel_avg_{datatype}.csv'), index=False)

