from itertools import product
from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
import mat73
import scipy.io as sio
from scipy.signal import resample, convolve
from scipy.optimize import minimize
from scipy.fft import rfft, rfftfreq, irfft
import h5py
from tqdm import tqdm
from mne.time_frequency import psd_array_multitaper
from mne.filter import filter_data, notch_filter
import sys
sys.path.insert(0, 'myfunctions')
from load_dataset import load_Dreem_dataset#, load_MGH_PSG_dataset
sys.path.insert(0, '/data/interesting_side_projects/sleep_general')
from mgh_sleeplab import load_mgh_signal

import matplotlib
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('font', **{'size': 14})
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('ticks')


if __name__ == '__main__':
    
    age_bins = [(20,30), (30,40), (40,50), (50,60)]
    sex_bins = [0,1]
    stage_bins = [1,2,3,4,5]
    stagenum2txt = {1:'N3',2:'N2',3:'N1',4:'R',5:'W'}
    sexnum2txt = {1:'M',0:'F'}
    
    window_time = 30  # [s]
    step_time = 30  # [s]
    NW = 7
    BW = NW*2./window_time
    n_jobs = 8
    notch_freq = 60.  # [Hz]
    fmin =       0.5  # [Hz]
    fmax =       20.  # [Hz]
                
    data_path = 'target_spec_and_input_signal.pickle'
    if os.path.exists(data_path):
        with open(data_path, 'rb') as ff:
            target_specs_db_mt, target_specs_db_fft, target_freqs, Fs_MGH, input_signals, Fs_Dreem = pickle.load(ff)
    else:
        ## create input
        
        # get list of files
        #df_mastersheet = pd.read_excel('subject_list_homedevice.xlsx')
        df = pd.read_excel('subject_list_homedevice_haoqi_computer_path.xlsx')
        df.loc[df.Sex=='F','Sex'] = 0
        df.loc[df.Sex=='M','Sex'] = 1
        
        input_signals = {}
        for age_bin, sex in product(age_bins, sex_bins):
            print(age_bin, sex)
            ids = np.where((df.Age>=age_bin[0])&(df.Age<age_bin[1])&(df.Sex==sex))[0]
            segs = []
            ss = []
            for idx in tqdm(ids):
                signal_path = df.signal_path.iloc[idx]
                label_path = df.label_path.iloc[idx]
                EEG, sleep_stage_, EEG_channels, combined_EEG_channels, Fs_Dreem, start_time = load_Dreem_dataset(signal_path, label_path)
                window_size = int(round(window_time*Fs_Dreem))
                step_size = int(round(step_time*Fs_Dreem))
                start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
                segs_ = EEG[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)
                sleep_stage_ = sleep_stage_.reshape(1,-1)[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))][0,:,window_size//2]
                segs.append(segs_)
                ss.append(sleep_stage_)
            segs = np.concatenate(segs)
            ss = np.concatenate(ss)
            for stage in stage_bins:
                input_signals[(age_bin, sex, stage)] = segs[ss==stage].astype('float32')
        
        ## create target
        
        """
        all_spec_data_path = '/data/brain_age_descriptive/all_data_AHI15_Diag.h5'
        with h5py.File(all_spec_data_path, 'r') as ff:
            sids = ff['subject'][()].astype(str)
            ages = ff['age'][()]
            sexs = ff['sex'][()]
            Fs_MGH = ff['Fs'][()]
            #stages = ff['sleep_stage'][()]
            #freqs = ff['freq'][()]
            #channel_names = ff['channelname'][()].astype(str)
        
        # take MGH only
        mgh_ids = np.where(np.char.startswith(sids, 'Feature_TwinData'))[0]
        sids = sids[mgh_ids]
        ages = ages[mgh_ids]
        sexs = sexs[mgh_ids]
        #stages = stages[mgh_ids]
        
        # take unique subjects
        sids2ids = defaultdict(list)
        for i, sid in enumerate(sids):
            sids2ids[sid].append(i)
        unique_sids = np.array(list(sids2ids.keys()))
        unique_ages = np.array([ages[sids2ids[sid][0]] for sid in unique_sids])
        unique_sexs = np.array([sexs[sids2ids[sid][0]] for sid in unique_sids])
    
        feature_folder = '/data/brain_age_descriptive/MGH_features'
        target_specs_db = {}
        for age_bin, sex in product(age_bins, sex_bins):
            print(age_bin, sex)
            ids = (unique_ages>=age_bin[0])&(unique_ages<age_bin[1])&(unique_sexs==sex)
            this_unique_sids = unique_sids[ids]
            res = defaultdict(list)
            for sid in tqdm(this_unique_sids):
                mat = sio.loadmat(os.path.join(feature_folder, sid), variable_names=['EEG_specs', 'sleep_stages', 'EEG_frequency'])
                this_sleep_stages = mat['sleep_stages'].flatten()
                target_freqs = mat['EEG_frequency'].flatten()
                this_specs_db = 10*np.log10(mat['EEG_specs'][...,:2])  # take frontal channels
                for stage in stage_bins:
                    res[stage].append(np.nanmean(this_specs_db[this_sleep_stages==stage], axis=0))
            for stage in stage_bins:
                target_specs_db[(age_bin, sex, stage)] = np.nanmean(res[stage], axis=0).T.astype('float32')  # channelxfreq
        """
        mastersheet_path = '/data/brain_age_descriptive/mycode/data/MGH_data_list.txt'
        df = pd.read_csv(mastersheet_path, sep='\t')
        df = df[df.state=='good'].reset_index(drop=True)
        df.loc[df.sex=='F','sex'] = 0
        df.loc[df.sex=='M','sex'] = 1
        
        target_specs_db_fft = {}
        target_specs_db_mt = {}
        for age_bin, sex in product(age_bins, sex_bins):
            print(age_bin, sex)
            ids = np.where((df.age>=age_bin[0])&(df.age<age_bin[1])&(df.sex==sex))[0]
            res_fft = defaultdict(list)
            res_mt = defaultdict(list)
            for idx in tqdm(ids):
                signal_path = df.signal_file.iloc[idx].replace('/Projects/', '/Projects_NEW/')
                label_path = df.label_file.iloc[idx].replace('/Projects/', '/Projects_NEW/')
                sid = os.path.basename(signal_path)
                
                EEG, params = load_mgh_signal(signal_path, channels=['F3','F4'], return_signal_dtype='array')
                Fs_MGH = params['Fs']
                sleep_stage_ = mat73.loadmat(label_path, only_include=['stage'])['stage'].flatten()
                #EEG, sleep_stage_, EEG_channels, combined_EEG_channels, Fs_MGH, start_time = load_MGH_PSG_dataset(signal_path, label_path)
                
                window_size = int(round(window_time*Fs_MGH))
                step_size = int(round(step_time*Fs_MGH))
                segs_ = notch_filter(segs_, Fs_MGH, notch_freq, fir_design="firwin", verbose=False)
                segs_ = filter_data(segs_, Fs_MGH, fmin, fmax, fir_design="firwin", verbose=False)
                start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
                segs_ = EEG[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))].transpose(1,0,2)
                sleep_stage_ = sleep_stage_.reshape(1,-1)[:,list(map(lambda x:np.arange(x,x+window_size), start_ids))][0,:,window_size//2]
                
                segs_ = notch_filter(segs_, Fs_MGH, notch_freq, fir_design="firwin", verbose=False)
                segs_ = filter_data(segs_, Fs_MGH, fmin, fmax, fir_design="firwin", verbose=False)
        
                spec_mt_, target_freqs = psd_array_multitaper(segs_, Fs_MGH, adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
                spec_fft_ = np.abs(rfft(segs_, axis=-1))
                #target_freqs = rfftfreq(segs_.shape[-1], 1./Fs_MGH)
                spec_db_mt_ = 10*np.log10(spec_mt_)
                spec_db_fft_ = 10*np.log10(spec_fft_)
                spec_db_mt_[np.isinf(spec_db_mt_)] = np.nan
                spec_db_fft_[np.isinf(spec_db_fft_)] = np.nan
                
                for stage in stage_bins:
                    res_mt[stage].append(np.nanmean(spec_db_mt_[sleep_stage_==stage], axis=(0,1)))
                    res_fft[stage].append(np.nanmean(spec_db_fft_[sleep_stage_==stage], axis=(0,1)))
                
            for stage in stage_bins:
                target_specs_db_mt[(age_bin, sex, stage)] = np.nanmean(res_mt[stage], axis=0).astype('float32')  # channelxfreq
                target_specs_db_fft[(age_bin, sex, stage)] = np.nanmean(res_fft[stage], axis=0).astype('float32')  # channelxfreq
                
        with open(data_path, 'wb') as ff:
            pickle.dump([target_specs_db_mt, target_specs_db_fft, target_freqs, Fs_MGH, input_signals, Fs_Dreem], ff)
            
        
    np.random.seed(2022)
    figure_folder = 'plots'
    
    freq_ids = (target_freqs>=0.5)&(target_freqs<=20)
    kernels = {}
    for age_bin, sex, stage in product(age_bins, sex_bins, stage_bins):
        print(age_bin, sex, stage)
        input_signal = input_signals[(age_bin, sex, stage)].astype(float)
        target_spec_db_fft  = target_specs_db_fft[(age_bin, sex, stage)].astype(float)
        target_spec_db_mt  = target_specs_db_mt[(age_bin, sex, stage)].astype(float)
        input_signal = resample(input_signal, int(round(input_signal.shape[-1]/Fs_Dreem*Fs_MGH)), axis=-1)
        
        input_spec_mt, input_freqs = psd_array_multitaper(input_signal, Fs_MGH, fmin=target_freqs.min(), fmax=target_freqs.max(), adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
        assert np.allclose(input_freqs, target_freqs)
        input_spec_db_mt = 10*np.log10(input_spec_mt)
        
        input_spec_fft = np.abs(rfft(input_signal, axis=-1))
        input_spec_db_fft = 10*np.log10(input_spec_fft)
        input_spec_fft = 10**(input_spec_db_fft.mean(axis=(0,1))/10)
        target_spec_fft = 10**(target_spec_db_fft/10)
        f_kernel = target_spec_fft/input_spec_fft
        kernel = irfft(f_kernel)#, n=int(round(Fs_MGH))*10)
        kernels[(age_bin, sex, stage)] = kernel
        
        input_signal_f = convolve(np.concatenate([input_signal[...,1:], input_signal], axis=-1), kernel.reshape(1,1,-1), mode='valid')
        #scale = input_signal.std(axis=-1).mean()/input_signal_f.std(axis=-1).mean()
        #kernel *= scale
        #input_signal_f = convolve(np.concatenate([input_signal[...,1:], input_signal], axis=-1), kernel.reshape(1,1,-1), mode='valid')
        
        input_spec_f_fft = np.abs(rfft(input_signal_f, axis=-1))
        input_spec_f_db_fft = 10*np.log10(input_spec_f_fft)
        
        input_spec_f_mt, input_freqs_f_mt = psd_array_multitaper(input_signal_f, Fs_MGH, fmin=target_freqs.min(), fmax=target_freqs.max(), adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
        input_spec_f_db_mt = 10*np.log10(input_spec_f_mt)
    
        """
        plt.close()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(target_freqs, target_spec_db_fft,c='k', label='average spectra of MGH PSG')
        ax.plot(target_freqs, input_spec_db_fft.mean(axis=(0,1)),c='r',label='average spectra of Dreem before filtering')
        ax.plot(target_freqs, input_spec_f_db_fft.mean(axis=(0,1))-1,c='b',label='average spectra of Dreem after filtering')
        ax.legend(frameon=False)
        ax.set_xlim(0,20)
        ax.set_ylim(-10,35)
        ax.set_xlabel('Hz')
        ax.set_ylabel('dB')
        seaborn.despine()
        plt.tight_layout()
        plt.show()
        """
        plt.close()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(target_freqs[freq_ids], target_spec_db_mt[freq_ids], c='k', label='average spectra of MGH PSG')
        ax.plot(target_freqs[freq_ids], input_spec_db_mt.mean(axis=(0,1))[freq_ids], c='r',label='average spectra of Dreem before filtering')
        ax.plot(target_freqs[freq_ids], input_spec_f_db_mt.mean(axis=(0,1))[freq_ids], c='b',label='average spectra of Dreem after filtering')
        ax.legend(frameon=False)
        ax.set_xlim(0,20)
        ax.set_ylim(-10,40)
        ax.set_xlabel('Hz')
        ax.set_ylabel('dB')
        seaborn.despine()
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(figure_folder, f'filter_results_age{age_bin}_sex{sexnum2txt[sex]}_stage{stagenum2txt[stage]}.png'), bbox_inches='tight', pad_inches=0.03)
        
    with open('filtering_kernels_Dreem_to_MGH.pickle', 'wb') as ff:
        pickle.dump(kernels, ff)
        
