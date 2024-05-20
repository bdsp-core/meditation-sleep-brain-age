from collections import Counter
import numpy as np
from scipy.signal import detrend, resample
from scipy.stats import mode
from joblib import Parallel, delayed
from mne.filter import filter_data, notch_filter
#from scikits.samplerate import resample
from mne.time_frequency import psd_array_multitaper


epoch_status_explanation = [
    'normal',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal',
    'spurious spectrum',]


def myprint(epoch_status, channel_names):
    nseg = epoch_status.shape[1]
    for chi in range(len(epoch_status)):
        print(f'[{channel_names[chi]}]')
        sm = Counter(epoch_status[chi])
        for k, v in sm.items():
            print(f'{k}: {v}/{epoch_status.shape[1]}, {v*100./epoch_status.shape[1]:.1f}%')


def segment_EEG(EEG, labels, window_time, step_time, Fs, newFs=200, notch_freq=None, bandpass_freq=None, start_end_remove_window_num=0, amplitude_thres=500, n_jobs=1, to_remove_mean=False, epoch_status='simple', make_artifact_eeg_nan=True, compute_spec=True):
    """Segment EEG signals.

    Arguments:
    EEG -- np.ndarray, size=(channel_num, sample_num)
    labels -- np.ndarray, size=(sample_num,)
    window_time -- in seconds
    step_time -- in seconds
    Fs -- in Hz

    Keyword arguments:
    newFs -- sfreq to be resampled
    notch_freq
    bandpass_freq
    start_end_remove_window_num -- default 0, number of windows removed at the beginning and the end of the EEG signal
    amplitude_thres -- default 500, mark all segments with np.any(EEG_seg>=amplitude_thres)=True
    to_remove_mean -- default False, whether to remove the mean of EEG signal from each channel
    epoch_status -- 'simple': compute based on simple rules; np.ndarray: externally assigned
    make_artifact_eeg_nan -- whether to make artifact EEG segments to be nan
    """
    std_thres = 0.2
    std_thres2 = 1.
    flat_seconds = 5
    padding = 0
    assert (type(epoch_status)==str and epoch_status=='simple') or (type(epoch_status)==np.ndarray)
    
    if to_remove_mean:
        EEG = EEG - np.nanmean(EEG,axis=1, keepdims=True)
    window_size = int(round(window_time*Fs))
    step_size = int(round(step_time*Fs))
    flat_length = int(round(flat_seconds*Fs))
    
    start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
    if start_end_remove_window_num>0:
        start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    labels_ = []
    for si in start_ids:
        labels2 = labels[si:si+window_size]
        labels2[np.isinf(labels2)] = np.nan
        labels2[np.isnan(labels2)] = -1
        label__ = mode(labels2).mode[0]
        if label__==-1:
            labels_.append(np.nan)
        else:
            labels_.append(label__)
    labels = np.array(labels_)
                
    if type(epoch_status)==str and epoch_status=='simple':
        # first assign normal to all epoch status
        epoch_status = np.zeros((len(EEG), len(start_ids))).astype(object)
        epoch_status[:] = epoch_status_explanation[0]
    
        # check nan sleep stage
        if np.any(np.isnan(labels)):
            ids = np.where(np.isnan(labels))[0]
            for i in ids:
                epoch_status[:,i] = epoch_status_explanation[1]
    
    if notch_freq is not None and Fs/2>notch_freq:# and bandpass_freq is not None and np.max(bandpass_freq)>=notch_freq:
        EEG = notch_filter(EEG, Fs, notch_freq, fir_design="firwin", verbose=False)  # (#window, #ch, window_size+2padding)
    if bandpass_freq is None:
        fmin = None
        fmax = None
    else:
        fmin = bandpass_freq[0]
        fmax = bandpass_freq[1]
    if fmax>=Fs/2:
        fmax = None
    if bandpass_freq is not None:
        EEG = filter_data(EEG, Fs, fmin, fmax, fir_design="firwin", verbose=False)#detrend(EEG, axis=1), n_jobs='cuda'
    
    # resample
    if Fs!=newFs:
        #r = newFs*1./Fs
        #EEG = Parallel(n_jobs=n_jobs, verbose=False)(delayed(resample)(EEG[i], r, 'sinc_best') for i in range(len(EEG)))
        EEG = Parallel(n_jobs=n_jobs, verbose=False)(delayed(resample)(EEG[i], int(len(EEG[i])/Fs*newFs)) for i in range(len(EEG)))
        EEG = np.array(EEG).astype(float)
        Fs = newFs
        window_size = int(round(window_time*Fs))
        step_size = int(round(step_time*Fs))
        flat_length = int(round(flat_seconds*Fs))
        start_ids = np.arange(0, EEG.shape[1]-window_size+1, step_size)
        if start_end_remove_window_num>0:
            start_ids = start_ids[start_end_remove_window_num:-start_end_remove_window_num]
    
    #segment into epochs
    EEG_segs = EEG[:,list(map(lambda x:np.arange(x-padding,x+window_size+padding), start_ids))].transpose(1,0,2)  # (#window, #ch, window_size+2padding)
    
    #TODO detrend(EEG_segs)
    #TODO remove_mean(EEG_segs) to remove frequency at 0Hz
    
    if compute_spec:
        NW = 10.
        BW = NW*2./window_time
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = np.inf
        specs, freq = psd_array_multitaper(EEG_segs, Fs, fmin=fmin, fmax=fmax, adaptive=False, low_bias=True, n_jobs=n_jobs, verbose='ERROR', bandwidth=BW, normalization='full')
        specs = specs.transpose(0,2,1)
    
    if type(epoch_status)==str and epoch_status=='simple':
        nan2d = np.any(np.isnan(EEG_segs), axis=2)
        for chi in range(nan2d.shape[1]):
            ids = np.where(nan2d[:,chi])[0]
            for i in ids:
                epoch_status[chi, i] = epoch_status_explanation[2]
       
        saturate_ratios = np.mean(np.abs(EEG_segs)>amplitude_thres, axis=2)
        saturate2d = saturate_ratios>0.1
        for chi in range(saturate2d.shape[1]):
            ids = np.where(saturate2d[:,chi])[0]
            for i in ids:
                epoch_status[chi,i] = epoch_status_explanation[3]
        
        # if there is any flat signal with flat_length
        short_segs = EEG_segs.reshape(EEG_segs.shape[0], EEG_segs.shape[1], EEG_segs.shape[2]//flat_length, flat_length)
        flat2d = np.any(detrend(short_segs, axis=3).std(axis=3)<=std_thres, axis=2)
        flat2d |= (np.std(EEG_segs,axis=2)<=std_thres2)
        for chi in range(flat2d.shape[1]):
            ids = np.where(flat2d[:,chi])[0]
            for i in ids:
                epoch_status[chi,i] = epoch_status_explanation[4]
        
        """
        specs_db = 10*np.log10(specs)
        bad_spec2d = np.any(np.abs(np.diff(np.diff(specs_db, axis=1), axis=1))>3.5, axis=1)
        for chi in range(bad_spec2d.shape[1]):
            ids = np.where(bad_spec2d[:,chi])[0]
            for i in ids:
                epoch_status[chi,i] = epoch_status_explanation[5]
        """
    
    lens = [len(EEG_segs), len(labels), len(start_ids), epoch_status.shape[1]]
    if compute_spec:
        lens.append(len(specs))
    if len(set(lens))>1:
        minlen = min(lens)
        EEG_segs = EEG_segs[:minlen]
        labels = labels[:minlen]
        start_ids = start_ids[:minlen]
        epoch_status = epoch_status[:,:minlen]
        if compute_spec:
            specs = specs[:minlen]
    epoch_status = epoch_status.astype(str)
    
    if make_artifact_eeg_nan: 
        # make signal in artifact epoch/channel into NaN
        # has to do it here because in the feature stage,
        # the channels are already averaged
        for chi in range(len(epoch_status)):
            artifact_ids = np.where(epoch_status[chi]!=epoch_status_explanation[0])[0]
            EEG_segs[artifact_ids, chi] = np.nan
            
    # normalize signal
    q1,q2,q3 = np.nanpercentile(EEG_segs, (25,50,75), axis=(0,2), keepdims=True)
    EEG_segs = (EEG_segs - q2) / (q3-q1)

    if compute_spec:
        return EEG_segs, labels, start_ids, epoch_status, specs, freq
    else:
        return EEG_segs, labels, start_ids, epoch_status

