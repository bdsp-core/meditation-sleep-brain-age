import datetime
import numpy as np
import pandas as pd
import mne
from mne.io.edf.edf import _get_info


def get_MGH_PSG_brain_age_dir():
    return 'brain_age_model_mgh'
    
def get_MGH_PSG_channels():
    return \
        ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'],\
        [[0,1],[2,3],[4,5]],\
        ['F', 'C', 'O']
        
def load_MGH_PSG_dataset(signal_path, annot_path=None, epoch_sec=30, really_load=True):

    if really_load:
        edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
        info = edf.info
    else:
        info, _, _ = _get_info(signal_path, None, None,None, (), False)
    Fs = info['sfreq']
    start_time = info['meas_date'].replace(tzinfo=None)
    #all_ch_names = info['ch_names']
    
    if really_load:
        # make sure channel_names is always close to F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, this is what is used in the brain age model
        ch_names = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'M1', 'M2']
        signals = edf.get_data(picks=ch_names)
        signals = np.array([
            signals[ch_names.index('F3')] - signals[ch_names.index('M2')],
            signals[ch_names.index('F4')] - signals[ch_names.index('M1')],
            signals[ch_names.index('C3')] - signals[ch_names.index('M2')],
            signals[ch_names.index('C4')] - signals[ch_names.index('M1')],
            signals[ch_names.index('O1')] - signals[ch_names.index('M2')],
            signals[ch_names.index('O2')] - signals[ch_names.index('M1')],
            ])
    
        # mne automatically converts to V, convert back to uV
        signals *= 1e6
    
        if annot_path is None: # get from edf.annotations
            stagetxt2int = {
                    'W':5,
                    'R':4,'REM':4,
                    'N1':3,'NREM1':3,
                    'N2':2,'NREM2':2,
                    'N3':1,'NREM3':1}
            sleep_stages = np.zeros(signals.shape[1])+np.nan
            for i in range(len(edf.annotations)):
                desc = edf.annotations.description[i].lower()
                #if not ('sleep' in desc and 'stag' in desc):
                if 'sleep stage' not in desc:
                    continue
                stage = desc.split(' ')[-1].upper()
                start = int(round(edf.annotations.onset[i]*Fs))
                end = int(round((edf.annotations.onset[i]+edf.annotations.duration[i])*Fs))
                start = max(0, start)
                end = min(len(sleep_stages), end)
                sleep_stages[start:end] = stagetxt2int.get(stage, np.nan)
        else:
            ss_df = pd.read_csv(annot_path)
            sleep_stages = ss_df['sleep stage corrected'].values
            sleep_stages = np.repeat(sleep_stages, int(round(30*Fs)))
            
            T = min(signals.shape[1], len(sleep_stages))
            signals = signals[:,:T]
            sleep_stages = sleep_stages[:T]
    else:
        signals = None
        sleep_stages = None
        
    # assumes EEG_channels = [F3M2, F4M1, C3M2, C4M1, O1M2, O2M1]
    # when computing features, the spectral features were averaged across left and right
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, pair_ch_ids, combined_ch_names = get_MGH_PSG_channels()
        
    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time


def get_Prodigy_brain_age_dir():
    return 'brain_age_model_prodigy'
    
def get_Prodigy_channels():
    return \
        ['LM', 'RM'],\
        [[0,1]],\
        ['F']

def load_Prodigy_dataset(signal_path, annot_path=None, epoch_sec=30, really_load=True):
    if really_load:
        edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
        info = edf.info
    else:
        info, _, _ = _get_info(signal_path, None, None,None, (), False)
    Fs = info['sfreq']
    start_time = info['meas_date'].replace(tzinfo=None)
    #all_ch_names = info['ch_names']
    
    if really_load:
        # make sure channel_names is always close to F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, this is what is used in the brain age model
        ch_names = ['L-EEG', 'R-EEG', 'Mastoid']
        signals = edf.get_data(picks=ch_names)
        signals = np.array([
            signals[ch_names.index('L-EEG')] - signals[ch_names.index('Mastoid')],
            signals[ch_names.index('R-EEG')] - signals[ch_names.index('Mastoid')],
            ])
        
        # mne automatically converts to V, convert back to uV
        signals *= 1e6
        
        # load labels
        if annot_path is None:
            sleep_stages = None
        else:
            ss_df = pd.read_csv(annot_path)
            sleep_stages = ss_df['sleep stage corrected'].values
            sleep_stages = np.repeat(sleep_stages, int(round(30*Fs)))
            
            T = min(signals.shape[1], len(sleep_stages))
            signals = signals[:,:T]
            sleep_stages = sleep_stages[:T]
    else:
        signals = None
        sleep_stages = None
    
    # when computing features, the spectral features were averaged across left and right
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, pair_ch_ids, combined_ch_names = get_Prodigy_channels()
        
    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time



def get_Dreem_brain_age_dir():
    return 'brain_age_model_dreem'
    
def get_Dreem_channels():
    return \
        ['F7-O1', 'F8-O2'],\
        [[0,1]],\
        ['F']
        
def load_Dreem_dataset(signal_path, annot_path=None, epoch_sec=30, really_load=True):
    if really_load:
        edf = mne.io.read_raw_edf(signal_path, preload=False, verbose=False, stim_channel=None)
        info = edf.info
    else:
        info, _, _ = _get_info(signal_path, None, None,None, (), False)
    Fs = info['sfreq']
    start_time = info['meas_date'].replace(tzinfo=None)
    all_ch_names = info['ch_names']
    
    # combined_EEG_channels is to define the name of the averaged features
    ch_names, pair_ch_ids, combined_ch_names = get_Dreem_channels()
    
    if really_load:
        """ # combinations appeared
'EEG F7-O1', 'EEG F8-F7', 'EEG F8-O2', 'EEG Fp1-F7', 'EEG Fp1-F8', 'EEG Fp1-O1', 'EEG Fp1-O2'
'EEG F7-O1', 'EEG F8-F7', 'EEG F8-O2', 'EEG Fpz-F7', 'EEG Fpz-F8', 'EEG Fpz-O1', 'EEG Fpz-O2'
'EEG F7-O1', 'EEG F7-O2', 'EEG F8-F7', 'EEG F8-O1', 'EEG F8-O2'
        """
        # when computing features, the spectral features were averaged across left and right
        ch_f7o1 = [x for x in all_ch_names if 'F7-O1' in x][0]
        ch_f8o2 = [x for x in all_ch_names if 'F8-O2' in x][0]
        ch_fp1o1 = [x for x in all_ch_names if 'Fp1-O1' in x or 'Fp1-01' in x or 'Fpz-01' in x or 'Fpz-O1' in x]
        if len(ch_fp1o1)==0:
            ch_fp1o1 = ch_f7o1
        else:
            ch_fp1o1 = ch_fp1o1[0]
        ch_fp1o2 = [x for x in all_ch_names if 'Fp1-O2' in x or 'Fp1-02' in x or 'Fpz-02' in x or 'Fpz-O2' in x]
        if len(ch_fp1o2)==0:
            ch_fp1o2 = ch_f8o2
        else:
            ch_fp1o2 = ch_fp1o2[0]
        signals = edf.get_data(picks=[ch_f7o1, ch_f8o2, ch_fp1o1, ch_fp1o2])
        
        # mne automatically converts to V, convert back to uV
        signals *= 1e6
        
        # determine use F7-O1 or Fp1-O1 based on %more than 490uV (Dreem clips at 500uV)
        high_ratio_F  = np.mean(np.abs(signals[0])>490)
        high_ratio_Fp = np.mean(np.abs(signals[2])>490)
        id1 = 0 if high_ratio_F<high_ratio_Fp else 2
        high_ratio_F  = np.mean(np.abs(signals[1])>490)
        high_ratio_Fp = np.mean(np.abs(signals[3])>490)
        id2 = 1 if high_ratio_F<high_ratio_Fp else 3
        signals = signals[[id1, id2]]
        
        ## load labels
        if annot_path is None:
            sleep_stages = None
        else:
            start_row = 0
            starttime = None
            with open(annot_path, 'r') as ff:
                for row in ff:
                    start_row += 1
                    if 'scorer time' in row.lower():
                        starttime = datetime.datetime.strptime(row.lower().split('scorer time:')[-1].strip(), '%m/%d/%y - %H:%M:%S')
                    elif row.strip()=='Sleep Stage\tTime [hh:mm:ss]\tEvent\tDuration[s]':
                        break
            assert starttime is not None, 'No scoring start time.'
            
            ss_df = pd.read_csv(annot_path, skiprows=start_row-1, sep='\t')
            ss_df = ss_df.rename(columns={'Time [hh:mm:ss.ms]':'Time [hh:mm:ss]'})
            ss_df['Time [hh:mm:ss]'] = pd.to_datetime(ss_df['Time [hh:mm:ss]'])
            ss_df.loc[ss_df['Event']=='SLEEP-MT', 'Event'] = np.nan
            ss_df.loc[ss_df['Event']=='SLEEP-REM', 'Event'] = 4
            ss_df.loc[ss_df['Event']=='SLEEP-S0', 'Event'] = 5
            ss_df.loc[ss_df['Event']=='SLEEP-S1', 'Event'] = 3
            ss_df.loc[ss_df['Event']=='SLEEP-S2', 'Event'] = 2
            ss_df.loc[ss_df['Event']=='SLEEP-S3', 'Event'] = 1
            
            # align signals and sleep stages
            #ss_starttime = datetime.datetime.combine(starttime, ss_df['Time [hh:mm:ss]'].iloc[0].to_pydatetime().time())
            #assert starttime.minute==ss_starttime.minute and starttime.second==ss_starttime.second, 'Start times in signal and sleep stage do not match.'
            oneday = datetime.timedelta(days=1)
            sleep_stages = np.zeros(signals.shape[1])+np.nan
            dt = datetime.timedelta(seconds=0)
            for i in range(len(ss_df)):
                this_time = ss_df['Time [hh:mm:ss]'].iloc[i].to_pydatetime()
                if i==0:
                    start_from_night = this_time.hour>=18
                if start_from_night and this_time.hour<=2:
                    dt = oneday
                this_time = datetime.datetime.combine(starttime+dt, this_time.time())
                start_sec = (this_time - starttime).total_seconds()
                start = int(round(start_sec*Fs))
                end   = int(round((start_sec+ss_df['Duration[s]'].iloc[i])*Fs))
                if end<=0:
                    continue
                elif start<0 and end>0:
                    start = 0
                elif start>=len(sleep_stages):
                    continue
                elif start<len(sleep_stages) and end>len(sleep_stages):
                    end = len(sleep_stages)
                sleep_stages[start:end] = ss_df['Event'].iloc[i]
    else:
        signals = None
        sleep_stages = None

    return signals, sleep_stages, ch_names, combined_ch_names, Fs, start_time

