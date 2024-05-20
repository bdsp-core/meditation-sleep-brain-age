from collections import Counter
import datetime
import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from scipy import io as sio
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras import regularizers
import sys
#from segment_EEG import *
#from extract_features_parallel import *
sys.path.insert(0, '/data/brain_age/mycode')
from dnn_regressor import DNNRegressor


epoch_length = 30 # [s]
start_end_remove_epoch_num = 1
amplitude_thres = 500 # [uV]
changepoint_epoch_num = 1
EEG_channels = ['F8-F7', 'F7-O1', 'F8-O2']
line_freq = 60.  # [Hz]
bandpass_freq = [0.5, 20.]  # [Hz]
tostudy_freq = [0.5, 20.]  # [Hz]
random_state = 2
n_jobs = 8


seg_mask_explanation = [
    'normal',
    'around sleep stage change point',
    'NaN in sleep stage',
    'NaN in EEG',
    'overly high/low amplitude',
    'flat signal',
    'NaN in feature',
    'NaN in spectrum',
    'muscle artifact',
    'spurious spectrum']


def load_MGH_dataset(data_path, label_path):
    mat = sio.loadmat(data_path)
    EEG = mat['s']
    channel_names = [mat['hdr']['signal_labels'][0,ii][0] for ii in range(mat['hdr']['signal_labels'].shape[1])]
    ids = [channel_names.index('F3-M2'),  # approximates F7
           channel_names.index('F4-M1'),  # approximates F8
           channel_names.index('O1-M2'),  # O1
           channel_names.index('O2-M1'),  # O2
          ]
    EEG = np.array([EEG[ids[1]]-EEG[ids[0]],  # approximates F8-F7
                    EEG[ids[0]]-EEG[ids[2]],  # approximates F7-O1
                    EEG[ids[1]]-EEG[ids[3]],  # approximates F8-O2
                   ])
                   
    with h5py.File(label_path, 'r') as ff:
        sleep_stages = ff['stage'][()].flatten()
        
    params = {'Fs':200}
    
    return EEG, sleep_stages, params


def myprint(seg_mask):
    sm = Counter(seg_mask)
    for ex in seg_mask_explanation:
        if ex in sm:
            print('%s: %d/%d, %g%%'%(ex,sm[ex],len(seg_mask),sm[ex]*100./len(seg_mask)))


if __name__=='__main__':
    np.random.seed(random_state)
    normal_only = True
    feature_dir = '/data/BHE_PSG_BA_features_for_homedevices/Dreem'
    subject_files = pd.read_csv('/data/brain_age/mycode/data/data_list.txt', sep='\t')

    """
    subject_err_path = 'err_subject_reason_MGH_PSG.txt'
    if os.path.isfile(subject_err_path):
        err_subject_reasons = []
        err_subjects = []
        with open(subject_err_path,'r') as f:
            for row in f:
                if row.strip()=='':
                    continue
                i = row.split(':::')
                err_subjects.append(i[0].strip())
                err_subject_reasons.append(i[1].strip())
    else:
        err_subject_reasons = []
        err_subjects = []

    features = []
    subject_num = len(subject_files)
    for si in range(subject_num):
        data_path = subject_files.signal_file.iloc[si]
        label_path = subject_files.label_file.iloc[si]
        subject_file_name = os.path.basename(subject_files.feature_file.iloc[si])
        feature_path = os.path.join(feature_dir, subject_file_name)
        if subject_file_name in err_subjects:
            continue
        if os.path.isfile(feature_path):
            print('====== [(%d)/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))

        else:
            print('\n====== [%d/%d] %s %s ======'%(si+1,subject_num,subject_file_name.replace('.mat',''),datetime.datetime.now()))
            try:
                # load dataset
                EEG, sleep_stages_, params = load_MGH_dataset(data_path, label_path)
                Fs = params.get('Fs')
                newFs = 200.
                
                # segment EEG
                EEG, segs_, sleep_stages_, seg_times_, seg_mask, specs_, freq, qs = segment_EEG(
                            EEG, sleep_stages_, epoch_length, epoch_length,
                            Fs, EEG_channels, newFs,
                            notch_freq=line_freq, bandpass_freq=bandpass_freq,
                            start_end_remove_window_num=start_end_remove_epoch_num,
                            amplitude_thres=amplitude_thres,
                            to_remove_mean=False, n_jobs=n_jobs)
                if segs_.shape[0] <= 0:
                    raise ValueError('Empty EEG segments')
                Fs = newFs
                
                if normal_only:
                    good_ids = np.where(np.in1d(seg_mask,seg_mask_explanation[:2]))[0]
                    if len(good_ids)<=100:
                        myprint(seg_mask)
                        raise ValueError('<=100 normal segments')
                    segs_ = segs_[good_ids]
                    specs_ = specs_[good_ids]
                    sleep_stages_ = sleep_stages_[good_ids]
                    seg_times_ = seg_times_[good_ids]
                else:
                    good_ids = np.arange(len(seg_mask))

                # extract features

                features_, feature_names = extract_features(
                    segs_, EEG_channels, None, Fs, 2,
                    tostudy_freq, 2, 1,
                    return_feature_names=True,
                    n_jobs=n_jobs, verbose=True)
                features_[np.isinf(features_)] = np.nan
                nan_ids = np.where(np.any(np.isnan(features_),axis=1))[0]
                for ii in nan_ids:
                    seg_mask[ii] = seg_mask_explanation[6]
                if normal_only:
                    good_ids2 = np.where(np.in1d(np.array(seg_mask)[good_ids],seg_mask_explanation[:2]))[0]
                    segs_ = segs_[good_ids2]
                    specs_ = specs_[good_ids2]
                    sleep_stages_ = sleep_stages_[good_ids2]
                    seg_times_ = seg_times_[good_ids2]

                myprint(seg_mask)
                
            except Exception as e:
                err_info = str(e).split('\n')[0].strip()
                print('\n%s.\nSubject %s is IGNORED.\n'%(err_info, subject_file_name))
                err_subject_reasons.append(err_info)
                err_subjects.append(subject_file_name)

                with open(subject_err_path,'a') as f:
                    msg_ = '%s::: %s\n'%(subject_file_name,err_info)
                    f.write(msg_)
                continue

            sio.savemat(feature_path, {
                'EEG_feature_names':feature_names,
                'EEG_features':features_,
                'EEG_specs':specs_,
                'EEG_frequency':freq,
                'sleep_stages':sleep_stages_,
                'seg_times':seg_times_,
                'age':subject_files.age.iloc[si],
                #'sex':subject_files.Sex.iloc[si],
                #'typeoftest':subject_files.TypeOfTest.iloc[si],
                'seg_mask':seg_mask,
                'Fs':Fs,})


    ## build input matrix X

    stages = ['W','N1','N2','N3','R']
    stage2num = {'W':5,'R':4,'N1':3,'N2':2,'N3':1}
    num2stage = {stage2num[x]:x for x in stage2num}

    # load EEG features

    # build BA features
    minimum_epochs_per_stage = 5
    X = []
    ages = []
    for pid in tqdm(range(len(subject_files))):
        feature_path = os.path.join(feature_dir, os.path.basename(subject_files.feature_file.iloc[pid]))
        if not os.path.exists(feature_path):
            continue
        thisdata = sio.loadmat(feature_path)
        sleep_stages = thisdata['sleep_stages'].flatten()
        features = thisdata['EEG_features']
        age = thisdata['age'][0,0]
        
        if pid==0:
            feature_names = np.array(list(map(lambda x:x.strip(), thisdata['EEG_feature_names'])))
            feature_num_each_stage = features.shape[1]

        features = np.sign(features)*np.log1p(np.abs(features))
        features2 = []
        for stage in stages:
            ids = sleep_stages==stage2num[stage]
            if ids.sum()>=minimum_epochs_per_stage:
                features2.append(features[ids].mean(axis=0))
            else:
                features2.append(np.zeros(features.shape[1])+np.nan)
        X.append(np.concatenate(features2))
        ages.append(age)
        
    feature_names = np.array(sum([[fn+'_'+stage for fn in feature_names] for stage in stages], []))
    sio.savemat('training_features.mat', {'X':X,
        'feature_names':feature_names,
        'ages':ages,
        })
    X = np.array(X)
    y = np.array(ages)
    """
    mat = sio.loadmat('training_features.mat')
    X = mat['X']
    y = mat['ages'].flatten()
    feature_names = np.char.strip(mat['feature_names'])
    
    """
    # make the features F7-O1, F8-O2
    ids = np.array(['F8-F7' not in x for x in feature_names])
    X = X[:,ids]
    feature_names = feature_names[ids]
    ids1 = np.array(['F7-O1' in x and not x.startswith('kurtosis') and not x.startswith('mean_gradient') for x in feature_names])
    ids2 = np.array(['F8-O2' in x and not x.startswith('kurtosis') and not x.startswith('mean_gradient') for x in feature_names])
    X[:,ids1] = np.nanmean(np.array([X[:,ids1],X[:,ids2]]), axis=0)
    X = X[:,~ids2]
    feature_names = feature_names[~ids2]
    ids = np.array([not x.startswith('kurtosis') and not x.startswith('mean_gradient') for x in feature_names])
    feature_names[ids] = np.char.replace(feature_names[ids], '_F7-O1', '_F')
    sio.savemat('training_features.mat', {'X':X, 'ages':y, 'feature_names':feature_names})
    """
    
    # remove recordings with missing sleep stages
    ids = ~np.any(np.isnan(X), axis=1)
    X = X[ids]
    y = y[ids]
    
    # train the model
    epochs = 1000
    activation = 'relu'
    n_hidden = []
    valid_patience = [100,80]
    lr = 0.1
    dropout = None
    reg = regularizers.l1(1)
    
    kf = KFold(n_splits=5)
    C1s = [0.,1.,5.,10.,15.,20.]
    losses = [[] for x in range(len(C1s))]
    for trids, teids in kf.split(X):
        Xtr = X[trids]
        Xte = X[teids]
        ytr = y[trids]
        yte = y[teids]
        
        # standardize features
        ss = StandardScaler().fit(Xtr)
        Xtr = ss.transform(Xtr)
        Xte = ss.transform(Xte)
        
        for ci, C1 in enumerate(tqdm(C1s)):
            model = DNNRegressor(n_hidden, batch_size=Xtr.shape[0], C=C1, reg=reg,
                        learning_rate=lr, activation=activation, shuffle=False,
                        valid_patience=valid_patience,
                        max_epoch=epochs, dropout=dropout, summary=False,
                        verbose=False, random_state=random_state+1990)
            model.fit(Xtr, ytr, Xva=Xte, yva=yte, train_va=False,)
                        #sample_weights=sample_weights_tr)
            CA = yte
            BA = model.predict(Xte).flatten()#+ytr_mean
            rmse_va = np.sqrt(np.mean((CA-BA)**2))
            corr_va = pearsonr(CA,BA)[0]
            corr_da_va = pearsonr(CA,BA-CA)[0]
            #coefs = model.model.layers[0].get_weights()[0].flatten()
            loss = -corr_va+np.abs(corr_da_va)#-np.sum(np.abs(coefs)<1e-4)*1./coefs.shape[0]
            losses[ci].append(loss)
            #print('%d current loss: %g, best loss: %g, va RMSE: %g, va corr: %g, va da corr: %g'%(cc, loss, best_loss, rmse_va, corr_va, corr_da_va))
    best_id = np.argmin([np.mean(x) for x in losses])
    best_C1 = C1s[best_id]
    print('best_C1', best_C1)
    
    # standardize features
    ss = StandardScaler().fit(X)
    X = ss.transform(X)
    
    model = DNNRegressor(n_hidden, batch_size=X.shape[0], C=best_C1, reg=reg,
                learning_rate=lr, activation=activation, shuffle=False,
                valid_patience=valid_patience,
                max_epoch=epochs, dropout=dropout, summary=False,
                verbose=False, random_state=random_state+2020)
    model.fit(X, y)
    
    coef, intercept = model.model.get_weights()
    coef = coef.flatten().astype(float)
    intercept = float(intercept[0])
    
    # save ss
    with open('feature_normalizer_dreem.pickle', 'wb') as f:
        pickle.dump([ss.mean_, ss.scale_], f)
    # save model
    with open('BA_model_dreem.pickle', 'wb') as ff:
        pickle.dump([coef, intercept], ff)
        
    # get and save BA adjustment bias values
    BA = np.dot(X, coef) + intercept
    CA = y
    
    age_groups = np.c_[np.arange(15, 75+1, 5),
                       np.arange(20, 80+1, 5)]
    bias = []
    for i in range(len(age_groups)):
        ids = (CA>=age_groups[i][0]) & (CA<=age_groups[i][1])
        bias.append(np.mean(CA[ids]-BA[ids]))
    df = pd.DataFrame(data    = np.c_[age_groups, bias],
                      columns = ['CA_min', 'CA_max', 'bias'])
    df.to_csv('BA_adjustment_bias.csv', index=False)

