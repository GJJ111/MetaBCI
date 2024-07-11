import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from brainda.datasets import AlexMI
from brainda.datasets import CRED


from brainda.paradigms import MotorImagery
from brainda.paradigms import Emotion
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import CSP
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet





if __name__ == '__main__':

    dataset = AlexMI()
    paradigm = MotorImagery(events=['right_hand', 'feet'])
    # dataset = CRED()
    # paradigm = Emotion(events=['angry', 'disgust'])

    # add 6-30Hz bandpass filter in raw hook
    #mne prepocess
    # def raw_hook(raw, caches):
    #     # do something with raw object
    #     raw = raw.notch_filter(freqs=(60))
    #     raw = raw.filter(l_freq=0.1, h_freq=30)
        
    #     ica = ICA(max_iter='auto')
    #     raw_for_ica = raw.copy().filter(l_freq=1, h_freq=None)
    #     ica.fit(raw_for_ica)
        
    #     raw.set_eeg_reference(ref_channels='average')
    #     caches['raw_stage'] = caches.get('raw_stage', -1) + 1
    #     events, event_id = mne.events_from_annotations(raw)
    #     epochs = mne.Epochs(raw, events, event_id=2, tmin=-1, tmax=2, baseline=(-0.5, 0),
    #                 preload=True, reject=dict(eeg=2e-4))
    #     evoked = epochs.average()
    #     return raw, caches
    
    
    #  metabci preprocess    
    def raw_hook(raw, caches):
        # do something with raw object
        raw.filter(6, 30, 
        l_trans_bandwidth=2, 
        h_trans_bandwidth=5, 
        phase='zero-double')
        caches['raw_stage'] = caches.get('raw_stage', -1) + 1
        return raw, caches
        
    
    def epochs_hook(epochs, caches):
        # do something with epochs object
        print(epochs.event_id)
        caches['epoch_stage'] = caches.get('epoch_stage', -1) + 1
        return epochs, caches

    def data_hook(X, y, meta, caches):
    # retrive caches from the last stage
        print("Raw stage:{},Epochs stage:{}".format(caches['raw_stage'], caches['epoch_stage']))
        # do something with X, y, and meta
        caches['data_stage'] = caches.get('data_stage', -1) + 1
        return X, y, meta, caches
    
    paradigm.register_raw_hook(raw_hook)
    paradigm.register_epochs_hook(epochs_hook)
    paradigm.register_data_hook(data_hook)

    X, y, meta = paradigm.get_data(
        dataset, 
        subjects=[1], 
        return_concat=True, 
        n_jobs=None, 
        verbose=False)

    # 5-fold cross validation
    set_random_seeds(38)
    kfold = 10
    indices = generate_kfold_indices(meta, kfold=kfold)

    # CSP with SVC classifier
    estimator = make_pipeline(*[
        CSP(n_components=4),
        SVC()
    ])

    accs = []
    for k in range(kfold):
        train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
        # merge train and validate set
        train_ind = np.concatenate((train_ind, validate_ind))
        p_labels = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
        accs.append(np.mean(p_labels==y[test_ind]))
    print(np.mean(accs))
    print(accs)