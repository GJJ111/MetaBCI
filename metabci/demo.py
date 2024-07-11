import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from brainda.datasets import CRED
from brainda.paradigms import Emotion
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.decomposition import CSP
dataset = CRED()
delay = 0.14 # seconds
channels = ["M1",
        "M2",
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "CB1",
        "O1",
        "OZ",
        "O2",
        "CB2"  ]
srate = 200 # Hz
duration = 60# seconds
n_bands = 5
#n_harmonics = 5
events = sorted(list(dataset.events.keys()))
freqs = [dataset.get_freq(event) for event in events]
phases = [dataset.get_phase(event) for event in events]

Yf = generate_cca_references(
 freqs, srate, duration,
 phases=None,
 n_harmonics=n_harmonics)

start_pnt = dataset.events[events[0]][1][0]
#重新加载范式
paradigm = SSVEP(
 srate=srate,
 channels=channels,
 intervals=[(start_pnt+delay, start_pnt+delay+duration+0.1)], # more seconds for TDCA
 events=events)

wp = [[8*i, 90] for i in range(1, n_bands+1)]
ws = [[8*i-2, 95] for i in range(1, n_bands+1)]
filterbank = generate_filterbank(
 wp, ws, srate, order=4, rp=1)
filterweights = np.arange(1, len(filterbank)+1)**(-1.25) + 0.25

def data_hook(X, y, meta, caches):
 filterbank = generate_filterbank(
 [[8, 90]], [[6, 95]], srate, order=4, rp=1)
 X = sosfiltfilt(filterbank[0], X, axis=-1)
 return X, y, meta, caches

paradigm.register_data_hook(data_hook)

set_random_seeds(64)
l = 5
models = OrderedDict([
 ('fbscca', FBSCCA(
 filterbank, filterweights=filterweights)),
 ('fbecca', FBECCA(
 filterbank, filterweights=filterweights)),
 ('fbdsp', FBDSP(
 filterbank, filterweights=filterweights)),
 ('fbtrca', FBTRCA(
 filterbank, filterweights=filterweights)),
 ('fbtdca', FBTDCA(
 filterbank, l, n_components=8,
 filterweights=filterweights)),
])

X, y, meta = paradigm.get_data(
dataset,
subjects=[1],
return_concat=True,
n_jobs=1,
verbose=False)
