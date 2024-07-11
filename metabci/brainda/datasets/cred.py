# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2020/12/27
# License: MIT License
"""
PeiYu  dataset.
"""
from typing import Union, Optional, Dict, List, cast
from pathlib import Path

from mne.io import Raw, read_raw_eeglab
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path
from ..utils.channels import upper_ch_names

CRED_URL = "https://xxxx/200Hz_rawdata/"
# C:\Users\Levovo\mne_data\MNE-cred-data\200Hz_rawdata

class CRED(BaseDataset):


    _EVENTS = {"angry": (1, (0, 60)), "disgust": (2, (0, 60)), "fear": (3, (0, 60)), "neutral": (4, (0, 60)), "sad": (5, (0, 60))}

    _CHANNELS = [
        "M1",
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
        "CB2"  
    ]

    def __init__(self):
        super().__init__(
            dataset_code="CRED",
            subjects=list(range(1, 25)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=200,
            paradigm="Emotion",#情绪识别数据集
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:
        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))

        subject = cast(int, subject)
        url = "{:s}{:02d}.set".format(CRED_URL, subject)
        dests = [
            [
                mne_data_path(
                    url,
                    self.dataset_code,
                    path=path,
                    proxies=proxies,
                    force_update=force_update,
                    update_path=update_path,
                )
            ]
        ]
        return dests

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        dests = self.data_path(subject)
        #montage = make_standard_montage("standard_1005")
        #montage.rename_channels(
        #    {ch_name: ch_name.upper() for ch_name in montage.ch_names}
        #)
        # montage.ch_names = [ch_name.upper() for ch_name in montage.ch_names]

        sess = dict()
        for isess, run_dests in enumerate(dests):
            runs = dict()
            for irun, run_array in enumerate(run_dests):
                raw = read_raw_eeglab(run_array, preload=True)
                raw = upper_ch_names(raw)
                #raw.set_montage(montage)
                runs["run_{:d}".format(irun)] = raw
            sess["session_{:d}".format(isess)] = runs

        return sess
