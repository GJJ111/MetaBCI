import numpy as np
import mne
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt

class SpaceAnalysis:
    def __init__(self, data, meta, event, srate, latency=0, channel='all'):
        """
        -author: Gao Jianming        
        -Create on:2024-7-30
        -update log:


         Args:
             1.data:EEG data (nTrials, nChannels, nTimes)
                 A matrix fullfilled with timepoint voltage
             2.meta:DataFrame
                 Concrete message of data,including subject ID,the events correspond to  specific trials,etc.
             3.event:String
                 Events needed to be extracted
             4.srate:Int
                 The sample rate of data
             5.latency:Float
                 The start timepoint of experiment (latency=0 indicate that the data recording begin with the stimuli performs)
                 default value=0
             6.channel:String
                 The wanted channel.if 'all',all channel will be extracted.default value = 'all'
        """
        
        if event == 'all_event':
            
            self.data_length = np.round(data.shape[2] / srate)
            if channel == "all":
                self.data = data
            else:
                self.data = data[:, channel, :]
            self.fs = srate
            
        else:
            
            sub_meta = meta[meta["event"] == event]
            event_id = sub_meta.index.to_numpy()
            self.data_length = np.round(data.shape[2] / srate)

            if channel == "all":
                self.data = data[event_id, :, :]
            else:
                self.data = data[event_id, channel, :]
            self.fs = srate
            

    
    def compute_plv(self, freq_bands):
        """
        计算脑电信号的相位锁定值(PLV)特征

        参数:
        freq_bands (list of tuples): 频率带的下限和上限,例如[(4, 8), (8, 13)]

        返回:
        plv_matrix (ndarray): 形状为 (n_trials, n_channels, n_channels, n_freq_bands) 的PLV矩阵
        """
        
        n_channels = self.data.shape[1]
        n_data = self.data.shape[2]
        n_trials = self.data.shape[0]
        n_freq_bands = len(freq_bands)
        plv_matrix = np.zeros((n_trials, n_channels, n_channels, n_freq_bands))

        for k, (f_min, f_max) in enumerate(freq_bands):
            # 滤波
            filtered_data = self.bandpass_filter(self.data, f_min, f_max, self.fs)

            # 计算瞬时相位
            hilbert_data = np.apply_along_axis(hilbert, axis=2, arr=filtered_data)
            phase_data = np.angle(hilbert_data)

            # 计算相位锁定值
            for i in range(n_channels):
                for j in range(i+1, n_channels):
                    phase_diff = phase_data[:, i, :] - phase_data[:, j, :]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
                    plv_matrix[:, i, j, k] = plv
                    plv_matrix[:, j, i, k] = plv

        return plv_matrix

    def bandpass_filter(self, data, f_min, f_max, sfreq, order=5):
        """
        对信号进行带通滤波
        """
        nyquist = sfreq / 2
        low = f_min / nyquist
        high = f_max / nyquist
        b, a = butter(order, [low, high], btype='band', analog=False)
        filtered_data = filtfilt(b, a, data, axis=2)
        return filtered_data
        
    
    # def compute_plv(self, freq_bands):
    #     """
    #     计算脑电信号的相位锁定值(PLV)特征

    #     参数:
    #     freq_bands (list of tuples): 频率带的下限和上限,例如[(4, 8), (8, 13)]

    #     返回:
    #     plv_matrix (ndarray): 形状为 (n_channels, n_channels, n_freq_bands) 的PLV矩阵
    #     """
    #     n_channels = self.data.shape[1]
    #     n_data=self.data.shape[2]
    #     n_trail=self.data.shape[0]
    #     n_freq_bands = len(freq_bands)
    #     plv_matrix = np.zeros((n_trail,n_channels, n_channels, n_freq_bands))

    #     for i in range(n_channels):
    #         for j in range(i+1, n_channels):
    #             for k, (f_min, f_max) in enumerate(freq_bands):
    #                 # 滤波
    #                 filtered_i = self.bandpass_filter(self.data[:, i, :], f_min, f_max, self.fs)
    #                 filtered_j = self.bandpass_filter(self.data[:, j, :], f_min, f_max, self.fs)

    #                 # 计算瞬时相位
    #                 hilbert_i = hilbert(filtered_i)
    #                 hilbert_j = hilbert(filtered_j)
    #                 phase_i = np.angle(hilbert_i)
    #                 phase_j = np.angle(hilbert_j)

    #                  # 计算相位锁定值
    #                 plv = np.abs(np.mean(np.exp(1j * (phase_i-phase_j)),axis=1))
    #                 plv_matrix[:,i, j, k] = plv
    #                 plv_matrix[:,j, i, k] = plv

    #     return plv_matrix

    # def bandpass_filter(self, data, f_min, f_max, sfreq, order=5):
    #     """
    #     对信号进行带通滤波
    #     """
    #     nyquist =  2/128
    #     low = f_min * nyquist
    #     high = f_max * nyquist
    #     b, a = signal.butter(order,[low,high], btype='band')
    #     k=data
    #     c=signal.filtfilt(b, a, data)
    #     return c
