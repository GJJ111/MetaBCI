import numpy as np
import mne
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

from brainda.datasets import CRED

from brainda.paradigms import Emotion
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_kfold_indices, match_kfold_indices)
from brainda.algorithms.dca import DCA

from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from brainda.algorithms.feature_analysis import TimeAnalysis
from brainda.algorithms.feature_analysis import SpaceAnalysis
from brainda.algorithms.feature_analysis import FrequencyAnalysis
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import joblib


if __name__ == '__main__':

    # dataset = AlexMI()
    # paradigm = MotorImagery(events=['right_hand', 'feet'])
    dataset = CRED()
    paradigm = Emotion(events=['angry', 'disgust','fear', 'neutral', 'sad'])
    

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
        raw.filter(3, 50, 
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
    svm = SVC()
    pca = PCA(n_components=0.99)
    # label=[]
    # Psd=np.zeros((0, 320))
    # Plv=np.zeros((0, 20480))
    # Mic=np.zeros((0, 12000))
    for n in range(1,2):##被试个数
        if n==16:
            pass
        else:
            
            X, y, meta = paradigm.get_data(
                dataset, 
                subjects=[n], 
                return_concat=True, 
                n_jobs=None, 
                verbose=False)
            
            #print(X.shape)
            dca=DCA(meta)
            dca.fit(X,y)
            # mic,label=dca.transform_mic()
            # for Event in np.unique(meta.event):
    joblib.dump(dca,'dca.pkl')
                
    model_dca=joblib.load('dca.pkl')     
    X, y, meta = paradigm.get_data(
                dataset, 
                subjects=[8], 
                return_concat=True, 
                n_jobs=None, 
                verbose=False)
    #dca=DCA(X,y,meta,dataset)
    #mic,label=dca.transform_mic()
    score=model_dca.predict(X)
    report = classification_report(y, score)
    print(report)
          
    #             # 提取PSD特征
                    
    #             sample1 = FrequencyAnalysis(X, meta, event=Event, srate=200)
    #             freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
    #             psd = np.zeros(( 0,5, 64))
    #             for i in range(sample1.data.shape[0]):
    #                 psd_band=np.zeros(( 5,64 ))
    #                 for k, (f_min, f_max) in enumerate(freq_bands):
    #                     f, den=sample1.power_spectrum_density(sample1.data[i])
    #                     idx = np.where((f >= f_min) & (f <= f_max))[0]
    #                     psd_band[k,:]= np.mean(den[:,idx],axis=1)
    #                 psd=np.append(psd,np.expand_dims(psd_band, axis=0),axis=0)

    #             print(psd.shape)
                
    #             #微状态
    #             Feature_R = TimeAnalysis(X, meta, dataset, event = Event, latency = 0)
    #             microstates, microstate_maps=Feature_R.microstates(Feature_R.data)
    #             print(microstates.shape)
                
    #             #plv
    #             Feature_S = SpaceAnalysis(X, meta, dataset, event = Event, srate=128,latency = 0)
    #             freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
    #             plv=Feature_S.compute_plv(freq_bands)
    #             print(plv.shape)
                


                
                
    #             # 将微状态特征展平为 (12000, 10)
    #             microstate_2d = microstates
                
    #             # 将PSD特征保持原样,形状为 (12000, 64)
    #             psd_2d = psd.reshape(10,-1)
                
    #             # 将PLV特征展平为 (12000, 3264)
    #             plv_2d = plv.reshape(10, -1)
                
                
    #             index = np.where(emotion_labels == Event)[0][0]
    #             num = emotion_to_num[index]
    #             for i in range(psd_2d.shape[0]):
    #                 label.append(num)
                
                
    #             Psd = np.append(Psd ,psd_2d,axis=0)
    #             Plv = np.append(Plv,plv_2d,axis=0)
    #             Mic = np.append(Mic,microstate_2d,axis=0)
            
        
    # # 使用选择的主成分数量进行 PCA 变换
    # psd_transformed = pca.fit_transform(Psd)
    # plv_transformed = pca.fit_transform(Plv)
    # mic_transformed = pca.fit_transform(Mic)
    # print(psd_transformed.shape)
    # print(plv_transformed.shape)
    # print(mic_transformed.shape)
    # #label=[]
    # label=np.array(label)
    # dca=DCA()
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # # 进行 5 折交叉验证
    # Score=[]
    # for train_index, test_index in kf.split(psd_transformed):
    #     # 获取训练集和测试集
    #     psd_train, psd_test = psd_transformed[train_index],psd_transformed[test_index]
    #     plv_train, plv_test = plv_transformed [train_index],plv_transformed [test_index]
    #     mic_train, mic_test = mic_transformed[train_index],mic_transformed[test_index]
    #     y_train, y_test = label[train_index], label[test_index]
    #     feature_X,Ax,Ay = dca.dcaFuse(psd_train.T, plv_train.T, y_train.T)
        
    #     psd_test=(np.dot(Ax,psd_test.T)).T
    #     plv_test=(np.dot(Ay,plv_test.T)).T
    #     feature=np.concatenate((psd_test, plv_test), axis=1)
        
    #     feature_Y,Bx,By = dca.dcaFuse(feature_X, mic_train.T, y_train.T)
    #     feature_test=(np.dot(Bx,feature.T)).T
    #     mic_test=(np.dot(By,mic_test.T)).T

    #     Tfeature=np.concatenate((feature_test, mic_test), axis=1)

    #     # 在这里使用你的机器学习模型进行训练和评估
    #     # 例如使用 sklearn 的 LogisticRegression 模型:
    #     dca.trainSVM(feature_Y.T, y_train)

    #     score = dca.predictSVM(Tfeature, y_test)
    #     report = classification_report(y_test, score)
    #     Score.append(score==y_test)
    