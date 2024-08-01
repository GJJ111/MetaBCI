import numpy as np
from numpy.linalg import eig, pinv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
import mne
from sklearn.svm import SVC

#from brainda.algorithms.feature_analysis import TimeAnalysis
from brainda.algorithms.feature_analysis import SpaceAnalysis
from brainda.algorithms.feature_analysis import FrequencyAnalysis
import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import joblib


# class DCA(???):
class DCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    """
    
    The Spatial-Temporal Discriminant Analysis (STDA) algorithm maximizes
    the discriminability of the projected features between target and non-target classes
    by alternately and synergistically optimizing the spatial and temporal dimensions of the EEG
    in order to learn two projection matrices. Using the learned two projection matrices to
    transform each of the constructed spatial-temporal two-dimensional samples into new one-dimensional
    samples with significantly lower dimensions effectively improves the covariance matrix parameter estimation
    and enhances the generalization ability of the learned classifiers under small training sample sets.
    
    author: Gao Jianming and Pei Yu

    email: gaojianming@tju.edu.cn, yupei2409@gmail.com

    Created on: 2022-05


    References
    ----------
    [1] XXXX
    
    Tip
    ----
        import numpy as np
        from metabci.brainda.algorithms.dca import DCA
        X_raw = np .... (样本量, 通道, 采样点)
        Y     = np ....
        
        dca = DCA()
        
        dca.fit(X_raw, Y)
        
        X_new =  ??
        
        dca.predict(X_new)
        
        print(clf3.transform(Xtest2))
    """
    
    def __init__(self,meta):
        
        # self.X=X
        # self.y=y
        self.meta=meta
        # self.dataset=dataset
        self.Ax = None
        self.Ay = None
        self.Xs = None
        self.Ys = None
        self.model = SVC()
        self.fs = 200
        self.pca_psd=None
        self.pca_plv=None
        

    
    def transform_psd(self,X):
        # (n_trial, n_channel, n_sample) --> # (n_trial, n_channel x n_band)
        
        n_channel=X.shape[1]
        Psd=np.zeros((0, 5,n_channel))
        
        # for Event in np.unique(self.meta.event):
        #     sample1 = FrequencyAnalysis(self.X, self.meta, event='all_event', srate=200)
        #     freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
        #     for i in range(sample1.data.shape[0]):
        #         psd_band=np.zeros(( 5,64 ))
        #         psd = np.zeros(( 0,5, 64))
        #         for k, (f_min, f_max) in enumerate(freq_bands):
        #             f, den=sample1.power_spectrum_density(sample1.data[i])
        #             idx = np.where((f >= f_min) & (f <= f_max))[0]
        #             psd_band[k,:]= np.mean(den[:,idx],axis=1)
        #         psd=np.append(psd,np.expand_dims(psd_band, axis=0),axis=0)
        #         Psd = np.append(Psd ,psd,axis=0)
        
        sample1 = FrequencyAnalysis(X, self.meta, event='all_event', srate=200)
        freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
        for i in range(sample1.data.shape[0]):
            psd_band=np.zeros(( 5,64 ))
            psd = np.zeros(( 0,5, 64))
            for k, (f_min, f_max) in enumerate(freq_bands):
                f, den=sample1.power_spectrum_density(sample1.data[i])
                idx = np.where((f >= f_min) & (f <= f_max))[0]
                psd_band[k,:]= np.mean(den[:,idx],axis=1)
            psd=np.append(psd,np.expand_dims(psd_band, axis=0),axis=0)
            Psd = np.append(Psd ,psd,axis=0)

        return Psd
    
    def transform_plv(self,X):
        # (n_trial, n_channel, n_sample) --> # (n_trial, n_channel x n_channel x n_band / 2)
        n_channel=X.shape[1]
        Plv=np.zeros((0, n_channel,n_channel,5))
        
        # for Event in np.unique(self.meta.event):
        #     Feature_S = SpaceAnalysis(self.X, self.meta, self.dataset, event = Event, srate=128,latency = 0)
        #     freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
        #     plv=Feature_S.compute_plv(freq_bands)
        #     Plv = np.append(Plv,plv,axis=0)
        
        Feature_S = SpaceAnalysis(X, self.meta, event='all_event', srate=200,latency = 0)
        freq_bands = [(1,4),(4, 8), (8, 13), (13, 30),(30, 50)]
        plv=Feature_S.compute_plv(freq_bands)
        Plv = np.append(Plv,plv,axis=0)
        return Plv
        
    # def transform_mic(self,X):
    #     # 不要了,无法满足实时处理需求
    #     n_samplel=X.shape[2]
    #     Mic=np.zeros((0, n_samplel))
    #     for Event in np.unique(self.meta.event):
    #         Feature_R = TimeAnalysis(X, self.meta, self.dataset, event = Event, latency = 0)
    #         microstates, microstate_maps=Feature_R.microstates(Feature_R.data)
    #         Mic = np.append(Mic,microstates,axis=0)

    #     return Pic
    
    def predict(self, X):
        
        # X : (1, ch, pnt)? (ch, pnt)
        if X.ndim==2:
            X=np.expand_dims(X, axis=0)
        
        psd=self.transform_psd(X)
        plv=self.transform_plv(X)
        #micl=self.transform_mic(X)
        # (n_trial, n_channel, n_sample) --> # (n_trial, 1)

        psd_2d = psd.reshape(psd.shape[0],-1)
        psd_transformed = self.pca_psd.transform(psd_2d)
        # plv
        plv_2d = plv.reshape(plv.shape[0], -1)
        plv_transformed = self.pca_plv.transform(plv_2d)
        
        
        psd_test=(np.dot(self.Ax,psd_transformed.T)).T
        plv_test=(np.dot(self.Ay,plv_transformed.T)).T
        
        
        feature=np.concatenate((psd_test, plv_test), axis=1)
        score  = self.model.predict(feature)
        # score = self.predictSVM(feature)
        # report = classification_report(self.y, score)
        return score
    
    def fit(self, X, y):
        
        # X (40+, ch, pnts)
        
        """
        xxxx

        Inputs:
            X_raw       :   pxn matrix containing the first set of training feature vectors
                        p:  dimensionality of the first feature set
                        n:  number of training samples
            Y       :   qxn matrix containing the second set of training feature vectors
                        q:  dimensionality of the second feature set
            label   :   1xn row vector of length n containing the class labels
                    
        Outputs:
            Ax  :   Transformation matrix for the first data set (rxp)
                    r:  maximum dimensionality in the new subspace
            Ay  :   Transformation matrix for the second data set (rxq)
            Xs  :   First set of transformed feature vectors (rxn)
            Ys  :   Second set of transformed feature vectors (rxn)
        """
        
        self.pca_psd = PCA(n_components=0.99)
        self.pca_plv = PCA(n_components=0.99)
        # psd
        psd=self.transform_psd(X)
        psd_2d = psd.reshape(psd.shape[0],-1)
        self.pca_psd.fit(psd_2d)
        psd_transformed=self.pca_psd.transform(psd_2d)
        #psd_transformed = pca.fit_transform(psd_2d)
        # plv
        plv=self.transform_plv(X)
        plv_2d = plv.reshape(plv.shape[0], -1)
        self.pca_plv.fit(plv_2d)
        plv_transformed=self.pca_plv.transform(plv_2d)
        
        #plv_transformed = pca.fit_transform(plv_2d)
        # # mic
        # mic_transformed = pca.fit_transform(mic)
        feature_X,Ax,Ay = self.dcaFuse(psd_transformed.T, plv_transformed.T, y.T)
        #feature_Y,Bx,By = self.dcaFuse(feature_X, mic_transformed.T, Y.T)
        self.model.fit(feature_X.T, y)
        self.Ax=Ax
        self.Ay=Ay
        return self
        
        
        
        
    
    def dcaFuse(self,X, Y, label):
        """
        DCAFUSE calculates the Discriminant Correlation Analysis (DCA) for 
        feature-level fusion in multimodal systems.

        Inputs:
            X       :   pxn matrix containing the first set of training feature vectors
                        p:  dimensionality of the first feature set
                        n:  number of training samples
            Y       :   qxn matrix containing the second set of training feature vectors
                        q:  dimensionality of the second feature set
            label   :   1xn row vector of length n containing the class labels
                    
        Outputs:
            Ax  :   Transformation matrix for the first data set (rxp)
                    r:  maximum dimensionality in the new subspace
            Ay  :   Transformation matrix for the second data set (rxq)
            Xs  :   First set of transformed feature vectors (rxn)
            Ys  :   Second set of transformed feature vectors (rxn)
        """
        p, n = X.shape
        if Y.shape[1] != n:
            raise ValueError('X and Y must have the same number of columns (samples).')
        elif len(label) != n:
            raise ValueError('The length of the label must be equal to the number of samples.')
        elif n == 1:
            raise ValueError('X and Y must have more than one column (samples)')
        q = Y.shape[0]

        # Normalize features (this has to be done for both train and test data)
        # X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
        # Y = (Y - np.mean(Y, axis=1, keepdims=True)) / np.std(Y, axis=1, keepdims=True)

        # Compute mean vectors for each class and for all training data
        # Compute mean vectors for each class and for all training data
        classes, counts = np.unique(label, return_counts=True)
        c = len(classes)

        cellX = [None] * c
        cellY = [None] * c
        nSample = np.zeros(c, dtype=int)

        for i in range(c):
            index = np.where(label == classes[i])[0]
            nSample[i] = len(index)
            cellX[i] = X[:, index]
            cellY[i] = Y[:, index]

        meanX = np.mean(X, axis=1)  # Mean of all training data in X
        meanY = np.mean(Y, axis=1)  # Mean of all training data in Y

        classMeanX = np.zeros((p, c))
        classMeanY = np.zeros((q, c))

        for i in range(c):
            classMeanX[:, i] = np.mean(cellX[i], axis=1)  # Mean of each class in X
            classMeanY[:, i] = np.mean(cellY[i], axis=1)  # Mean of each class in Y

        PhibX = np.zeros((p, c))
        PhibY = np.zeros((q, c))

        for i in range(c):
            PhibX[:, i] = np.sqrt(nSample[i]) * (classMeanX[:, i] - meanX)
            PhibY[:, i] = np.sqrt(nSample[i]) * (classMeanY[:, i] - meanY)
            
        # Diagonalize the between-class scatter matrix (Sb) for X
        artSbx = np.dot(PhibX.T, PhibX)  # Artificial Sbx (artSbx) is a (c x c) matrix
        eigVals, eigVecs = np.linalg.eig(artSbx)
        eigVals = np.abs(eigVals)

        # Ignore zero eigenvalues
        max_eig_val = np.max(eigVals)
        zero_eig_idx = np.where(eigVals / max_eig_val < 1e-6)[0]
        eigVals = np.delete(eigVals, zero_eig_idx)
        eigVecs = np.delete(eigVecs, zero_eig_idx, axis=1)

        # Sort in descending order
        sort_idx = np.argsort(-eigVals)
        eigVals = eigVals[sort_idx]
        eigVecs = eigVecs[:, sort_idx]

        # Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
        SbxEigVecs = np.dot(PhibX, eigVecs)
        
        # Normalize to unit length to create orthonormal eigenvectors for Sbx
        cx = len(eigVals)
        for i in range(cx):
            SbxEigVecs[:, i] = SbxEigVecs[:, i] / np.linalg.norm(SbxEigVecs[:, i])

        # Unitize the between-class scatter matrix (Sbx) for X
        SbxEigVals = np.diag(eigVals)  # SbxEigVals is a (cx x cx) diagonal matrix
        Wbx = np.dot(SbxEigVecs, np.linalg.inv(np.sqrt(SbxEigVals)))  # Wbx is a (p x cx) matrix which unitizes Sbx

        # Diagolalize the between-class scatter matrix (Sb) for Y
        artSby = np.dot(PhibY.T, PhibY)  # Artificial Sby (artSby) is a (c x c) matrix
        eigVals, eigVecs = np.linalg.eig(artSby)
        eigVals = np.abs(eigVals)
        
        # Ignore zero eigenvalues
        maxEigVal = np.max(eigVals)
        zeroEigIndx = np.where(eigVals/maxEigVal < 1e-6)[0]
        eigVals = np.delete(eigVals, zeroEigIndx)
        eigVecs = np.delete(eigVecs, zeroEigIndx, axis=1)

        # Sort in descending order
        sorted_idx = np.argsort(-eigVals)
        eigVals = eigVals[sorted_idx]
        eigVecs = eigVecs[:, sorted_idx]

        # Calculate the actual eigenvectors for the between-class scatter matrix (Sby)
        SbyEigVecs = np.dot(PhibY, eigVecs)

        # Normalize to unit length to create orthonormal eigenvectors for Sby
        cy = len(eigVals)
        for i in range(cy):
            SbyEigVecs[:, i] = SbyEigVecs[:, i] / np.linalg.norm(SbyEigVecs[:, i])

        # Unitize the between-class scatter matrix (Sby) for Y
        SbyEigVals = np.diag(eigVals)  # SbyEigVals is a (cy x cy) diagonal matrix
        Wby = np.dot(SbyEigVecs, np.linalg.inv(np.sqrt(SbyEigVals)))  # Wby is a (q x cy) matrix which unitizes Sby
        
        # Project data in a space, where the between-class scatter matrices are identity and the classes are separated

        r = min(cx, cy)  # Maximum length of the desired feature vector

        Wbx = Wbx[:, :r]
        Wby = Wby[:, :r]

        Xp = np.dot(Wbx.T, X)  # Transform X (pxn) to Xprime (rxn)
        Yp = np.dot(Wby.T, Y)  # Transform Y (qxn) to Yprime (rxn)

        # Unitize the between-set covariance matrix (Sxy)
        # Note that Syx == Sxy'

        Sxy = np.dot(Xp, Yp.T)  # Between-set covariance matrix

        Wcx, S, Wcy = np.linalg.svd(Sxy)  # Singular Value Decomposition (SVD)

        Wcx = np.dot(Wcx, np.diag(np.power(S, -0.5)))  # Transformation matrix for Xp
        Wcy = np.dot(Wcy, np.diag(np.power(S, -0.5)))  # Transformation matrix for Yp

        Xs = np.dot(Wcx.T, Xp)  # Transform Xprime to XStar
        Ys = np.dot(Wcy.T, Yp)  # Transform Yprime to YStar

        Ax = np.dot(Wcx.T, Wbx.T)  # Final transformation Matrix of size (rxp) for X
        Ay = np.dot(Wcy.T, Wby.T)  # Final transformation Matrix of size (rxq) for Y
        
        # Ax=Ax.reshape(Ax.shape[0],1)
        # Ay=Ay.reshape(Ay.shape[0],1)
        
        # Xs=Xs.reshape(Xs.shape[0],1)
        # Ys=Ys.reshape(Ys.shape[0],1)
        feature=np.concatenate((Xs, Ys), axis=0)
        return feature,Ax,Ay
    
    
    # def trainSVM(self, X, y):
    #     """
    #     Train an SVM classifier using the fused features
    #     """
    #     # scaler = StandardScaler()
    #     # X = scaler.fit_transform(X)
    #     self.model = SVC()
    #     self.model.fit(X, y)

    # def predictSVM(self, X):
    #     """
    #     Predict using the trained SVM classifier
    #     """
    #     y_pred = self.model.predict(X)
    #     return y_pred
    
    
# np.random.seed(42)
# X = np.random.randn(10, 50)  # 第一个数据集,10维，50个样本
# Y = np.random.randn(20, 50)  # 第二个数据集,20维，50个样本
# label = np.random.randint(0, 3, size=50)  # 50个样本的类标签,取值为 0, 1, 2
# Ax, Ay, Xs, Ys = dcaFuse(X, Y, label)
    
    

# print(f"第一个数据集的变换矩阵 Ax 大小: {Ax.shape}")
# print(f"第二个数据集的变换矩阵 Ay 大小: {Ay.shape}")
# print(f"变换后的第一个数据集 Xs 大小: {Xs.shape}")
# print(f"变换后的第二个数据集 Ys 大小: {Ys.shape}")