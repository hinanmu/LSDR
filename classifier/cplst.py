#@Time      :2018/12/3 22:05
#@Author    :zhounan
# @FileName: cplst.py

import numpy as np
from numpy import linalg as la
from sklearn.linear_model import Ridge
from sklearn.preprocessing import minmax_scale

class CPLST():
    def __init__(self, m, alpha=0.1):
        '''init

         Parameters
        ----------
        m:      label space compressed dimension, less than label number
        alpha:  linear regression regular coefficient
        '''
        self.m = m
        self.alpha = alpha

    def fit(self, X, y):
        '''fit model

         Parameters
        ----------
        X:  numpy.ndarray
            train input feature
        y:  numpy.ndarray {0,1}
            train output
        '''
        y[y == 0] = -1
        z, self.Vm, self.shift = self.encode(X, y)
        X_bias = np.c_(np.ones(X.shape[0]), X)

        #通过正规方程求解系数
        self.w = (np.linalg.inv(X_bias.T * X_bias + self.alpha * np.eye(X_bias.shape[1])) * X_bias.T) * y


    def encode(self, X, y):
        '''encode y use svd

         Parameters
        ----------
        y:  numpy.ndarray {0,1}
            train output of shape :code:`(n_samples, n_target)`

        Returns
        -------
        z:      numpy.ndarray
                dimensionality reduction matrix of y shape :code:`(n_samples, m)`
        Vm:     numpy.ndarray
                top mright singular matrix after svd shape :code:`(n_features, m)`
        shift:  numpy.ndarray
                mean of y by col shape :code:`(1, n_features)`
        '''
        shift = np.mean(y, axis=0)
        y_shift = y - shift

        H = X * (np.linalg.inv(X.T * X + self.alpha * np.eye(X.shape[1])) * X.T)
        _, _, V = la.svd(y_shift.T * H * y_shift)
        Vm = V[:, 1:self.m]
        z = y_shift * Vm
        return z, Vm, shift

    def predict(self, X):
        '''encode y use svd

        Parameters
       ----------
       X:   numpy.ndarray
            train input feature :code:`(n_samples, n_features)`

       Returns
       -------
       y_pred:      numpy.ndarray {0, 1}
                    predict of y shape :code:`(n_samples, n_traget)`
       y_pred_prob: numpy.ndarray [0, 1]
                    predict probility of y  shape :code:`(n_features, n_traget)`
        '''
        z_pred = self.w * np.c_(np.ones(X.shape(0)), X)
        y_real = z_pred * self.Vm.T + self.shift
        y_pred = np.zeros(y_real.shape)
        y_pred[y_real > 0] = 1
        y_pred[y_real <= 0] = 0
        y_pred_prob = minmax_scale(y_real, axis=1)
        return y_pred, y_pred_prob
