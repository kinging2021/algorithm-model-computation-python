# -*- coding: utf-8 -*-
"""Compare all detection algorithms by plotting decision boundaries and
the number of decision boundaries.
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

# supress warnings for clean output
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all models
from pyod.models.abod import ABOD
from pyod.models.lof import LOF

from scipy import stats

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define the number of inliers and outliers
outliers_fraction = 0.005


# initialize a set of detectors for LSCP
detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                 LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                 LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                 LOF(n_neighbors=50)]

random_state = np.random.RandomState(42)
# Define nine outlier detection tools to be compared
classifiers = {
    'ABOD': ABOD(contamination=outliers_fraction),
    # 'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction, check_estimator=False,
    #                                                     random_state=random_state),
    # 'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction, random_state=random_state),
    # 'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
    # 'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state),
    # 'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
    # 'Average KNN': KNN(method='mean', contamination=outliers_fraction),
    # 'Median KNN': KNN(method='median', contamination=outliers_fraction),
    # 'Local Outlier Factor (LOF)': LOF(n_neighbors=35, contamination=outliers_fraction),
    # 'Local Correlation Integral (LOCI)': LOCI(contamination=outliers_fraction),
    # 'Minimum Covariance Determinant (MCD)': MCD(contamination=outliers_fraction, random_state=random_state),
    # 'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    # 'Principal Component Analysis (PCA)': PCA(contamination=outliers_fraction, random_state=random_state),
    # 'Stochastic Outlier Selection (SOS)': SOS(contamination=outliers_fraction),
    # 'Locally Selective Combination (LSCP)': LSCP(detector_list, contamination=outliers_fraction,
    #                                              random_state=random_state),
    # 'Connectivity-Based Outlier Factor (COF)': COF(n_neighbors=35, contamination=outliers_fraction),
    # 'Subspace Outlier Detection (SOD)': SOD(contamination=outliers_fraction),
}


file = 'GSB01-2019.7.xlsx'
filename = '/home/junze/jupyter/data/能效分析/' + file
dh = pd.read_excel(filename, sheet_name='一级指标-1号锅炉单耗')
fhl = pd.read_excel(filename, sheet_name='一级指标-#1锅炉负荷率')
dh.rename(columns={'时间': 'time', '采集值': 'dh_cjz', '平均值': 'dh_mean', '最大值': 'dh_max', '最小值': 'dh_min'}, inplace=True)
fhl.rename(columns={'时间': 'time', '采集值': 'fhl_cjz', '平均值': 'fhl_mean', '最大值': 'fhl_max', '最小值': 'fhl_min'},
           inplace=True)
data = pd.merge(dh, fhl, on='time')

dh_list = data['dh_mean'].to_list()
fhl_list = data['fhl_mean'].to_list()

param = {
    'y': dh_list,
    'x': fhl_list,
    'x_step': 2,
    'x_range': [0, 100],
    'y_range': [50, 300],
    'bounds': [False, False, False, False],
    'degree': 6,
    'out_size': 1000,
}

from algorithm.base.fitting.ee_curve_gsb import call

c = call(param=param)
X = c.data
xx, yy = np.meshgrid(np.linspace(0, 100, 100), np.linspace(50, 300, 100))


# compare model performances
for i, (clf_name, clf) in enumerate(classifiers.items()):
    print(i + 1, 'fitting', clf_name)

    # Fit the model
    clf.fit(X)

    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)

    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    plt.figure()
    
    # fill blue colormap from minimum anomaly score to threshold value
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)

    # draw red contour line where anomaly score is equal to threshold
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

    # scatter plot of inliers with white dots
    data_false = X[y_pred == 0, :]
    plt.scatter(data_false[:, 0], data_false[:, 1], c='white', s=20, edgecolor='k')

    # scatter plot of outliers with black dots
    data_true = X[y_pred == 1, :]
    plt.scatter(data_true[:, 0], data_true[:, 1], c='black', s=20, edgecolor='k')

    plt.title(clf_name + ', ' + str(data_true.shape[0]) + ', ' + str(data_true.shape[0] / X.shape[0])[:6] + ', ' + file)
    plt.show()
