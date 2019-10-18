import numpy as np

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

from algorithm.exception import ParamError


def get_classifier(outliers_fraction, method, random):
    method_list = ['Angle-based Outlier Detector (ABOD)',
                   'Cluster-based Local Outlier Factor (CBLOF)',
                   'Feature Bagging',
                   'Histogram-base Outlier Detection (HBOS)',
                   'Isolation Forest',
                   'K Nearest Neighbors (KNN)',
                   'Average KNN',
                   'Local Outlier Factor (LOF)',
                   'Minimum Covariance Determinant (MCD)',
                   'One-class SVM (OCSVM)',
                   'Principal Component Analysis (PCA)',
                   'Locally Selective Combination (LSCP)']
    if method not in method_list:
        raise ParamError("unknown method name %s" % method)
    random_state = np.random.RandomState(random)
    classifiers = {
        'Angle-based Outlier Detector (ABOD)': lambda:
        ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)': lambda:
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=random_state),
        'Feature Bagging': lambda:
        FeatureBagging(LOF(n_neighbors=35),
                       contamination=outliers_fraction,
                       random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': lambda: HBOS(
            contamination=outliers_fraction),
        'Isolation Forest': lambda: IForest(contamination=outliers_fraction,
                                            random_state=random_state),
        'K Nearest Neighbors (KNN)': lambda: KNN(
            contamination=outliers_fraction),
        'Average KNN': lambda: KNN(method='mean',
                                   contamination=outliers_fraction),
        'Local Outlier Factor (LOF)': lambda:
        LOF(n_neighbors=35, contamination=outliers_fraction),
        'Minimum Covariance Determinant (MCD)': lambda: MCD(
            contamination=outliers_fraction, random_state=random_state),
        'One-class SVM (OCSVM)': lambda: OCSVM(contamination=outliers_fraction),
        'Principal Component Analysis (PCA)': lambda: PCA(
            contamination=outliers_fraction, random_state=random_state),
        'Locally Selective Combination (LSCP)': lambda: LSCP(
            (lambda: [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                      LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                      LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                      LOF(n_neighbors=50)])(), contamination=outliers_fraction,
            random_state=random_state)
    }
    return classifiers[method]()


def get_outlier_labels(data,
                      outliers_fraction=0.01,
                      method="Principal Component Analysis (PCA)",
                      random=42):
    data = np.array(data)
    classifier = get_classifier(outliers_fraction, method, random)
    classifier.fit(data)
    label = classifier.predict(data)
    return label
