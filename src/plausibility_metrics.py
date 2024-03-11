# from .sample import SimpleConstraints
from sklearn.svm import SVC
from light_famd.famd import FAMD as LIGHT_FAMD
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.svm import OneClassSVM
from statsmodels.nonparametric.kernel_density import KDEMultivariate
# from isolation_forest.learn.isolation_forest import IsolationForest 
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest

# from .simple_substructure import ConnectionBraces, SubstructureSimple
from pathlib import Path
import json 
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KernelDensity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler



class ClassIsolationForest():
    def __init__(self) -> None:
        # c = SimpleConstraints()
        # self.brace_dict = ConnectionBraces.ordered_dict()
        # self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        # with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
        with open("real_structures.json") as f:
            real = json.load(f)
        self.brace_dict = {    
            "NONE": 0,
            "H": 1,
            "Z": 2,
            "IZ": 3,
            "ZH": 4,
            "IZH": 5,
            "K": 6,
            "X": 7,
            "XH": 8,
            "nan": 9,
        }
        self.N_BRACES = len(self.brace_dict)
        self.max_layers = max([d["n_layers"] for d in real])

        encodings_real = [self.encode(d) for d in real]
        X = np.array(encodings_real)

        df = pd.DataFrame(X, columns=self.transformed_columns)

        self.instance_df = pd.DataFrame(columns=self.transformed_columns)
        continuous = ["total_height", "radius_bottom", "radius_top", "legs", "n_layers"]
        # continuous_idx = [df.columns.get_loc(x) for x in continuous]
        self.ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous)],
            remainder='passthrough',
        )
        data_trans = self.ct.fit_transform(df)
        self.transformed_training_data = data_trans.copy()

        print("SKLearn Isolation Forest")
        # self.clf = IsolationForest(n_estimators=100, max_samples=256, n_jobs=11)
        self.clf = SKLearnIsolationForest(n_estimators=100, max_samples=100, n_jobs=11)
        # self.clf = IsolationForest(n_estimators=100, max_samples=100, n_jobs=11)
        self.clf.fit(data_trans)
        # scores_1 = np.array(self.clf.score_samples(df))
        # sns.displot(scores_1)


    def get_real_plaus(self):
        neg_anomaly_score = -1*self.clf.score_samples(self.transformed_training_data)
        return neg_anomaly_score

    def score(self, samples):
        encodings = [self.encode(d) for d in samples]
        x = np.array(encodings)
        df_ = pd.DataFrame(x, columns=self.transformed_columns)
        data_trans_ = self.ct.transform(df_)
        neg_anomaly_score = -1*self.clf.score_samples(data_trans_)
        return neg_anomaly_score
    

    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
    def encode(self, d, one_hot=True, native=False):
        basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]

        # fill design's braces according to max_layers with dummies ("NONE")
        braces = d["connection_types"]
        if native:
            braces = np.array([b for b in braces] + ["NONE"] * (self.max_layers - 1 - len(braces)))
        else:
            braces = np.array([self.brace_dict[b] for b in braces] + [self.brace_dict["NONE"]] * (self.max_layers - 1 - len(braces)))
            if one_hot:
                braces = self.get_one_hot(braces, self.N_BRACES)

        # fill design's layer_heights according to max_layers with dummies
        layer_heights = d["layer_heights"]
        layer_heights = np.array(layer_heights + [d["total_height"]] * (self.max_layers - 2 - len(layer_heights))) / d["total_height"]
        self.transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"] + ["brace" + str(i) for i in range(len(braces.flatten()))] + ["layer_height" + str(i) for i in range(len(layer_heights))]

        to_return = np.array([*basics, *braces.flatten(), *layer_heights])
        # return a flat encoding

        return to_return

class ClassKDE():
    def __init__(self) -> None:
        # c = SimpleConstraints()
        # self.brace_dict = ConnectionBraces.ordered_dict()
        # self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        # with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
        with open("real_structures.json") as f:
            real = json.load(f)
        self.brace_dict = {    
            "NONE": 0,
            "H": 1,
            "Z": 2,
            "IZ": 3,
            "ZH": 4,
            "IZH": 5,
            "K": 6,
            "X": 7,
            "XH": 8,
            "nan": 9,
        }
        self.N_BRACES = len(self.brace_dict)
        self.max_layers = max([d["n_layers"] for d in real])

        encodings_real = [self.encode(d) for d in real]
        X = np.array(encodings_real)

        df = pd.DataFrame(X, columns=self.transformed_columns)
        self.instance_df = pd.DataFrame(columns=self.transformed_columns)
        continuous = ["total_height", "radius_bottom", "radius_top", "legs", "n_layers"]
        # continuous_idx = [df.columns.get_loc(x) for x in continuous]
        self.ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous)],
            remainder='passthrough',
        )
        data_trans = self.ct.fit_transform(df)
        self.transformed_training_data = data_trans.copy()

        print("KDE")
        self.clf = KernelDensity()
        self.clf.fit(data_trans)
        # print(self.clf.__repr__())
        # pdf_values = self.clf.pdf(df)
        # log_likelihoods = np.log(pdf_values)
        # sns.displot(log_likelihoods)
        # self.clf.fit(data_trans)
    def get_real_plaus(self):
        neg_anomaly_score = -1*self.clf.score_samples(self.transformed_training_data)
        return neg_anomaly_score

    def score(self, samples):
        encodings = [self.encode(d) for d in samples]
        x = np.array(encodings)
        df_ = pd.DataFrame(x, columns=self.transformed_columns)
        data_trans_ = self.ct.transform(df_)
        neg_anomaly_score = -1*self.clf.score_samples(data_trans_)
        return neg_anomaly_score
    def transform(self, dct):
        self.encode(dct)

    def get_str_list(self):
        # Create a list where each character corresponds to a variable type
        result_list = []

        for column in self.transformed_columns:
            if column in self.continuous:
                result_list.append("c")
            elif column in self.ordered:
                result_list.append("o")
            elif column in self.unordered:
                result_list.append("u")
            else:
                # Handle the case where the column doesn't match any type
                result_list.append("?")

        result_string = "".join(result_list)
        return result_string

    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])
    def encode(self, d, one_hot=True, native=False):
        basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]

        # fill design's braces according to max_layers with dummies ("NONE")
        braces = d["connection_types"]
        if native:
            braces = np.array([b for b in braces] + ["NONE"] * (self.max_layers - 1 - len(braces)))
        else:
            braces = np.array([self.brace_dict[b] for b in braces] + [self.brace_dict["NONE"]] * (self.max_layers - 1 - len(braces)))
            if one_hot:
                braces = self.get_one_hot(braces, self.N_BRACES)

        # fill design's layer_heights according to max_layers with dummies
        layer_heights = d["layer_heights"]
        layer_heights = np.array(layer_heights + [d["total_height"]] * (self.max_layers - 2 - len(layer_heights))) / d["total_height"]
        self.transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"] + ["brace" + str(i) for i in range(len(braces.flatten()))] + ["layer_height" + str(i) for i in range(len(layer_heights))]

        to_return = np.array([*basics, *braces.flatten(), *layer_heights])
        # return a flat encoding

        return to_return
   
class DiscriminatorOneClass():
    def __init__(self, gamma="scale", nu=0.5) -> None:
        self.gamma = gamma
        self.nu = nu
        # c = SimpleConstraints()
        # self.brace_dict = ConnectionBraces.ordered_dict()
        self.brace_dict = {    
            "NONE": 0,
            "H": 1,
            "Z": 2,
            "IZ": 3,
            "ZH": 4,
            "IZH": 5,
            "K": 6,
            "X": 7,
            "XH": 8,
            "nan": 9,
        }
        self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
            real = json.load(f)

        self.max_layers = max([d["n_layers"] for d in real])

        encodings_real = [self.encode(d) for d in real]
        X = np.array(encodings_real)

        df = pd.DataFrame(X, columns=self.transformed_columns)
        self.instance_df = pd.DataFrame(columns=self.transformed_columns)
        continuous = ["total_height", "radius_bottom", "radius_top", "legs", "n_layers"]
        # continuous_idx = [df.columns.get_loc(x) for x in continuous]
        self.ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous)],
            remainder='passthrough',
        )
        data_trans = self.ct.fit_transform(df)
        self.transformed_training_data = data_trans.copy()
        # self.clf = SVC(kernel="linear", C=1, gamma=0.001)
        print("One Class SVM gamma={} nu={}".format(self.gamma, self.nu))
        self.clf = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu)
        self.clf.fit(data_trans)

    def get_real_plaus(self):
        neg_anomaly_score = -1*self.clf.score_samples(self.transformed_training_data)
        return neg_anomaly_score

    def score(self, samples):
        encodings = [self.encode(d) for d in samples]
        x = np.array(encodings)
        df_ = pd.DataFrame(x, columns=self.transformed_columns)
        data_trans_ = self.ct.transform(df_)
        neg_anomaly_score = -1*self.clf.score_samples(data_trans_)
        return neg_anomaly_score
    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])


    def encode(self, d, one_hot=True, native=False):
        basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]

        # fill design's braces according to max_layers with dummies ("NONE")
        braces = d["connection_types"]
        if native:
            braces = np.array([b for b in braces] + ["NONE"] * (self.max_layers - 1 - len(braces)))
        else:
            braces = np.array([self.brace_dict[b] for b in braces] + [self.brace_dict["NONE"]] * (self.max_layers - 1 - len(braces)))
            if one_hot:
                braces = self.get_one_hot(braces, self.N_BRACES)

        # fill design's layer_heights according to max_layers with dummies
        layer_heights = d["layer_heights"]
        layer_heights = np.array(layer_heights + [d["total_height"]] * (self.max_layers - 2 - len(layer_heights))) / d["total_height"]
        self.transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"] + ["brace" + str(i) for i in range(len(braces.flatten()))] + ["layer_height" + str(i) for i in range(len(layer_heights))]

        to_return = np.array([*basics, *braces.flatten(), *layer_heights])
        # return a flat encoding

        return to_return