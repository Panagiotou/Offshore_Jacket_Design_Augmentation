from .sample import SimpleConstraints
from sklearn.svm import SVC
from light_famd.famd import FAMD as LIGHT_FAMD
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.svm import OneClassSVM
from statsmodels.nonparametric.kernel_density import KDEMultivariate
# from isolation_forest.learn.isolation_forest import IsolationForest 
from sklearn.ensemble import IsolationForest as SKLearnIsolationForest

from .simple_substructure import ConnectionBraces, SubstructureSimple
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

class ClassLocalOutlierFactor():
    def __init__(self) -> None:
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
        self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        # with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
        with open("real_structures.json") as f:
            real = json.load(f)

        self.max_layers = max([d["n_layers"] for d in real])

        encodings_real = [self.encode(d) for d in real]
        X = np.array(encodings_real)

        df= pd.DataFrame(X, columns=self.transformed_columns)
        self.instance_df = pd.DataFrame(columns=self.transformed_columns)
        continuous = ["total_height", "radius_bottom", "radius_top", "legs", "n_layers"]
        # continuous_idx = [df.columns.get_loc(x) for x in continuous]
        self.ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous)],
            remainder='passthrough',
        )
        data_trans = self.ct.fit_transform(df)
        self.transformed_training_data = data_trans.copy()

        print("Local Outlier Factor")
        # self.clf = IsolationForest(n_estimators=100, max_samples=256, n_jobs=11)
        self.clf = LocalOutlierFactor(novelty=True)
        self.clf.fit(data_trans)

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

class ClassIsolationForest():
    def __init__(self) -> None:
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
        self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        # with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
        with open("real_structures.json") as f:
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

        print("SKLearn Isolation Forest")
        # self.clf = IsolationForest(n_estimators=100, max_samples=256, n_jobs=11)
        self.clf = SKLearnIsolationForest(n_estimators=100, max_samples=100, n_jobs=11)
        # self.clf = IsolationForest(n_estimators=100, max_samples=100, n_jobs=11)
        self.clf.fit(data_trans)
        # scores_1 = np.array(self.clf.score_samples(df))
        # sns.displot(scores_1)

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
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
        self.N_BRACES = len(self.brace_dict)
        # the real structures are stored outside of the graphstructure package
        # with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
        with open("real_structures.json") as f:
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

        print("KDE")
        self.clf = KernelDensity()
        self.clf.fit(data_trans)
        # print(self.clf.__repr__())
        # pdf_values = self.clf.pdf(df)
        # log_likelihoods = np.log(pdf_values)
        # sns.displot(log_likelihoods)
        # self.clf.fit(data_trans)

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
class Discriminator():
    def __init__(self, n_samples=10000) -> None:
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
        self.N_BRACES = len(self.brace_dict)

        samples = c.sample(n=n_samples)
        designs = [s.to_dict() for s in samples]
        self.max_layers = max([d["n_layers"] for d in designs])

        # the real structures are stored outside of the graphstructure package
        with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
            real = json.load(f)

        encodings_sampled = [self.encode(d) for d in designs]
        encodings_real = [self.encode(d) for d in real]

        data = np.concatenate((encodings_sampled, encodings_real))
        labels = np.concatenate(([0]*len(encodings_sampled), [1]*len(encodings_real)))

        X, y = shuffle(data, labels, random_state=42)

        df = pd.DataFrame(X, columns=self.transformed_columns)
        self.instance_df = pd.DataFrame(columns=self.transformed_columns)
        continuous = ["total_height", "radius_bottom", "radius_top", "legs", "n_layers"]
        # continuous_idx = [df.columns.get_loc(x) for x in continuous]
        self.ct = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous)],
            remainder='passthrough',
        )
        data_trans = self.ct.fit_transform(df)

        # self.clf = SVC(kernel="linear", C=1, gamma=0.001)
        self.clf = SVC(kernel="rbf", C=1, gamma=0.001, class_weight={1:100})
        self.clf.fit(data_trans, y)
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
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
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
    
class DiscriminatorOneClassFAMD():
    def __init__(self, gamma="scale", nu=0.5) -> None:
        self.famd = Famd()
        data_trans = self.famd.latent_real
        self.gamma = gamma
        self.nu = nu
        self.transformed_training_data = data_trans.copy()
        # self.clf = SVC(kernel="linear", C=1, gamma=0.001)
        self.clf = OneClassSVM(kernel="rbf", gamma=self.gamma, nu=self.nu)
        self.clf.fit(data_trans)
    def encode(self, d):
        return self.famd.encode(d)

class Famd():
    def __init__(self) -> None:
        c = SimpleConstraints()
        self.brace_dict = ConnectionBraces.ordered_dict()
        self.N_BRACES = len(self.brace_dict)
        print(self.brace_dict)
        self.max_layers = max(c.n_layers)

        # the real structures are stored outside of the graphstructure package
        with open(Path(__file__).parent.parent / "util/real_structures.json") as f:
            real = json.load(f)

        encodings_real = [self.encode(d, one_hot=False, native=True) for d in real]

        self.transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"]
        self.brace_cols = ["brace" + str(i) for i in range(self.max_layers-1)] 
        self.transformed_columns += self.brace_cols
        self.transformed_columns += ["layer_height" + str(i) for i in range(self.max_layers-2)]


        df = pd.DataFrame(encodings_real, columns=self.transformed_columns)

        self.nominal_features = self.brace_cols
        self.ordinal_features = ["n_layers", "legs"]
        self.BERNOULLI = ["legs"]
        self.discrete_features = self.nominal_features + self.ordinal_features
        self.continuous_features = list(set(self.transformed_columns) - set(self.nominal_features) - set(self.ordinal_features))

        df[self.ordinal_features] = df[self.ordinal_features].astype("int")
        df[self.continuous_features] = df[self.continuous_features].astype("float")
        df = df.infer_objects()

        var_transform_only = self.get_dtypes(df)

        p = df.shape[1]

        self.dtypes_dict_famd = {'continuous': float, 'categorical': str, 'ordinal': str,\
              'bernoulli': str, 'binomial': str}
                
        # Feature category (cf)
        self.dtype = {df.columns[j]: self.dtypes_dict_famd[var_transform_only[j]] for j in range(p)}

        df = df.astype(self.dtype, copy=True)

        self.famd = LIGHT_FAMD(n_components = 2, n_iter=3, copy=True, check_input=False, \
            engine='sklearn', random_state = 2023)
        
        self.latent_real = self.famd.fit_transform(df)

        # Calculate mean and covariance matrix of the data
        self.mean_latent_real = np.mean(self.latent_real, axis=0)
        covariance_matrix_latent_real = np.cov(self.latent_real, rowvar=False)
        # Calculate the inverse of the covariance matrix
        self.inv_covariance_matrix_latent_real = inv(covariance_matrix_latent_real)

        self.gmm = GaussianMixture(n_components=3, random_state=2023).fit(self.latent_real)
    
    def transform(self, dct):
        encoded = self.encode(dct, one_hot=False, native=True)
        df = pd.DataFrame([encoded], columns=self.transformed_columns)
        df[self.ordinal_features] = df[self.ordinal_features].astype("int")
        df[self.continuous_features] = df[self.continuous_features].astype("float")
        df = df.infer_objects()
        df = df.astype(self.dtype, copy=True)
        instance_transformed_famd = self.famd.transform(df)
        return instance_transformed_famd
    
    def mahalanobis(self, dct):
        encoded = self.encode(dct, one_hot=False, native=True)
        df = pd.DataFrame([encoded], columns=self.transformed_columns)
        df[self.ordinal_features] = df[self.ordinal_features].astype("int")
        df[self.continuous_features] = df[self.continuous_features].astype("float")
        df = df.infer_objects()
        df = df.astype(self.dtype, copy=True)
        instance_transformed_famd = self.famd.transform(df)[0]

        mahalanobis_distance = mahalanobis(instance_transformed_famd, self.mean_latent_real, self.inv_covariance_matrix_latent_real)
        return mahalanobis_distance
    
    def gmm_log_likelihood(self, dct):
        encoded = self.encode(dct, one_hot=False, native=True)
        df = pd.DataFrame([encoded], columns=self.transformed_columns)
        df[self.ordinal_features] = df[self.ordinal_features].astype("int")
        df[self.continuous_features] = df[self.continuous_features].astype("float")
        df = df.infer_objects()
        df = df.astype(self.dtype, copy=True)
        instance_transformed_famd = self.famd.transform(df)
        log_likelihood = self.gmm.score(instance_transformed_famd)
        #invert because minimization problem (I want to maximize the likelihood)
        return -1*log_likelihood
    
    def get_dtypes(self, train):
        unique_counts = train.nunique()
        var_distrib = []     
        var_transform_only = []     
        # Encode categorical datas
        for colname, dtype, unique in zip(train.columns, train.dtypes, train.nunique()):
            if unique < 2:
                print("Dropped", colname, "because of 0 var")
                train.drop(colname, axis=1, inplace=True)
                continue

            if (dtype==int or dtype == "object") and unique==2:
                #bool
                
                var_distrib.append('bernoulli')
                if colname in self.BERNOULLI:
                    var_transform_only.append('bernoulli')
                else:
                    var_transform_only.append('categorical')
                    
                le = LabelEncoder()
                # Convert them into numerical values               
                train[colname] = le.fit_transform(train[colname]) 
            elif dtype==int and unique > 2:
                # ordinal
                
                var_distrib.append('ordinal')
                var_transform_only.append('ordinal')
                
                le = LabelEncoder()
                # Convert them into numerical values               
                train[colname] = le.fit_transform(train[colname]) 
            elif dtype == "object":
                var_distrib.append('categorical')
                var_transform_only.append('categorical')
                
                le = LabelEncoder()
                # Convert them into numerical values               
                train[colname] = le.fit_transform(train[colname]) 
            elif dtype == float:
                var_distrib.append('continuous')
                var_transform_only.append('continuous')
            
        return var_transform_only


    # from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
    def get_one_hot(self, targets, nb_classes):
        res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[nb_classes])


    def encode(self, d, one_hot=True, native=False):
        basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]
        
        # fill design's braces according to max_layers with dummies ("NONE")
        braces = d["connection_types"]
        if native:
            braces = np.array([b for b in braces] + ["nan"] * (self.max_layers - 1 - len(braces)))
        else:
            braces = np.array([self.brace_dict[b] for b in braces] + [self.brace_dict["nan"]] * (self.max_layers - 1 - len(braces)))
            if one_hot:
                braces = self.get_one_hot(braces, self.N_BRACES)
        
        # fill design's layer_heights according to max_layers with dummies
        layer_heights = d["layer_heights"]
        layer_heights = np.array(layer_heights + [d["total_height"]] * (self.max_layers - 2 - len(layer_heights))) / d["total_height"]
        
        # return a flat encoding
        return np.array([*basics, *braces.flatten(), *layer_heights])
