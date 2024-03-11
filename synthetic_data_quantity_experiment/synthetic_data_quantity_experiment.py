import sys
import json
import pandas as pd
import numpy as np
from src.utils_methods import encode, get_var_metadata, post_process, transform_df_to_json, generate_plots, check_constraints
import random

def random_equal_splits(lst, split_size, num_splits):
    if split_size > len(lst):
        return [random.sample(lst, len(lst)) for _ in range(num_splits)]

    random.shuffle(lst)
    return [lst[i:i+split_size] for i in range(0, len(lst), split_size)][:num_splits]

# # Example usage:
# my_list = [1, 2, 3, 4, 5]
# split_size = 2
# num_splits = 2

# result = random_equal_splits(my_list, split_size, num_splits)
# print(result)

import json
import os
data_folder = "results/evaluated/"

FOLDERS = ["vae", "dgmm", "llm"]
# FOLDERS = ["vae"]

DATA_ARRAY = {}
DATA_REAL = {}

DATA_NAMES = ["real"] + FOLDERS

with open(data_folder + "real/real_structures_evaluated.json") as f:
    real = json.load(f)
    DATA_REAL["real"] = real
    total_max_layers = max(d.get('n_layers', 0) for d in real)

# for folder in FOLDERS:
#     DATA_ARRAY[folder] = {}
#     path_seed = os.path.join(data_folder,folder, "seeds")
#     seeds = [d for d in os.listdir(path_seed) if os.path.isdir(os.path.join(path_seed, d))]
#     for seed in seeds:
#         DATA_ARRAY[folder][seed] = []
#         seed_files = os.listdir(os.path.join(path_seed,seed))
#         for seed_file in seed_files:
#             with open(os.path.join(path_seed,seed,seed_file)) as f:
#                 data = json.load(f)
#                 max_layers = max(d.get('n_layers', 0) for d in data)
#                 if max_layers > total_max_layers: total_max_layers = max_layers
#                 DATA_ARRAY[folder][seed].append(data)

number_of_instances = list(range(100, 40001, 100))
number_of_splits = 3
for folder in FOLDERS:
    DATA_ARRAY[folder] = {}
    huge_file = os.path.join(data_folder,folder, "{}(real)_100000_evaluated.json".format(folder))
    with open(huge_file) as f:
        data = json.load(f)
        max_layers = max(d.get('n_layers', 0) for d in data)
        if max_layers > total_max_layers: total_max_layers = max_layers

        for seed in number_of_instances:
            DATA_ARRAY[folder][seed] = random_equal_splits(data, seed, number_of_splits)
                
print(total_max_layers)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def get_unique_all(X, columns_one_hot):

    all_categories = {}
    for column in columns_one_hot:
        unique_categories = set()
        for df in X:
            unique_categories.update(df[column].unique())
        all_categories[column] = mapping = {val: idx for idx, val in enumerate(list(unique_categories))} 
    return all_categories

def encode_dataset(dataset, y_columns_to_extract, transformed_columns, brace_dict, N_BRACES):
    encoding = [encode(d, max_layers, brace_dict, N_BRACES, one_hot=False, native=True, normalize_layer_heights=True) for d in dataset]

    x = pd.DataFrame(encoding, columns=transformed_columns)
    extracted_data = [{col: entry[col] for col in y_columns_to_extract} for entry in dataset]
    # Create a DataFrame
    y = pd.DataFrame(extracted_data)

    return x, y

# augmented = pd.concatenate(X) 
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_one_hot, columns_standard, unique_categories, columns):
        self.columns_one_hot = columns_one_hot
        self.columns_standard = columns_standard
        self.columns = columns
        self.unique_categories = unique_categories
        self.encoders = []

    def fit(self, X, y=None):
        for i in range(X.shape[-1]):
            name = self.columns[i]
            if name in self.columns_standard:
                encoder = StandardScaler()
                encoder.fit(X.iloc[:,i:i+1])
            self.encoders.append(encoder)
        return self

    def transform(self, X, y=None):
        transformed = []
        for i, encoder in enumerate(self.encoders):
            name = self.columns[i]
            if name in self.columns_standard:
                tr = encoder.transform(X.iloc[:, i:i+1])
                transformed.append(tr)
            else:
                mapping = self.unique_categories[name]
                tr = X[name].map(mapping).values.reshape(-1,1)
                transformed.append(tr)
                
        return np.concatenate(transformed, axis=-1)
    
brace_dict = {    
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

brace_dict_inv = dict(zip(brace_dict.values(), brace_dict.keys()))

N_BRACES = len(brace_dict)
max_layers = total_max_layers
transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"]
brace_cols = ["brace" + str(i) for i in range(max_layers-1)] 
transformed_columns += brace_cols
transformed_columns += ["layer_height" + str(i) for i in range(max_layers-2)]
Y = {}
X = {}
unique_categories = {}
y_columns_to_extract = ["Cost", "compression", "tipover", "torque", "combined", "wave"]
columns_one_hot = brace_cols
columns_standard = list(set(transformed_columns) - set(brace_cols))

    

X_real, Y_real  = encode_dataset(real, y_columns_to_extract, transformed_columns, brace_dict, N_BRACES)



for keys, values in DATA_ARRAY.items():
    X[keys] = {}
    Y[keys] = {}
    for keys_seed, values_seed in values.items():
        X[keys][keys_seed] = {}
        X[keys][keys_seed]["synthetic"] = []
        Y[keys][keys_seed] = {}
        Y[keys][keys_seed]["synthetic"] = []

        for dataset in values_seed:
            x, y = encode_dataset(dataset, y_columns_to_extract, transformed_columns, brace_dict, N_BRACES)

            unique_categories = get_unique_all([X_real, x], columns_one_hot)

            enc_columntransfo = CustomTransformer(columns_one_hot, columns_standard, unique_categories, X_real.columns)
            cty = MinMaxScaler()
            X_real_transformed = pd.DataFrame(enc_columntransfo.fit_transform(X_real))
            y_real_transformed = cty.fit_transform(Y_real)

            X[keys][keys_seed]["real"] = X_real_transformed
            X[keys][keys_seed]["synthetic"].append(enc_columntransfo.transform(x))

            Y[keys][keys_seed]["real"] = y_real_transformed
            Y[keys][keys_seed]["synthetic"].append(cty.transform(y))

import numpy as np
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import NotFittedError

def compute_metrics(y_test, y_pred, problem):
    return [m(y_test, y_pred) for m in problem["metrics"]]

def train_eval(X_train, y_train, X_test, y_test, problem):
    model = problem["model"](**problem["args"])
    try:
        model.predict(X_test)
        print("Model is already fitted!")
        exit(1)
    except:
        pass
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, problem)
    return metrics, y_pred

def make_problem(X_real, y_real, X_s, y_s, problem, names_all, print_percentages=False):
    if "regression" in problem.keys():
        X_r = np.array(X_real)
        y_r = np.array(y_real[:,problem["regression"]])
        ll = []
        for y in y_s:
            ll.append(np.array(y[:,problem["regression"]]))
    else:
        X_r = np.array(X_real)
        threshold = problem["threshold_metric"](y_real[:,problem["classification"]])
        y_r_binary = (y_real[:,problem["classification"]] > threshold).astype(int)
        y_r = np.array(y_r_binary)
        if print_percentages:
            print('\n'.join(f'{names_all[0]} Class {c}: {count/len(y_r)*100:.2f}%' for c, count in enumerate(np.bincount(y_r))))

        ll = []
        for y, n in zip(y_s, names_all[1:]):
            threshold = problem["threshold_metric"](y[:,problem["classification"]])
            y_binary = (y[:,problem["classification"]] > threshold).astype(int)
            y_binary = np.array(y_binary)
            if print_percentages:
                print('\n'.join(f'{n} Class {c}: {count/len(y_binary)*100:.2f}%' for c, count in enumerate(np.bincount(y_binary))))

            ll.append(y_binary)
    return X_r, y_r, [np.array(x) for x in X_s], ll

def run_experiments(X_real, y_real, synthetic_Xs, synthetic_ys, problems, names_all, output_names, bw=0.5,
                     plot=False, augment=False, num_repeats = 1, num_folds = 2):
    problems_all = []
    for problem in problems:
        print("Model", problem["model_name"])    

        X_real_, y_real_, synthetic_Xs_, synthetic_ys_ = make_problem(X_real, y_real, synthetic_Xs, synthetic_ys, problem, names_all)
        # Assuming you have X and y defined
        # Create an MLP regression model

        # Set up repeated cross-validation

        rkf = RepeatedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=42)
        all_metrics_mean = []
        all_metrics_std = []
        # for train_i in range(len(names_in)):
        #     metrics_all = [[]]*len(names_in)
        metrics_all = []
        for i, (train_index, test_index) in enumerate(rkf.split(X_real_)):    
            # print("split", i)
            X_train_real, X_test_real = X_real_[train_index], X_real_[test_index]
            y_train_real, y_test_real = y_real_[train_index], y_real_[test_index]


            X_trains = [X_train_real]
            y_trains = [y_train_real]
            X_tests = [X_test_real]
            y_tests = [y_test_real]

            metrics_split = []
            for X_synthetic, y_synthetic in zip(synthetic_Xs_, synthetic_ys_):
                X_trains.append(X_synthetic)
                y_trains.append(y_synthetic)
            k = 0 

            for X_tr, y_tr in zip(X_trains, y_trains):
                # print("synthetic", k)
                setup_metrics = []
                preds = []  
                for X_t, y_t in zip(X_tests, y_tests):
                    results, pred = train_eval(X_tr, y_tr, X_t, y_t, problem)
                    setup_metrics.append(results)
                    preds.append(pred)
                k += 1
                metrics_split.append(setup_metrics)
                
            metrics_all.append(metrics_split)
            # Calculate the overall average scores across all repeats
        metrics_all = np.array(metrics_all)    
        problems_all.append(metrics_all)
    return np.array(problems_all)

def reshape_get_mean(arr):
    original_shape = arr.shape
    new_shape = (original_shape[0], original_shape[1]*original_shape[2], *original_shape[3:])
    result = arr.reshape(new_shape)
    average = np.mean(result, axis=1)
    std = np.std(result, axis=1)
    return average, std

def get_real_synth_means(outputs):
    re = outputs[:,:,:1, :, :]
    sy = outputs[:,:,1:, :, :]
    re_av, re_std = reshape_get_mean(re)
    sy_av, sy_std = reshape_get_mean(sy)
    return re_av, re_std, sy_av, sy_std

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

all_columns = list(range(y_real_transformed.shape[1]))



#random forest, boosting, xgb (forest), snn (sequential nn)

problem_regression = {"regression":all_columns, 
                      "metrics":[mean_squared_error, r2_score, mean_absolute_percentage_error],
                      "metric_names":["MSE", "R2", "MAPE"]}
                      
# models = [MultiOutputRegressor(LGBMRegressor(random_state=42)), DecisionTreeRegressor(random_state=42), RandomForestRegressor(random_state=42)]
models = [CatBoostRegressor, DecisionTreeRegressor, RandomForestRegressor]
args = [{"random_state":42, "loss_function":"MultiRMSE", "verbose":False, "iterations":100, "learning_rate":0.01}, {"random_state":42}, {"random_state":42}]

model_names = ["CBR", "DT", "RF"]
problems_regression = []
for model, name, arg in zip(models, model_names, args):
    problem = problem_regression.copy()
    problem["model"] = model
    problem["model_name"] = name
    problem["args"] = arg
    problems_regression.append(problem)

names_all = DATA_NAMES
print(names_all)

problems_used = problems_regression

x_axis = []
y_axis = []
names_axis = []
seeds_all = []
from collections import defaultdict

results_out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
results_out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

for keys, values in X.items():
    for keys_seed, values_seed in values.items():
        print(keys, "-", keys_seed)
        seeds_all.append(keys_seed)
        X_real_transformed = X[keys][keys_seed]["real"]
        y_real_transformed = Y[keys][keys_seed]["real"]

        outputs = run_experiments(X_real_transformed, y_real_transformed, X[keys][keys_seed]["synthetic"], Y[keys][keys_seed]["synthetic"], problems_used, names_all, Y_real.columns, num_folds=3, num_repeats=3)
        average_real, std_real, average_synth, std_synth =  get_real_synth_means(outputs)
        for i in range(len(problems_used)):
            model_name = problems_used[i]["model_name"]
            for j, metr in enumerate(problems_used[i]["metric_names"]):
                results_out[keys_seed][model_name][metr]["real"]["average"] = average_real[i][0][j]
                results_out[keys_seed][model_name][metr]["real"]["std"] = std_real[i][0][j]
                results_out[keys_seed][model_name][metr][keys]["average"] = average_synth[i][0][j]
                results_out[keys_seed][model_name][metr][keys]["std"] = std_synth[i][0][j]
            # break
        # break
            
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [defaultdict_to_dict(item) for item in d]
    else:
        return d

# Convert the nested defaultdict to a regular dictionary
regular_dict = defaultdict_to_dict(results_out)


import json
output_filename = "number_of_instances_100.json"
with open(output_filename, "w") as json_file:
    json.dump(regular_dict, json_file, indent=4)