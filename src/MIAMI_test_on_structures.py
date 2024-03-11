import sys
sys.path.append("src")

import os 

import pandas as pd
from copy import deepcopy
from gower import gower_matrix
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder 

from MIAMI import MIAMI
from init_params import dim_reduce_init
from data_preprocessing import compute_nj

from shapely.geometry import Polygon as polygon

import autograd.numpy as np

import colorcet as cc
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import ListedColormap
import seaborn as sns


main_folder = 'results/structures/'
res_folder = main_folder + "MIAMI/"

data_folder = main_folder + "data/"   




    
dtypes_dict = {'continuous': float, 'categorical': str, 'ordinal': int,\
              'bernoulli': int, 'binomial': int}
    
dtypes_dict_famd = {'continuous': float, 'categorical': str, 'ordinal': str,\
              'bernoulli': str, 'binomial': str}
#===========================================#
# Importing data
#===========================================#

inf_nb = 1E12

sub_design = "bivariate"

# acceptance_rate =
le_dict = {}



import json
with open(data_folder + "optimal_run_pop_200_gen_50.json") as f:
    synthetic_orig = json.load(f)
with open(data_folder + "real_structures.json") as f:
    real_orig = json.load(f)
with open(data_folder + "random_dataset_1000_designs.json") as f:
    rand_orig = json.load(f)
    
# DATA_ARRAY = [real_orig, synthetic_orig]
# DATA_NAMES = ["real", "synthetic (GA)"]
DATA_ARRAY = [real_orig]
DATA_NAMES = ["real"]

   
# from https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

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
def encode(d, max_layers, one_hot=True, native=False):
    basics = [d["legs"], d["total_height"], d["radius_bottom"], d["radius_top"], d["n_layers"]]
    
    # fill design's braces according to max_layers with dummies ("NONE")
    braces = d["connection_types"]
    if native:
        braces = np.array([b for b in braces] + ["nan"] * (max_layers - 1 - len(braces)))
    else:
        braces = np.array([brace_dict[b] for b in braces] + [brace_dict["nan"]] * (max_layers - 1 - len(braces)))
        if one_hot:
            braces = get_one_hot(braces, N_BRACES)
    
    # fill design's layer_heights according to max_layers with dummies
    layer_heights = d["layer_heights"]
    layer_heights = np.array(layer_heights + [d["total_height"]] * (max_layers - 2 - len(layer_heights))) / d["total_height"]
    
    # return a flat encoding
    return np.array([*basics, *braces.flatten(), *layer_heights])

def get_cols(d, max_layers, one_hot=True, native=False):
    
    # fill design's braces according to max_layers with dummies ("NONE")
    braces = d["connection_types"]
    if native:
        braces = np.array([b for b in braces] + ["nan"] * (max_layers - 1 - len(braces)))
    else:
        braces = np.array([brace_dict[b] for b in braces] + [brace_dict["nan"]] * (max_layers - 1 - len(braces)))
        if one_hot:
            braces = get_one_hot(braces, N_BRACES)
    
    # fill design's layer_heights according to max_layers with dummies
    layer_heights = d["layer_heights"]
    layer_heights = np.array(layer_heights + [d["total_height"]] * (max_layers - 2 - len(layer_heights))) / d["total_height"]
    transformed_columns = ["legs", "total_height", "radius_bottom", "radius_top", "n_layers"] + ["brace" + str(i) for i in range(len(braces.flatten()))] + ["layer_height" + str(i) for i in range(len(layer_heights))]
    
    # return a flat encoding
    return transformed_columns, ["brace" + str(i) for i in range(len(braces.flatten()))]

max_layers = max(d.get('n_layers', 0) for sublist in DATA_ARRAY for d in sublist)

transformed_columns = []
brace_cols = []
encoded = []
for dataset, name in zip(DATA_ARRAY, DATA_NAMES):
    
    encoding = [encode(d, max_layers, one_hot=False, native=True) for d in dataset]
    transformed_columns_, brace_cols_ = get_cols(dataset[0], max_layers, one_hot=False)
    if len(transformed_columns_)> len(transformed_columns):
        transformed_columns = transformed_columns_
        brace_cols = brace_cols_
    encoded.append(encoding)
    
dataframes = []
for count, encoding in enumerate(encoded):
    df_ = pd.DataFrame(encoding, columns=transformed_columns)
    df_["label"] = [DATA_NAMES[count]]*len(df_)
    dataframes.append(df_.copy())
    
train_original = pd.concat(dataframes, axis=0, ignore_index=True)

# Concatenate the two dataframes together and reindex
# train_original = pd.concat([real, synthetic, rand], axis=0, ignore_index=True)

# synth_original = pd.concat([synthetic], axis=0, ignore_index=True)

# train_original = pd.concat([real], axis=0, ignore_index=True)

nominal_features = brace_cols
ordinal_features = ["n_layers", "legs"]
BERNOULLI = ["legs"]

discrete_features = nominal_features + ordinal_features


continuous_features = list(set(transformed_columns) - set(nominal_features) - set(ordinal_features))

train_original[ordinal_features] = train_original[ordinal_features].astype("int")
train_original[continuous_features] = train_original[continuous_features].astype("float")

# synth_original[ordinal_features] = synth_original[ordinal_features].astype("int")
# synth_original[continuous_features] = synth_original[continuous_features].astype("float")

train = train_original.drop("label", axis=1)

# synth = synth_original.drop("label", axis=1)





train = train.infer_objects()
# synth = synth.infer_objects()

numobs = len(train)
print("Running with", numobs, "observations!!!!")

#*****************************************************************
# Formating the data
#*****************************************************************
# 
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
        if colname in BERNOULLI:
            var_transform_only.append('bernoulli')
        else:
            var_transform_only.append('categorical')
            
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype==int and unique > 2:
        # ordinal
        
        var_distrib.append('ordinal')
        var_transform_only.append('ordinal')
        
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype == "object":
        var_distrib.append('categorical')
        var_transform_only.append('categorical')
        
        le = LabelEncoder()
        # Convert them into numerical values               
        train[colname] = le.fit_transform(train[colname]) 
        le_dict[colname] = deepcopy(le)
    elif dtype == float:
        var_distrib.append('continuous')
        var_transform_only.append('continuous')
        
    
var_distrib = np.array(var_distrib)



nj, nj_bin, nj_ord, nj_categ = compute_nj(train, var_distrib)

nb_cont = np.sum(var_distrib == 'continuous')     

p = train.shape[1]
        
# Feature category (cf)
dtype = {train.columns[j]: dtypes_dict_famd[var_transform_only[j]] for j in range(p)}

train_famd = train.astype(dtype, copy=True)
numobs = len(train)

# authorized_ranges = np.expand_dims(np.stack([[-np.inf,np.inf] for var in var_distrib]).T, 1)
authorized_ranges = None

#===========================================#
# Model Hyper-parameters
#===========================================#

n_clusters = 3
r = np.array([2, 1])
k = [n_clusters]

seed = 2023
init_seed = 2024
    
# !!! Changed eps
eps = 1E-05
it = 50
maxstep = 100
target_nb_pseudo_obs = 100
# sample 5 times because we loose some from constraints
nb_points= 5*target_nb_pseudo_obs

#*****************************************************************
# Run MIAMI
#*****************************************************************
print("Initialize dimensionality reduction")    
init, transformed_famd_data, famd  = dim_reduce_init(train_famd, n_clusters, k, r, nj, var_distrib, seed = 2023,\
                                use_light_famd=True)

print(init.keys())
print(len(init["eta"][0]))
print(len(init["H"]))

print("Computing distance matrix")
# # Defining distances over the features
# dm = gower_matrix(train, cat_features = cat_features) 

distances = pdist(transformed_famd_data)

dm = squareform(distances)
print("done")

print("Training MIAMI with", train.shape)
out = MIAMI(train, n_clusters, r, k, init, var_distrib, nj, le_dict, authorized_ranges, target_nb_pseudo_obs=target_nb_pseudo_obs, nb_points=nb_points, it=it,\
                eps=eps, maxstep=maxstep, seed=seed, perform_selec = True, dm = dm, max_patience = 0)
print('MIAMI has kept one observation over', round(1 / out['share_kept_pseudo_obs']),\
        'observations generated')
    
acceptance_rate = out['share_kept_pseudo_obs']
print(acceptance_rate)
pred = pd.DataFrame(out['y_all'], columns = train.columns) 
print(pred.shape)

import re

#================================================================
# Inverse transform the datasets
#================================================================
pred_trans = pred.copy()

for j, colname in enumerate(train.columns):
    if colname in le_dict.keys():
        pred_trans[colname] = le_dict[colname].inverse_transform(pred[colname].astype(int))
    
pred_post = pred_trans.copy()    
pred_famd = pred.copy()    

pred_famd[discrete_features] = pred_famd[discrete_features].astype(int)


layer_height_cols = pred_post.filter(like='layer_height').columns.tolist()

# Define a function to apply to each row
def replace_layer_height(row, brace_cols, layer_height_cols):
    n_layers = int(row['n_layers'])
    layers_set_one = layer_height_cols[n_layers-2:]
    braces_set_nan = brace_cols[n_layers-1:]
    # Set the layer_height columns to 1.0 for all rows after the nth row
    row[layers_set_one] = 1.0
    row[braces_set_nan] = "nan"
    
    # Return the modified row
    return row

# Apply the function to each row of the DataFrame
pred_post = pred_post.apply(lambda x: replace_layer_height(x, brace_cols, layer_height_cols), axis=1)

pred_famd = pred_famd.astype(dtype, copy=True)

pred_post_famd = pred_famd.copy()
pred_post_famd[layer_height_cols] = pred_post[layer_height_cols]



pred_post_famd[discrete_features] = pred_post_famd[discrete_features].astype(int)

pred_post_famd = pred_post_famd.astype(dtype, copy=True)
    
    
print("Saved to", res_folder + 'preds' + str(target_nb_pseudo_obs) +'.csv')
# Store the predictions
pred_post.to_csv(res_folder + 'preds' + str(target_nb_pseudo_obs) +'.csv', index = False)
#break

import json
def transform_df_to_json(input_df):
    output = []
    id_counter = 1

    for index, row in input_df.iterrows():
        n_layers = row["n_layers"]
        connection_types = [row[f"brace{i}"] for i in range(n_layers) if row[f"brace{i}"] != "nan"]
        layer_heights = [row[f"layer_height{i}"] for i in range(n_layers-2)]

        output.append({
            "legs": row["legs"],
            "total_height": row["total_height"],  # Dividing total_height by 2 as shown in the example
            "connection_types": connection_types,
            "radius_bottom": row["radius_bottom"] + row["radius_top"],
            "radius_top": row["radius_top"],
            "n_layers": row["n_layers"],
            "layer_heights": layer_heights,
            "id": id_counter
        })

        id_counter += 1

    return output
json_output = transform_df_to_json(pred_post)
output_filename = res_folder + 'preds' + str(target_nb_pseudo_obs) +".json"
with open(output_filename, "w") as json_file:
    json.dump(json_output, json_file, indent=4)