import sys
import json
import numpy as np
import pandas as pd

sys.path.append("..")

from graphstructure.simple_substructure import SubstructureSimple
from graphstructure.metrics import MetricFunction, REAL_LIFE_LOADS

metrics = MetricFunction.from_loads(REAL_LIFE_LOADS)

import math
def test_all(designs):
    objs = [] 
    evaluated_designs = []
    k=0
    for d in designs[71600:]:
        sub = SubstructureSimple.from_dict(d)
        threshold = 0.001
        l_h = all(abs(d['layer_heights'][i] - d['layer_heights'][i+1]) < threshold for i in range(len(d['layer_heights'])-1))
        if not l_h:
            m_v = metrics(sub)
            obj_values = np.array(m_v[0])
            if any([math.isinf(x) for x in obj_values]):
                print('This design cant be tested skipping:', d)
                continue
                # print('This design cant be tested in ops and thus got scores of 0.0:', d)
                # obj_values[np.where(obj_values == 0.0)] = np.inf
            objs.append(obj_values)
            evaluated_designs.append(d)
        else:
            print('This design cant be tested skipping:', d)
            continue

        print(k)
        k+=1
    return objs, evaluated_designs

def to_df(objs):
    # print(metrics.metric_names)
    return pd.DataFrame(objs, columns=metrics.metric_names)

import os
def evaluate(designs, file_path, out_dir):
    objs, evaluated_designs = test_all(designs)
    df = to_df(objs)
    data_merged = [{**d, **df.loc[i]} for i, d in enumerate(evaluated_designs)]
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = out_dir + filename +"_evaluated.json"
    with open(output_filename, "w") as json_file:
        json.dump(data_merged, json_file, indent=4)
    print("Saved file", output_filename, "total designs", len(evaluated_designs))    


def copy_folder_structure(in_dir, out_dir):
    for root, dirs, files in os.walk(in_dir):
        relative_path = os.path.relpath(root, in_dir)
        new_dir = os.path.join(out_dir, relative_path)
        os.makedirs(new_dir, exist_ok=True)

out_dir = "data/evaluated/"
in_dir = "data/to_evaluate/"
copy_folder_structure(in_dir, out_dir)

folders = [d for d in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, d))]

for folder in folders:
    files = os.listdir(in_dir + folder)
    for file_ in files:
        with open(in_dir + folder + "/" + file_) as f:
            synthetic = json.load(f)
            evaluate(synthetic, file_, out_dir + folder + "/")


files = os.listdir(in_dir)


for file_ in files:
    with open(in_dir + file_) as f:
        synthetic = json.load(f)
        evaluate(synthetic, file_, out_dir)
