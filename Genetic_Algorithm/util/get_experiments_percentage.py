import sys
sys.path.append("..")
from graphstructure.sample import SimpleConstraints
from graphstructure.algorithm import run_algorithm, PlotCallback, RealSubstructureSampler, PrintCallback, WeightedRankAndCrowdingSurvival
from graphstructure.app_interface import AlgorithmWrapper, RunSettings
from graphstructure.metrics import REAL_LIFE_LOADS, MetricFunction
import json
import numpy as np

def print_greater_number():
    if not hasattr(print_greater_number, 'prev_number'):
        print_greater_number.prev_number = -1

    n = print_greater_number.prev_number + 1
    print(n)
    print_greater_number.prev_number = n

def get_relevant_plausibility(plausibility_string, objectives_thresholds):
    plausibility_string = plausibility_string.lower()
    for key in objectives_thresholds.keys():
        if "famd" in plausibility_string:
            if "famd" in key.lower() and plausibility_string in key.lower():
                return key
        else:
            if "famd" not in key.lower() and plausibility_string in key.lower():
                return key
    return None

def get_settings_name(objectives, objective_args, constraints, probabilities, sampling=False, survival=False, objective_thresholds=False, termination=False, combine_stabilities=False):
    # prepare the algorithm run
    run_settings = RunSettings(objectives=objectives,
                            constraints=constraints, combine_stabilities=combine_stabilities, probabilities=probabilities, sampling=sampling,
                            survival=survival, objective_args=objective_args, objective_thresholds=objective_thresholds, termination=termination)
   
    output = "PL[{}]".format(REAL_LIFE_LOADS.get("plausibility"))

    if objective_args is not None:
        output += "_".join(["_{}_{:.1f}".format(k, v) if isinstance(v, (int, float)) else "_{}_{}".format(k, v) for k, v in objective_args.items()])
        print(output)
    if sampling:
        output += "_sampling_real"
    if WEIGHTED:
        output += "_weighted_{}".format(weight_scale)
    if CONSTRAIN:
        output += "_constrained_{}".format(OBJECTIVES_CONSTRAINTS_CHOOSE)
    output += ".json"
    return run_settings, output


print(REAL_LIFE_LOADS)
SAMPLING_REAL = False
WEIGHTED = False
weight_scale = 10
CONSTRAIN = True
OBJECTIVES_CONSTRAINTS_CHOOSE = "mean_3s"


# NU = np.arange(0, 1.1, 0.1).tolist()
NU = 0.5

import json
with open("../util/real_structures.json") as f:
    real = json.load(f)
import pandas as pd
with open("../util/real_structures_evaluated_nu.json") as f:
    real_eval = json.load(f)
with open("../util/data/optimal_run_pop_100_gen_10000_plausibility_None_evaluated.json") as f:
    ga_no_plaus = json.load(f)

irrelevant = set(real[0].keys())

metrics_all = ['plausibility (FAMD GMM)', 'plausibility (FAMD Mahalanobis)',
 'plausibility (FAMD One Class SVM) gamma=scale nu=0.5', 'plausibility (One Class SVM) gamma=scale nu=0.5',
  "plausibility (KDE)",
  'plausibility (Isolation Forest)', "plausibility (local outlier factor)"]
metric_names_all = ['FAMD GMM', 'FAMD Mahalanobis',
 "FAMD One Class SVM", "One Class SVM",
  "KDE", "Isolation Forest", "local outlier factor"]


nus = [0.25, 0.75, 0.9]
metrics_all = ["plausibility (One Class SVM) gamma=scale nu={}".format(nu) for nu in nus]
metric_names_all = ["One Class SVM"] * len(metrics_all)

what_metric = 0

if len(sys.argv)>1:
    what_metric = int(sys.argv[1])

NU = nus[what_metric]

metric = metrics_all[what_metric]
metric_no_specs = metric_names_all[what_metric]

df_real_eval = pd.DataFrame(real_eval)
col_set = set(df_real_eval.columns)
objectives = list(col_set - irrelevant)
df_real_eval_objs = df_real_eval[objectives][metric]

objectives_median = df_real_eval_objs.median()
objectives_std = df_real_eval_objs.std()

baseline = [0, 0.25, 0.5, 0.75]
stds = baseline + [1+x for x in baseline] + [2+x for x in baseline] + [3+x for x in baseline]
interpolated_values = []
interpolated_names = []
for stdd in stds:
    interpolated_values.append(objectives_median + stdd*objectives_std)
    interpolated_names.append("median_{}s".format(stdd))
    # percentage_under_threshold = (df_real_eval_objs.to_numpy() < objectives_median + stdd*objectives_std).mean() * 100
    # print(percentage_under_threshold)

import concurrent.futures
import subprocess
import signal




def run_script(seed, threshold_value, plausibility_in, max_gen=10, pop=100, label="", nu=0.5):
    specific_args = ['--seed', str(seed), '--objective_threshold', str(threshold_value),
     "--plausibility", str(plausibility_in), "--max_gen", str(max_gen),
      "--pop", str(pop), "--label", str(label), "--nu", str(nu)]
    script = ['python', 'run_algo_inputs.py'] + specific_args
    print(script)
    subprocess.run(script)

def handle_interrupt(signum, frame):
    executor.shutdown(wait=False)
    print("Execution interrupted. Exiting.")
    exit()

# Register the signal handler for Ctrl+C
signal.signal(signal.SIGINT, handle_interrupt)


seed_values = list(range(10))  # Example list of seed values
print(interpolated_values)
GENERATIONS = 100
POPULATION = 100

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for value, name in zip(interpolated_values,interpolated_names):
        # Create a ThreadPoolExecutor
            # Submit each script to the executor
            for seed in seed_values:
                futures.append(executor.submit(run_script, seed, value, metric_no_specs, max_gen=GENERATIONS, pop=POPULATION, label=name, nu=NU))
            # Wait for all the scripts to complete
    for seed in seed_values:
        futures.append(executor.submit(run_script, seed, float('inf'), metric_no_specs, max_gen=GENERATIONS, pop=POPULATION, nu=NU))

    concurrent.futures.wait(futures)




# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = []
#     for seed in seed_values:
#         futures.append(executor.submit(run_script, seed, None, "none", max_gen=GENERATIONS, pop=POPULATION))
#     concurrent.futures.wait(futures)
    