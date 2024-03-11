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

def execute_run(n_pop, n_gen, objectives, objective_args, constraints, probabilities, sampling=False, survival=False, objective_thresholds=False, termination=False, combine_stabilities=False):
    # prepare the algorithm run
    run_settings = RunSettings(population=n_pop, generations=n_gen, objectives=objectives,
                            constraints=constraints, combine_stabilities=combine_stabilities, probabilities=probabilities, sampling=sampling,
                            survival=survival, objective_args=objective_args, objective_thresholds=objective_thresholds, termination=termination)
    alg_wrapper = AlgorithmWrapper(run_settings)
    alg_wrapper.setup()
    cb = PrintCallback()

    result = alg_wrapper.run(cb)
    output = "optimal_run_pop_{}_gen_{}_plausibility_{}".format(n_pop, n_gen, REAL_LIFE_LOADS.get("plausibility"))

    if objective_args:
        output += "_".join(["_{}_{:.1f}".format(k, v) if isinstance(v, (int, float)) else "_{}_{}".format(k, v) for k, v in objective_args.items()])
        print(output)
    if sampling:
        output += "_sampling_real"
    if WEIGHTED:
        output += "_weighted_{}".format(weight_scale)
    if CONSTRAIN:
        output += "_constrained_{}".format(OBJECTIVES_CONSTRAINTS_CHOOSE)
    output += ".json"
    with open(output, "w") as f:
        json.dump([x[0] for x in result.X], f, indent=4)
    print("Saved file", output)


def process_objective_args(nu):
    objective_args = {"nu": nu}
    execute_run(n_pop, n_gen, REAL_LIFE_LOADS, objective_args, constraints, False, probs, sampling, survival)


print(REAL_LIFE_LOADS)
SAMPLING_REAL = False
WEIGHTED = False
weight_scale = 10
CONSTRAIN = False
OBJECTIVES_CONSTRAINTS_CHOOSE = "mean_3s"


# NU = np.arange(0, 1.1, 0.1).tolist()
NU = 0.5

import json
with open("../util/real_structures.json") as f:
    real = json.load(f)

sampling = False
if SAMPLING_REAL:
    sampling = RealSubstructureSampler(real)

if CONSTRAIN:
    irrelevant = set(real[0].keys())
    import pandas as pd
    with open("../util/real_structures_evaluated.json") as f:
        real_eval = json.load(f)
    
    df_real_eval = pd.DataFrame(real_eval)
    col_set = set(df_real_eval.columns)
    objectives = list(col_set - irrelevant)
    df_real_eval_objs = df_real_eval[objectives]
    objectives_max = dict(df_real_eval_objs.max())
    objectives_mean = dict(df_real_eval_objs.mean())
    objectives_std = dict(df_real_eval_objs.std())
    objectives_mean_plus_1_std = dict(df_real_eval_objs.mean() + df_real_eval_objs.std())
    objectives_mean_plus_2_std = dict(df_real_eval_objs.mean() + 2*df_real_eval_objs.std())
    objectives_mean_plus_3_std = dict(df_real_eval_objs.mean() + 3*df_real_eval_objs.std())
    if OBJECTIVES_CONSTRAINTS_CHOOSE=="mean":
        OBJECTIVES_CONSTRAINTS = objectives_mean
    elif OBJECTIVES_CONSTRAINTS_CHOOSE=="max":
        OBJECTIVES_CONSTRAINTS = objectives_max
    elif OBJECTIVES_CONSTRAINTS_CHOOSE=="mean_s":
        OBJECTIVES_CONSTRAINTS = objectives_mean_plus_1_std
    elif OBJECTIVES_CONSTRAINTS_CHOOSE=="mean_2s":
        OBJECTIVES_CONSTRAINTS = objectives_mean_plus_2_std
    elif OBJECTIVES_CONSTRAINTS_CHOOSE=="mean_3s":
        OBJECTIVES_CONSTRAINTS = objectives_mean_plus_3_std
    else:
        print(OBJECTIVES_CONSTRAINTS_CHOOSE, "not implemented")
        exit(1)

# using default constraints here for simplicity
constraints = SimpleConstraints(total_height=(17, 95), radius_bottom=(9, 29), radius_top=(2, 18), legs=(3, 4), n_layers=(2, 6))

# some settings for the algorithm run
n_pop = 100
n_gen = 2

# define mutation and crossover probabilities (only used for SubstructureSimple)
probs = {"mutation": 0.2, "crossover": 0.8}

REAL_LIFE_LOADS["plausibility"] = "none"

survival = False
if WEIGHTED:
    metrics = MetricFunction.from_loads(REAL_LIFE_LOADS)
    plausibility_metric_index = [i for i, string in enumerate(metrics.metric_names) if "plausibility" in string][0]
    survival = WeightedRankAndCrowdingSurvival(index=plausibility_metric_index, scaling=weight_scale)

from pymoo.termination.default import DefaultMultiObjectiveTermination

termination = DefaultMultiObjectiveTermination(
    xtol=1e-8,
    cvtol=1e-6,
    ftol=0.0025,
    period=30,
    n_max_gen=1000,
    n_max_evals=100000
)

objective_thresholds = False
if CONSTRAIN:
    plausibility_key = get_relevant_plausibility(REAL_LIFE_LOADS["plausibility"], OBJECTIVES_CONSTRAINTS)
    plausibility_threshold = OBJECTIVES_CONSTRAINTS[plausibility_key]

    objective_thresholds = {plausibility_key: plausibility_threshold}

objective_args=False
if REAL_LIFE_LOADS["plausibility"].lower() == "one class svm":
    objective_args= {"nu":NU}
    
execute_run(n_pop, n_gen, REAL_LIFE_LOADS, objective_args, constraints, probs, sampling, survival, objective_thresholds=objective_thresholds, termination=False)
