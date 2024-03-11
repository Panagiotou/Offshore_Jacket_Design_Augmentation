import sys
sys.path.append("..")
from graphstructure.sample import SimpleConstraints
from graphstructure.algorithm import run_algorithm, PlotCallback, RealSubstructureSampler, PrintCallback, WeightedRankAndCrowdingSurvival
from graphstructure.app_interface import AlgorithmWrapper, RunSettings
from graphstructure.metrics import REAL_LIFE_LOADS, MetricFunction, get_plausibility_metric_name
import json
import numpy as np
import click

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

def execute_run(n_pop, n_gen, objectives, objective_args, constraints, probabilities, sampling=False, survival=False, objective_thresholds=False, termination=False, combine_stabilities=False, seed=0, folder="", label=""):
    # prepare the algorithm run
    run_settings = RunSettings(population=n_pop, generations=n_gen, objectives=objectives,
                            constraints=constraints, combine_stabilities=combine_stabilities, probabilities=probabilities, sampling=sampling,
                            survival=survival, objective_args=objective_args, objective_thresholds=objective_thresholds, termination=termination)
    alg_wrapper = AlgorithmWrapper(run_settings)
    alg_wrapper.setup()
    cb = PrintCallback()

    result = alg_wrapper.run(cb, seed=seed)

    if objective_thresholds:
        output = "P[{}]_thr_{:.2f}".format(REAL_LIFE_LOADS.get("plausibility"), list(objective_thresholds.values())[0])
    else:
        output = "P[{}]".format(REAL_LIFE_LOADS.get("plausibility"))
    if label:
        output += "_{}".format(label)
   
    output += "_seed_" + str(seed) 
    output += ".json"
    with open(os.path.join(folder,output), "w") as f:
        json.dump([x[0] for x in result.X], f, indent=4)
    print("Saved file", output)

import os
@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--objective_threshold', default="none", help='Thresh')
@click.option('--folder', default="ga_results", help='folder out', type=str)
@click.option('--plausibility', default="One Class SVM", help='folder out', type=str)
@click.option('--max_gen', default=100, help='Number of generations', type=int)
@click.option('--pop', default=100, help='Population', type=int)
@click.option('--label', default="", help='label', type=str)
@click.option('--nu', default="0.5", help='nu', type=str)
def run(seed, objective_threshold, folder, plausibility, max_gen, pop, label, nu):
    NU = float(nu)

    sampling = False


    # using default constraints here for simplicity
    constraints = SimpleConstraints(total_height=(17, 95), radius_bottom=(9, 29), radius_top=(2, 18), legs=(3, 4), n_layers=(2, 6))

    # some settings for the algorithm run
    n_pop = pop
    n_gen = -1

    # define mutation and crossover probabilities (only used for SubstructureSimple)
    probs = {"mutation": 0.2, "crossover": 0.8}

    REAL_LIFE_LOADS["plausibility"] = plausibility

    folder += "_" + REAL_LIFE_LOADS["plausibility"]

    objective_args=False
    plausibility_key = get_plausibility_metric_name(REAL_LIFE_LOADS["plausibility"], "scale", NU)

    if "one class svm" in REAL_LIFE_LOADS["plausibility"].lower():
        objective_args= {"nu":NU}


    survival = False

    from pymoo.termination.default import DefaultMultiObjectiveTermination

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=max_gen,
        n_max_evals=100000
    )
    objective_thresholds = False

    if str(objective_threshold).lower() == "none":
        objective_thresholds = False
    else:
        objective_thresholds = {plausibility_key: float(objective_threshold)}

    print(objective_thresholds)


    if objective_args:
        folder += "_".join(["_{}_{:.2f}".format(k, v) if isinstance(v, (int, float)) else "_{}_{}".format(k, v) for k, v in objective_args.items()])

    if not os.path.exists(folder):
        os.makedirs(folder)
    execute_run(n_pop, n_gen, REAL_LIFE_LOADS, objective_args, constraints, probs, sampling, survival, objective_thresholds=objective_thresholds, termination=termination, seed=seed, folder=folder, label=label)

if __name__ == '__main__':
    run()