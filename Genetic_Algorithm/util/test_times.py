import sys
sys.path.append("..")
from graphstructure.sample import SimpleConstraints
from graphstructure.algorithm import run_algorithm, PlotCallback, PrintCallback
from graphstructure.app_interface import AlgorithmWrapper, RunSettings
from graphstructure.metrics import REAL_LIFE_LOADS
import json

# using default constraints here for simplicity
constraints = SimpleConstraints(total_height=(17, 95), radius_bottom=(9, 29), radius_top=(2, 18), legs=(3, 4), n_layers=(2, 6))
probs = {"mutation": 0.2, "crossover": 0.8}
REAL_LIFE_LOADS["plausibility"] = 'FAMD GMM'
n_gen = 50

import time
import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to store results
target_values = range(1000, 2001, 100)
execution_times = []

for n_pop in target_values:
    times_for_target = []
    print(f"Processing population = {n_pop}")
    start_time = time.time()

    # prepare the algorithm run
    run_settings = RunSettings(population=n_pop, generations=n_gen, objectives=REAL_LIFE_LOADS,
                            constraints=constraints, combine_stabilities=False, probabilities=probs)
    alg_wrapper = AlgorithmWrapper(run_settings)
    alg_wrapper.setup()
    cb = PrintCallback()

    result = alg_wrapper.run(cb)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Done population", n_pop, " time ", execution_time)
    execution_times.append(execution_time)

# Save results to a file
with open('execution_times.txt', 'w') as f:
    for target_nb_pseudo_obs, mean_time in zip(target_values, execution_times):
        f.write(f'Target obs: {target_nb_pseudo_obs}, Mean Time: {mean_time:.4f}\n')


