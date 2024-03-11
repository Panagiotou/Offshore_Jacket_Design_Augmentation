"""Various implementations that can be used for the webapp and notebooks.

This file contains definitions of classes and functions that can both be used
by the backend of the webapp aswell as by the various notebooks in ../util/

The `LivePlotDataCallback` is a class that implements the pymoo Callback
interface and sends data through a given web socket. This is mainly used for
live plots in the webapp.

The `RunSettings` class defines all necessary hyperparameters for a run. It is
serializable to json and can thus be easily loaded from files.

The `AlgorithmWrapper` is a class that takes an instance of `RunSettings` and
sets up an algorithm run accordingly and executes it. It also manages things
like early termination with pymoo Callbacks.


In general, this file realizes, that the webapp python script is easier to
overview as it does not have to implement all this functionality. As also
notebooks can make use of some of the functions and classes, this was deemed
a good place to bring the functionality together.
"""

from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.termination import Termination
from pymoo.optimize import minimize
from scipy.cluster import vq

from .algorithm import (
    ComplexSubstructureCrossover,
    ComplexSubstructureMutation,
    SimpleSubstructureCrossover,
    SimpleSubstructureMutation,
    SubstructureDupElim,
    SubstructureProblem,
    SubstructureSampler,
    DEFAULT_PROBABILITIES
)
from .metrics import MetricFunction, REAL_LIFE_LOADS, TestCollection, total_mass
from .sample import ComplexConstraints, SimpleConstraints, SubstructureConstraints


def objectives_to_plot_data(gen: int, data, names: list[str],
                            designs: list[dict], clusters, max_gen: int):
    """Transform plot data to a form that is sendable through a socket.

    Args:
        gen: Generation index of the data.
        data: Objective values, where rows are values for one design and
            columns are objectives.
        names: Objective names, len(names) must be equal to data.shape[1]
        designs: List of designs, corresponding to the objective data.
            E.g. designs[21] was evaluated to data[21] objective values.
        clusters: Objective values (same structure as data), but contains only
            the cluster centroid values.
        max_gen: The number generations in the current run.

    Returns:
        Dict with information to be sent to the webapp.
    """

    # USE THIS IF THERE ARE PROBLEMS WITH SERIALIZATION
    # for d in designs:
    #     for k, v in d.items():
    #         print(k, v, type(v))
    #         if isinstance(v, list):
    #             print([type(x) for x in v])
    return {
        "generation": gen,
        "data": obj_matrix_to_dict(data, names),
        "designs": designs,
        "clusters": obj_matrix_to_dict(clusters, names),
        "max_gen": max_gen
    }

class LivePlotDataCallback(Callback):
    """Callback implementation to send plot data to the webapp.

    A callback class which takes a flask-socketio object to send results
    from the genetic algorithm to the front end.

    Attributes:
        socketio: socketio instance that the data are sent to.
        pop_size: Population size of the run.
        n_gen: Number of generations in the current run.
        values: Copy of the pymoo plot_values.
        terminated: Bool showing if the run has been terminated.
        data_names: Objective names
        clusters: Number of clusters to send. If <= 0, dont cluster the data.
    """

    def __init__(self, socketio, pop_size: int, n_gen: int,
                 obj_names: list[str]) -> None:
        super().__init__()
        self.socketio = socketio
        self.pop_size = pop_size
        self.n_gen = n_gen

        # we know the amount of data points at the start of the run
        entries = self.pop_size
        self.values = np.zeros((entries, len(obj_names)))

        self.terminated = False
        self.data_names = obj_names
        self.clusters = 0

    def notify(self, algorithm, **_):
        """Callback function for when a generation is finished.

        Sends data through the socket. If the algorithm has been terminated,
        no data is sent.

        Args:
            algorithm (pymoo Algorithm): Object that holds the objective data.
            **_: Unused
        """
        if self.terminated:
            return

        this_gen = algorithm.n_gen

        # add new evaluation points to the plot
        self.values[:] = algorithm.pop.get("plot_values")

        if self.clusters <= 0:
            cluster_data = []
        else:
            cluster_data = cluster_means(self.values, self.clusters)

        self.socketio.emit('gen_finished',
                objectives_to_plot_data(
                    this_gen,
                    self.values,
                    self.data_names,
                    [X[0] for X in algorithm.pop.get("X")],
                    cluster_data,
                    self.n_gen
                )
        )

    def terminate(self):
        """ Setter for the terminated bool. """
        self.terminated = True

def cluster_means(data, n: int):
    """Cluster objective data with k-Means.

    Args:
        data: Objective values of designs. Columns are single objectives.
        n: Number of clusters to run k-Means for.

    Returns:
        Cluster centroids. These are not necessarily real points. If you want
        to have a cluster representative, take the design closest to the center.
    """

    if n == 1:
        return np.expand_dims(np.mean(data, axis=0), axis=0)
    centroids, _ = vq.kmeans(data, n)
    return centroids

def obj_matrix_to_dict(data, names: list[str]):
    """Converts a objective value matrix into a dict

    The keys of the dict are the objective names, the values are the arrays of
    objective values to their respective names.

    Args:
        data: Objective value matrix. First axis is data points, second are
            the objectives.
        names: List of objective names.

    Returns:
        Dict[key:obj_name, value:obj_values]
    """

    if len(data) == 0:
        return {}
        # return {k: [] for k in names}

    assert len(data[0]) == len(names), f"{len(data[0])=} doesnt match {len(names)=} in obj. matrix"
    obj_vectors = data.T
    return {k: obj_vectors[i].tolist() for i, k in enumerate(names)}


def multiply_test_results(t_list: list[float]) -> float:
    """Multiply the objective values of one design.

    This is used, when an algorithm is not optimizing multiple objectives at
    the same time but instead optimizing for a minimum combined objective.

    Args:
        t_list: List of test results.

    Returns:
        Product of all given test values.
    """
    res = 1
    for test in t_list:
        res *= test

    return res

@dataclass
class RunSettings():
    """A collection of settings for an algorithm run.

    Attributes:
        algorithm: String resembling the type of algorithm used.
            Currently, only 'nsga2' and 'ga' are supported.
        population: Population size of the run.
        generations: Number of generations of the run.
        objectives: Dict with load values for each objective.
            E.g. { "torque": 2e10, "compression": 4.6e4 }
        architecture: Decides the used substructure architecture for the run.
            Currently 'simple' or 'complex'.
        constraints: SubstructureConstraints object that sets design
            constraints for the algorithm run.
        combine_stabilities: Bool deciding if the static stability test values
            should be combined into one value.
        probabilities: Dict with probabilities for mutation and crossover.
        tracking: Bool deciding, if genealogical tracking information is saved
            in design dicts throughout the run.
    """

    algorithm: str = "nsga2"
    population: int = 20
    generations: int = 5
    objectives: dict = field(default_factory=lambda: REAL_LIFE_LOADS)
    architecture: str = "simple"
    constraints: SubstructureConstraints = None
    combine_stabilities: bool = True
    probabilities: dict = field(default_factory=lambda: DEFAULT_PROBABILITIES)
    tracking: bool = False
    sampling: bool = False
    survival: bool = False
    objective_args: bool = False
    objective_thresholds: bool = False
    termination: bool = False

    def __post_init__(self) -> None:
        """ Setup the objective_function and constraints if missing. """
        if self.objective_thresholds:
            print("Thresholds detected {}".format(self.objective_thresholds))

        if self.algorithm == "nsga2":
            combination_function = None
        else:
            combination_function = multiply_test_results

        if self.constraints is None:
            if self.architecture == "simple":
                self.constraints = SimpleConstraints()
            else:
                self.constraints = ComplexConstraints()

        if self.combine_stabilities:
            self.objective_function = MetricFunction(
                total_mass,
                TestCollection(self.objectives),
                metric_names=["Cost", "Instability"]
            )
        else:
            if not self.objective_args: 
                self.objective_function = MetricFunction.from_loads(self.objectives)
            else:
                self.objective_function = MetricFunction.from_loads(self.objectives, **self.objective_args)

        self.objective_function.combination_function = combination_function

        self.n_clusters = 0

    def to_dict(self):
        """ Creates a dict containing all settings. """

        constraints_dict = self.constraints.to_dict()
        return {
            "algorithm": self.algorithm,
            "population": self.population,
            "generations": self.generations,
            "load_compression": self.objectives["compression"],
            "load_tipover": self.objectives["tipover"],
            "load_torque": self.objectives["torque"],
            "architecture": ("simple" if isinstance(self.constraints, SimpleConstraints) else "complex"),
            "combine_stabilities": self.combine_stabilities,
            "probabilities": self.probabilities,
            "tracking": self.tracking
        } | constraints_dict

    @staticmethod
    def from_dict(d: dict):
        """Creates a RunSettings object from the given dict.

        Args:
            d: Dictionary to be parsed.

        Returns:
            A RunSettings object.
        """

        arch = d["architecture"]
        if arch == "simple":
            constraints = SimpleConstraints.from_dict(d)
        else:
            constraints = ComplexConstraints.from_dict(d)
        settings = RunSettings(
            algorithm    = d["algorithm"],
            population   = int(d["population"]),
            generations  = int(d["generations"]),
            objectives   = {
                "compression": float(d.get("load_compression", 0)),
                "tipover": float(d.get("load_tipover", 0)),
                "torque": float(d.get("load_torque", 0))
            },
            architecture = arch,
            constraints  = constraints,
            combine_stabilities = d["combine_stabilities"],
            probabilities = d["probabilities"],
            tracking = d["tracking"]
        )

        settings.n_clusters = d["n_clusters"]
        return settings


class CustomTermination(Termination):
    """An implementation of pymoo's Termination class.

    Used for terminating a running algorithm prematurely.

    Attributes:
        n_gens: Number of generations.
        force_termination: Bool that determines if the algorithm should be
            terminated.
    """

    def __init__(self, n_gens) -> None:
        super().__init__()

        self.n_gens = n_gens

    def do_continue(self, algorithm):
        return not (self.force_termination or algorithm.n_gen == self.n_gens)

    def terminate(self):
        self.force_termination = True

    def _update(self, algorithm):
        # simply return relative progress of generations
        return algorithm.n_gen / self.n_gens


@dataclass
class AlgorithmWrapper():
    """A wrapper class for setting up and running a genetic algorithm run.

    Attributes:
        settings: A RunSettings instance deciding the algorithm setup.
        problem: pymoo Problem instance.
        termination: Termination callback.
        callback: Post-generation Callback.
    """

    settings: RunSettings

    def setup(self):
        s = self.settings
        if not s.sampling:
            sampling = SubstructureSampler(s.constraints)
        else:
            sampling = s.sampling

        if s.architecture == "complex":
            crossover = ComplexSubstructureCrossover(s.constraints)
            mutation = ComplexSubstructureMutation(s.constraints)
        else:
            crossover = SimpleSubstructureCrossover(s.constraints, s.probabilities["crossover"], s.tracking)
            mutation = SimpleSubstructureMutation(s.constraints, s.probabilities["mutation"], s.tracking)

        duplicates = SubstructureDupElim()
        if not s.objective_thresholds:
            self.problem = SubstructureProblem(s.objective_function)
        else:
            
            self.problem = SubstructureProblem(s.objective_function, objective_thresholds=s.objective_thresholds)
        if s.algorithm.lower() == "nsga2":
            if not s.survival:   
                self.algorithm = NSGA2(
                    pop_size=s.population, sampling=sampling, crossover=crossover,
                    mutation=mutation, eliminate_duplicates=duplicates
                )
            else:
                self.algorithm = NSGA2(
                    pop_size=s.population, sampling=sampling, crossover=crossover,
                    mutation=mutation, eliminate_duplicates=duplicates, survival=s.survival
                )

        elif s.algorithm.lower() == "ga":
            self.algorithm = GA(
                pop_size=s.population, sampling=sampling, crossover=crossover,
                mutation=mutation, eliminate_duplicates=duplicates
            )
        if s.termination:
            self.termination = s.termination
        else:
            self.termination = CustomTermination(s.generations)

    def run(self, callback=None, seed=1):
        """The actual running of the algorithm.

        Args:
            callback: Executable object (e.g. function), that is run after each
                generation has finished.
            seed (int): Random number generation seed to yield reproducible
                results for a run.

        Returns:
            The pymoo Result object returned by minimize().
        """

        self.callback = callback
        res = minimize(self.problem, self.algorithm, self.termination, seed=seed,
                verbose=False, save_history=True, callback=self.callback,
                copy_termination=False)

        return res

    def terminate(self):
        """ Terminate the current run by notifying the generation callback """

        self.termination.terminate()
        try:
            # to not e.g. send data or draw plots anymore
            self.callback.terminate()
        except:
            print("Calling self.callback.terminate() failed!")

    def set_n_clusters(self, n):
        """ Set the number of clusters the callback returns.

        Args:
            n (int): Number of clusters
        """

        try:
            self.callback.clusters = n
        except:
            print("Cant set number of clusters in callback!")

