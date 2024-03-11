"""Implementation of the genetic algorithm for optimizing structure designs.

Various classes implementing pymoo interfaces are created to be able to tell
pymoo how to handle the substructure designs, how to mutate them and how to
perform crossover between two different designs.

Author: Anton Kriese, Freie Universitaet Berlin
"""

import bisect
import copy
import time

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, calc_crowding_distance, randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival
from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.misc import find_duplicates, has_feasible
import random


from .graph3d import Connections, SubstructureComplex, SubstructureInterface
from .metrics import REAL_LIFE_LOADS, MetricFunction, TestCollection, total_mass
from .sample import ComplexConstraints, SimpleConstraints, SubstructureConstraints

DEFAULT_PROBABILITIES = {"mutation": 0.1, "crossover": 0.8}

def complete_connection(design: dict, allowed: list[str] = []):
    """ Make sure a given design yields full connection between all nodes.

    This function checks if a design contains enough connections to ensure
    that all nodes are part of the same connected component (fully connected graph).
    If this is not the case, allowed connections are randomly added until the
    'fully connected' condition is met.

    TODO: what to do, if no full connection is possible with the given constraints?

    Args:
        design: A SubstructureComplex dict to be completed.
        allowed: list of strings which connection types are allowed.

    Returns:
        A copied design dict with the completed connections.
    """

    design_copy = copy.deepcopy(design)
    random_access = allowed or list(design_copy["connections"].keys())
    n = len(random_access)
    while True:
        sub = SubstructureComplex.from_dict(design_copy)
        # if fully connected, break loop
        if sub.is_fully_connected():
            break
        if all(design_copy["connections"][k] for k in random_access):
            print("All allowed connections active, but no connectivity possible!")
            print(design_copy)
            break
        design_copy["connections"][random_access[np.random.randint(n)]] = True

    return design_copy

def get_impossible_connections(design: dict) -> list[str]:
    """ Determine connection types that cannot occur with a given design.

    Some connection types should not be allowed for certain design parameters
    of an individual structure. These are determined here to be excluded later.

    Args:
        design: A ComplexSubstructure design dict that is checked for possible
             connection types.

    Returns:
        The list of impossible connections as strings.
    """

    disallowed = []
    if design["legs"] < 4:
        disallowed.extend(["u_opposite", "h_opposite"])
    if len(design["hori_layer_dists"]) == 0:
        disallowed.extend(["u_in_twin", "u_out_twin", "u_in_right", "u_in_left",
            "u_out_right", "u_out_left", "h_twin", "h_out_right", "h_out_left"])

    return disallowed

def remove_disallowed(design: dict, allowed: list[str]):
    """Remove all connection types, that are not allowed.

    Works inplace on the given dict, thus has no return value.

    Args:
        design: A Substructure dict.
        allowed: list of strings resembling the allowed connection types.
    """
    for k in design["connections"].keys():
        if k not in allowed:
            design["connections"][k] = False

def postprocess_design(design: dict, allowed: list[str], constraints: ComplexConstraints):
    """Process a created design so it is viable for further use.

    A set of actions that are performed after a new structure has been built
    either after sampling, crossover or mutation. These actions are:
        1. Enforcing global structre constraints of the GA run.
        2. Remove forbidden connection types.
        3. Add enough connections so that the structure ends up fully connected.

    Args:
        design: A ComplexSubstructure design dict.
        allowed: list of strings resembling the allowed connection types.
        constraints: ComplexConstraints to be forced onto the given design.

    Returns:
        The postprocessed design.
    """

    # enfore constraints
    constraints.enforce_on_design(design)

    # determine which connections are possible for this design
    disallowed = get_impossible_connections(design)
    allowed_connections = [c for c in allowed if c not in disallowed]

    # remove all disallowed connectios
    remove_disallowed(design, allowed_connections)

    # add allowed connections at random if the structure is not fully connected
    design = complete_connection(design, allowed_connections)
    return design

def change_min_max(init, min, max, multiplier=1.0):
    """Calculates a value of change for a design parameter.

    Change a value within constraints at random.

    Args:
        init (number): Initial value of the parameter to be changed.
        min (number): Minimum constraint for the parameter.
        max (number): Maximum constraint for the parameter.
        multiplier (float): Number between 0 and 1 resembling how much
            parameter is allowed to change. 1.0 means the parameter can become
            any value between min and max. 0.1 means the parameter can at most
            change by 10% of the difference between min and max.

    Returns:
        The determined change value for the parameter.
    """

    if min == max:
        return 0

    i = 0
    while True:
        mult = np.random.rand() * (max-min) * multiplier
        change = np.sign(np.random.rand() - 0.5) * mult
        if min <= init+change <= max:
            return change
        i += 1
        if i == 1000:
            raise ValueError(f"""change_min_max() stuck with parameters:
                             {init=}, {min=}, {max=}, {multiplier=}""")


class SubstructureSampler(Sampling):
    """A custom sampling class for the genetic algorithm's seed.

    This class mainly calls the sample_substructure_space() function
    and ensures the viability of the sampled structures.
    """

    def __init__(self, constraints: SubstructureConstraints) -> None:
        """Constructor for the SubstructureSampler class.

        Simply sets the constraints member to know what to sample.

        Args:
            constraints: SubstructureConstraints object to sample from.
        """
        super().__init__()
        self.constraints = constraints
        self.sampled = 0

    def _do(self, _problem, n_samples, **_kwargs):
        """Execute the sampler.

        Overwrite of pymoo's _do() function. The object's constraint
        object is used to get a sample.

        Args:
            _problem: Unused pymoo problem object.
            n_samples (int): Number of samples to return.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array of `n_samples` samples.
        """

        designs = [s.to_dict() | { "id": str(self.sampled+i), "parents": [] }
                   for i, s in enumerate(
                       self.constraints.sample_fully_connected(n_samples))]
        self.sampled += n_samples
        return np.array((designs), dtype=object).reshape((n_samples, 1))

class SubstructureSamplerExtra(Sampling):
    """A custom sampling class for the genetic algorithm's seed.

    This class mainly calls the sample_substructure_space() function
    and ensures the viability of the sampled structures.
    """

    def __init__(self, constraints: SubstructureConstraints) -> None:
        """Constructor for the SubstructureSampler class.

        Simply sets the constraints member to know what to sample.

        Args:
            constraints: SubstructureConstraints object to sample from.
        """
        super().__init__()
        self.constraints = constraints
        self.sampled = 0

    def _do(self, n_samples, **_kwargs):
        """Execute the sampler.

        Overwrite of pymoo's _do() function. The object's constraint
        object is used to get a sample.

        Args:
            _problem: Unused pymoo problem object.
            n_samples (int): Number of samples to return.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array of `n_samples` samples.
        """

        designs = [s.to_dict() | { "id": str(self.sampled+i), "parents": [] }
                   for i, s in enumerate(
                       self.constraints.sample_fully_connected(n_samples))]
        self.sampled += n_samples
        return np.array((designs), dtype=object).reshape((n_samples, 1))
    
class RealSubstructureSampler(Sampling):
    """A custom sampling class for the genetic algorithm's seed.

    This class mainly calls the sample_substructure_space() function
    and ensures the viability of the sampled structures.
    """

    def __init__(self, data: list) -> None:
        """Constructor for the SubstructureSampler class.

        Simply sets the constraints member to know what to sample.

        Args:
            constraints: SubstructureConstraints object to sample from.
        """
        super().__init__()
        self.data = data
        self.sampled = 0

    def _do(self, _problem, n_samples, **_kwargs):
        """Execute the sampler.

        Overwrite of pymoo's _do() function. The object's constraint
        object is used to get a sample.

        Args:
            _problem: Unused pymoo problem object.
            n_samples (int): Number of samples to return.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array of `n_samples` samples.
        """
        # Ensure you sample all available elements at least once
        sampled_elements = random.sample(self.data, min(len(self.data), n_samples))

        # If there are still elements to sample, use random.choices() to potentially sample them again
        if len(sampled_elements) < n_samples:
            additional_samples = random.choices(self.data, k=n_samples - len(sampled_elements))
            sampled_elements.extend(additional_samples)
            
        designs = [s | { "id": str(self.sampled+i), "parents": [] }
                   for i, s in enumerate(sampled_elements)]
        
        self.sampled += n_samples
        return np.array((designs), dtype=object).reshape((n_samples, 1))

class ComplexSubstructureCrossover(Crossover):
    """Crossover implementation for ComplexSubstructure.

    Used to combine two parent ComplexSubstructures during the genetic
    algorithm.
    """

    def __init__(self, constraints: ComplexConstraints):
        """Constructor for the ComplexSubstructureCrossover class.

        We use a crossover of two parents resulting in one child here.

        Args:
            constraints: ComplexConstraints object that is enforced
                on children of the crossover execution.
        """

        self.n_parents = 2
        self.n_children = 1
        self.constraints = constraints
        self.allowed = self.constraints.get_allowed()
        super().__init__(self.n_parents, self.n_children, prob=0.9)

    def _do(self, _problem, X, **_kwargs):
        """Combine the given pairings.

        This creates a design dict for each given crossover pairing.
        A child gets each design parameter from either one of the parents.
        For the connection types a similar system is used, as for each
        possible connection type, either the first or the second parent's
        boolean value is used.
        This can obviously result in invalid designs which have to be
        cleaned up afterwards.

        Args:
            _problem: Unused pymoo problem object.
            X (np.array): A matrix containing the pairings to mate.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array containing all produced children.
        """
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_children, n_matings, n_var), None, dtype=object)
        for k in range(n_matings):
            a, b = copy.deepcopy(X[0, k][0]), copy.deepcopy(X[1, k][0])
            design = {}

            for property in a.keys():
                if property == 'connections':
                    x_conns = {}
                    for conn_key in a[property].keys():
                        if np.random.rand() < 0.5:
                            x_conns[conn_key] = a[property][conn_key]
                        else:
                            x_conns[conn_key] = b[property][conn_key]
                    design['connections'] = x_conns
                    continue
                elif np.random.rand() < 0.5:
                    design[property] = a[property]
                else:
                    design[property] = b[property]

            design = postprocess_design(design, self.allowed, self.constraints)
            Y[0, k] = design
        return Y

class SimpleSubstructureCrossover(Crossover):
    """Crossover implementation for SimpleSubstructure.

    Used to combine two parent SimpleSubstructures during the genetic
    algorithm.
    """

    def __init__(self, constraints: ComplexConstraints, probability=0.8, tracking=False):
        """Constructor for the SimpleSubstructureCrossover class.

        We use a crossover of two parents resulting in one child here.

        Args:
            constraints: SimpleConstraints object that is enforced
                on children of the crossover execution.
            probability (float): value between 0 and 1 resembling how often
                the whole crossover step is actually performed.
            tracking (bool): If true, the ids and parents' ids are tracked.
        """

        self.n_parents = 2
        self.n_children = 1
        self.constraints = constraints
        self.tracking = tracking
        super().__init__(self.n_parents, self.n_children, prob=probability)
        self.total_ids = 0

    def _do(self, _problem, X, **_kwargs):
        """Combine the given pairings.

        This creates a design dict for each given crossover pairing.
        A child gets each design parameter from either one of the parents.
        For the connection types a similar system is used, as for each
        possible connection type, either the first or the second parent's
        boolean value is used.
        This can obviously result in invalid designs which have to be
        cleaned up afterwards.

        Additionally, the id and parents of a child can be saved in the
        design dicts. This can be used to recreate a genealogical tree.

        For the layer heights, the parents' layer heights are combined
        and the respective amount of layer heights are drawn at random.

        Args:
            _problem: Unused pymoo problem object.
            X (np.array): A matrix containing the pairings to mate.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array containing all produced children.
        """

        _, n_matings, n_var = X.shape
        Y = np.full((self.n_children, n_matings, n_var), None, dtype=object)
        for k in range(n_matings):
            a, b = copy.deepcopy(X[0, k][0]), copy.deepcopy(X[1, k][0])
            design = {}
            self.total_ids += 1
            design["id"] = "c" + str(self.total_ids)
            if self.tracking:
                design["parents"] = [a, b]
            for property in a.keys():
                if property in ["layer_heights", "connection_types", "id", "parents"]:
                    continue

                if np.random.rand() < 0.5:
                    design[property] = a[property]
                else:
                    design[property] = b[property]

            # sample layer_heights from both designs
            layers = design["n_layers"]
            total_height = design["total_height"]
            all_heights = [x for x in set(a["layer_heights"] + b["layer_heights"]) if x < total_height]
            if len(all_heights) < layers-2:
                all_heights.extend((np.random.rand(layers) * total_height).tolist())
            design["layer_heights"] = sorted(
                    np.random.choice(all_heights, layers-2, replace=False).tolist()
            )

            # sample conntection types from both designs
            all_c_types = list(set(a["connection_types"] + b["connection_types"]))
            design["connection_types"] = np.random.choice(all_c_types, layers-1).tolist()

            # switch bottom and top radius if top is bigger
            if (b:=design["radius_bottom"]) < (t:=design["radius_top"]):
                design["radius_bottom"], design["radius_top"] = t, b

            Y[0, k] = design
        return Y

# TODO: Remove all hardcoded values and make them either constants or parameters
class ComplexSubstructureMutation(Mutation):
    """The custom Mutation implementation for ComplexSubstructure.

    For each value of the design, the value is changed within its constraints
    at a certain random chance.
    """

    def __init__(self, constraints: ComplexConstraints) -> None:
        """Constructor for the ComplexSubstructureMutation class.

        Args:
            constraints: ComplexConstraints object enforced on mutated designs.
        """

        super().__init__()
        self.constraints = constraints
        # precompute the allowed connections, as they are constantly the same
        # (not as disallowed, which depend on the individual design parameters)
        self.allowed = self.constraints.get_allowed()

    def _do(self, _problem, X, **_kwargs):
        """Execute the mutation process on a given set of designs.

        Each given design is mutated by fixed rules.
        Every parameter is changed with a chance of a certain percentage.
        Each parameter can have a different probability of change.
        Parameters that depend on each other (e.g. hori_layer_dists
        and h_layers) are mutated respecting each other.

        total_height is currently not mutated.

        Args:
            _problem: Unused pymoo problem object.
            X (np.array): Array of designs to mutate.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array containing the mutated designs
        """

        Y = np.full_like(X, None, dtype=object)
        for i, design_ref in enumerate(X):
            # not doing a deep copy led to a reference bug before
            design = copy.deepcopy(design_ref[0])

            # mutate number of h_layers in 0.1 of cases
            h_layer_dists = design['hori_layer_dists']
            h_layers = len(h_layer_dists) + 1
            if np.random.rand() < 0.1:
                change = change_min_max(h_layers, *self.constraints.h_layers)

                if change == 1:
                    min, max = self.constraints.h_layer_dist
                    h_layer_dists.append(np.random.rand()*(max-min)+min)
                else:
                    h_layer_dists = h_layer_dists[1:] # pop first item

                h_layers = int(h_layers+change)

            # mutate h_layer_dists in 0.2 of cases
            if h_layers > 1 and np.random.rand() < 0.2:
                min, max = self.constraints.h_layer_dist
                h_layer_dists = np.array(h_layer_dists) + (np.random.rand(h_layers-1) - 0.5) * (max-min)/10
                h_layer_dists = np.minimum(np.maximum(h_layer_dists, min), max).tolist()
            design['hori_layer_dists'] = h_layer_dists

            # mutate legs in 0.3 of cases
            if np.random.rand() < 0.3:
                change = int(change_min_max(design['legs'], *self.constraints.legs))
                design['legs'] += change

            # mutate n_layers in 0.1 of cases
            if np.random.rand() < 0.1:
                change = int(change_min_max(design['n_layers'], *self.constraints.layers))
                design['n_layers'] += change

            # mutate init_radius in 0.2 of cases
            if np.random.rand() < 0.2:
                change = change_min_max(design['init_radius'],
                        *self.constraints.radius, 0.3)
                design['init_radius'] += change

            # mutate radius_evolution in 0.3 of cases
            if np.random.rand() < 0.3:
                change = change_min_max(design['radius_evolution'],
                        *self.constraints.radius_evolution, 0.1)
                design['radius_evolution'] += change

            # mutate layer_dist_evolution in 0.3 of cases
            if np.random.rand() < 0.3:
                change = change_min_max(design['layer_dist_evolution'],
                        *self.constraints.layer_dist_evolution, 0.1)
                design['layer_dist_evolution'] += change

            # mutate rotation in 0.2 of cases
            if np.random.rand() < 0.2:
                change = change_min_max(design['rotation'],
                        *self.constraints.rotation, 0.4)
                design['rotation'] += change

            # TODO should we mutate the total_height?

            # mutate connections in 0.05 of cases
            for k, v in design['connections'].items():
                if np.random.rand() < 0.05:
                    design['connections'][k] = not v

            design = postprocess_design(design, self.allowed, self.constraints)
            Y[i] = design
        return Y

class SimpleSubstructureMutation(Mutation):
    """The custom Mutation implementation for SimpleSubstructure.

    For each value of the design, the value is changed within its constraints
    at a certain random chance.
    """

    def __init__(self, constraints: SimpleConstraints, probability=0.1, tracking=False) -> None:
        """Constructor for the SimpleSubstructureMutation class.

        Args:
            constraints: SimpleConstraints object enforced on mutated designs.
            probability (float): value between 0 and 1 resembling how often
                a mutation is actually performed.
            tracking (bool): If true, the ids and parents' ids are tracked.
        """
        super().__init__()
        self.constraints = constraints
        self.probability = probability
        self.mutatables = ["n_layers", "layer_heights", "connection_types", "legs",
                           "radius_bottom", "radius_top", "total_height"]
        self.total_ids = 0
        self.tracking = tracking

    def _do(self, problem, X, **kwargs):
        """Execute the mutation process on a given set of designs.

        Each mutation process only allows for one parameter to be changed.
        This is a difference to the behavior of ComplexSubstructureMutation.
        Parameters that depend on each other are mutated respecting each other.
        E.g. if `n_layers` changed, `layer_heights` gets a new height or
        loses one depending on the `n_layers` change.

        Args:
            _problem: Unused pymoo problem object.
            X (np.array): Array of designs to mutate.
            **_kwargs: Unused additional arguments.

        Returns:
            A numpy array containing the mutated designs.
        """

        Y = np.full_like(X, None, dtype=object)
        connection_options = [s.value["name"] for s in self.constraints.allowed_connections]
        for d_index, design_ref in enumerate(X):
            # not doing a deep copy led to a reference bug before
            design = copy.deepcopy(design_ref[0])

            # dont mutate at all if mutation probability space is not hit
            if np.random.rand() > self.probability:
                Y[d_index] = design
                continue

            self.total_ids += 1
            design["id"] = "m" + str(self.total_ids)
            if self.tracking:
                design["parents"] = [design_ref[0]]

            # choose what attribute to mutate
            mutate = np.random.choice(self.mutatables)

            # mutate number of layers
            if mutate == "total_height":
                total_height = design["total_height"]
                min, max = self.constraints.total_height[0], self.constraints.total_height[1]
                new_height = np.random.rand()*(max-min)+min
                design["total_height"] = float(new_height)

                heights = design["layer_heights"]
                # also scale layer heights accordingly
                if len(heights):
                    new_heights = [height / total_height * new_height for height in heights]
                    design["layer_heights"] = new_heights

            if mutate == "n_layers":
                change = int(change_min_max(design["n_layers"], *self.constraints.n_layers))
                design["n_layers"] += change
                if change >= 1:
                    heights = design["layer_heights"]
                    # add a new layer
                    for _ in range(change):
                        new_layer_height = -1
                        while True:
                            new_layer_height = float(np.random.rand() * design["total_height"])
                            if new_layer_height not in heights:
                                # heights.append(new_layer_height)
                                break

                        bisect.insort(heights, new_layer_height)
                        insert_idx = heights.index(new_layer_height)

                        # add a new brace for the new layer
                        design["connection_types"].insert(insert_idx, np.random.choice(connection_options))
                elif change < 0:
                    # delete a random layer and its brace
                    for _ in range(-change):
                        layer_idx = np.random.randint(len(design["layer_heights"]))
                        del design["layer_heights"][layer_idx]
                        del design["connection_types"][layer_idx]

            elif mutate == "legs":
                options = list(range(self.constraints.legs[0], self.constraints.legs[1]+1))
                options.remove(design["legs"])
                if len(options) != 0:
                    design["legs"] = int(np.random.choice(options))
            # TODO mutate layer heights too
            elif mutate == "layer_heights":
                layer_heights = [0] + design["layer_heights"] + [design["total_height"]]
                if len(layer_heights) == 2:
                    pass
                else:
                    # mutate a height just between neighboring heights
                    idx = np.random.randint(len(layer_heights)-2) + 1
                    min, max = layer_heights[idx-1], layer_heights[idx+1]

                    # move layer between layers above and below
                    new_height = np.random.rand() * (max - min) + min
                    design["layer_heights"][idx-1] = new_height
            elif mutate == "connection_types":
                idx = np.random.randint(len(design["connection_types"]))
                design["connection_types"][idx] = np.random.choice(connection_options)
            elif mutate.startswith("radius"):
                radius_bottom = design["radius_bottom"]
                radius_top = design["radius_top"]

                if mutate == "radius_bottom":
                    min, max = radius_top, self.constraints.radius_bottom[1]
                    change_bottom = change_min_max(radius_bottom, min, max, 0.4)
                    new_bottom = radius_bottom + change_bottom
                    design["radius_bottom"] = float(new_bottom)

                if mutate == "radius_top":
                    min, max = self.constraints.radius_top[0], radius_bottom
                    change_top = change_min_max(radius_top, min, max, 0.4)
                    new_top = radius_top + change_top
                    design["radius_top"] = float(new_top)

            Y[d_index] = design
        return Y


class SubstructureDupElim(ElementwiseDuplicateElimination):
    """ Duplicate elimination implementation for Substructure objects """

    def is_equal(self, a, b):
        """ Simple duplicate elimination.

        It compares the connections first and then the rest of the design
        parameters.

        This handles both the complex and simple architectures.

        TODO: Dont evaluate something like close init_radius as False.
        E.g. init_radius=211 vs 213 isnt too much of a difference if
        the rest of the parameters are all the same.

        Args:
            a (pymoo individual): First individual to be compared.
            b (pymoo individual): Second individual to be compared.

        Returns:
            Boolean value representing if `a` and `b` are duplicates.
        """

        da, db = a.X[0], b.X[0]

        if isinstance(SubstructureInterface.from_dict(da), SubstructureComplex):
            if Connections.from_dict(da) != Connections.from_dict(db):
                return False

            for key in da.keys():
                if key == "connections":
                    continue
                if da[key] != db[key]:
                    return False
        else: # easy comparison between SubstructureSimple dicts
            return da == db
        return True

class SubstructureProblem(ElementwiseProblem):
    """ Problem implementation for substructures.

    Here, the evaluation of generated designs happens.
    The initialization takes a MetricFunction object, which can contain
    mutliple objective functions.
    Each of these functions are then executed with the Substructure object
    passed to them.

    As the genetic algorithm optimizes towards the minimum values, the evaluation
    values are turned, when `invert_goals` is True (to prioritize bad structures).
    """

    def __init__(self, objective_function: MetricFunction, invert_goals: bool=False, objective_thresholds: bool=False):
        """Constructor of the SubstructureProblem class.

        Args:
            objective_function: Object that returns performance values upon
                calling it with a substructure object as argument.
            invert_goals: If True, inverts the goals leading to "optimization"
                bad designs.
        """
        self.n_ieq_constr=0
        self.objective_thresholds = objective_thresholds
        self.invert_goals = invert_goals
        self.objective_function = objective_function
        if self.objective_thresholds:
            self.n_ieq_constr=len(self.objective_thresholds)
        self.metric_names = self.objective_function.metric_names
        

        super().__init__(n_var=1, n_obj=objective_function.n_objectives, n_constr=0, n_ieq_constr=self.n_ieq_constr)


        if self.objective_thresholds:
            for k, v in self.objective_thresholds.items():
                print(self.metric_names)
                idx = self.metric_names.index(k)
                print("Creating threshold for objective '{}' with index '{}' to be <= {}".format(self.metric_names[idx], idx, v))

    def _evaluate(self, design, out, *args, **kwargs):
        """Evaluate a given design.

        Args:
            design (pymoo individual): The design to be tested.
            out: A pymoo object containing test values and values to plot.
            *args: Unused.
            **kwargs: Unused.
        """

        sub = SubstructureInterface.from_dict(design[0])
        # give a penalty for structures with no connections

        objectives, values_for_plotting = self.objective_function(sub)

        if self.invert_goals:
            objectives = [-val for val in objectives]

        out['F'] = np.array(objectives)
        # we set this unused key in the result dict to the values needed for plotting
        out['plot_values'] = np.array(values_for_plotting)

        if self.objective_thresholds:
            constraints = []
            for k, v in self.objective_thresholds.items():
                idx = self.metric_names.index(k)
                constr = objectives[idx] - v
                constraints.append(constr)
            out["G"] = np.array(constraints)

class PlotCallback(Callback):
    """A pymoo Callback for plotting live values of a running algorithm.

    Plots each generation's objective values to a scatter plot.
    obj_indices decides which objective values should been
    displayed on the x and y axes of the scatter plot.

    Attributes:
        pop_size: Population size of the algorithm run.
        n_gen: Number of generations of the current run.
        idcs: Tuple of indices that specify which objectives are shown.
        scatter: plt.scatter object.
        values: Values of the x and y axes.
        intensities: Array that holds the dot indices.
            Older generations fade out.
        last_time: Time saved after each generation to register running time.
    """

    def __init__(self, pop_size, n_gen, obj_indices=(0, 1)) -> None:
        super().__init__()
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.figure, self.axes = plt.subplots()

        assert len(obj_indices) == 2, "Give exactly two objectives indices (for x and y)!"
        self.idcs = np.array(obj_indices)

        # create a custom color map for faded dots
        colors = [[0,0,1,0], [0,0,1,0.5], [0,0.2,0.4,1]]
        cmap = LinearSegmentedColormap.from_list("", colors)
        self.scatter = self.axes.scatter([], [], c=[], cmap=cmap)

        # we know the amount of data points at the start of the run
        entries = self.pop_size * self.n_gen
        self.values = np.zeros((entries, 2))
        self.intensities = np.zeros(entries)

        self.last_time = time.time()

    def notify(self, algorithm, **kwargs):
        this_gen = algorithm.n_gen
        # determine how much time the last generation took
        now = time.time()
        last_gen_time = now - self.last_time
        self.last_time = now

        insert_start, insert_end = (this_gen-1)*self.pop_size, this_gen*self.pop_size

        # add new evaluation points to the plot
        self.values[insert_start:insert_end] = algorithm.pop.get("plot_values")[:, self.idcs]
        self.scatter.set_offsets(self.values)

        # fade out older dots in the plot
        self.intensities *= 0.8
        self.intensities[insert_start:insert_end] = 1
        self.scatter.set_array(self.intensities)

        # adjust axis limits to new data
        self.axes.set_xlim(0, np.max(self.values[:, 0]))
        self.axes.set_ylim(0, np.max(self.values[:, 1]))

        self.axes.set_title(f"Objective Space after Generation {this_gen} (took {last_gen_time:.2f} sec.)")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

def calc_biased_crowding_distance(F, filter_out_duplicates=True, index_important=1, index_scaling=10):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # _F[:, index_important] *= index_scaling  #ADDED LINE

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        weights = np.ones(n_obj)
        weights[index_important] = 1/index_scaling 

        # Normalize the weights so they add up to 1
        weights = weights / np.sum(weights)
        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = weights * dist_to_last[:-1] / norm, weights * dist_to_next[1:] / norm
        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj
        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    # crowding[np.isinf(crowding)] = 1e+14
    return crowding

class WeightedRankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None, index=1, scaling=0.1) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.index = index
        self.scaling = scaling

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_biased_crowding_distance(F[front, :], index_important=self.index, index_scaling=self.scaling)

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                # print("Front {} of length {} is split".format(k, len(front)))
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]
                # print("Splitted has length", len(I))

            # otherwise take the whole front unsorted
            else:
                # print("Front {} of length {} is taken entirely".format(k, len(front)))
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]
    

class BiasedPlausibleRankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None, threshold=-1, index=1, constraints=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.threshold = threshold
        self.index = index
        self.sampler = SubstructureSamplerExtra(constraints)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        pass_threshold_from_front = np.where(F[:, self.index] <= self.threshold)
        filtered_F = F[pass_threshold_from_front]
        new_random = self.sampler._do(n_samples=len(F)-len(filtered_F))
        
        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # indices_from_front = front[I]
            # results_from_front = F[indices_from_front, :]
            # # print(results_from_front.shape)
            # # print(front.shape)
            # pass_threhsold_from_front = np.where(results_from_front[:, self.index] <= self.threshold)

            # surviving_from_front = front[pass_threhsold_from_front]

            surviving_from_front = front[I]
            # print(surviving_from_front.shape)
            # extend the survivors by all or selected individuals
            survivors.extend(surviving_from_front)
        surviving = pop[survivors]
        # surviving.extend(new_random)
        print(len(surviving))
        print(len(pop))
        # exit(1)
        return surviving

class RankAndCrowdingSurvival(Survival):

    def __init__(self, nds=None) -> None:
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []
        print(F)
        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        print("Number of fronts", len(fronts))
        print(fronts)
        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]
    
class PrintCallback(Callback):
    """A pymoo Callback for printing the generation
    """

    def __init__(self) -> None:
        super().__init__()

    def notify(self, algorithm, **kwargs):
        this_gen = algorithm.n_gen
        print("Generation", this_gen)

class KeepGenerationsCallback(Callback):
    """A pymoo Callback for printing the generation
    """

    def __init__(self, print_objectives=None) -> None:
        super().__init__()
        self.generation_instances = []
        self.print_objectives = print_objectives

    def notify(self, algorithm, **kwargs):
        instances = [x.X[0] for x in algorithm.pop]
        objectives = np.array([x.F for x in algorithm.pop])
        self.generation_instances.append(instances)
        this_gen = algorithm.n_gen
        print("Generation {} number of instances {}".format(this_gen, len(instances)))
        if self.print_objectives:
            print(np.max(objectives, axis=0))

def run_algorithm(pop_size: int=20, n_gen: int=100,
                  loads: dict[str, float]=REAL_LIFE_LOADS,
                  invert_goals: bool=False, complex: bool=False,
                  probabilities: dict=DEFAULT_PROBABILITIES):
    """Run the genetic algorithm with some defaults.

    Prepares constraints, Problem, Crossover, Mutation and Algorithm objects.
    Runs the algortihm at the end.

    Args:
        pop_size: Population size of the run.
        n_gen: Number of generations to run.
        loads: Load dict for the structural tests to be performed.
        invert_goals: If True, inverts the goals leading to "optimization".
        complex: Determines if Simple or Complex substructures are used.
        probabilities: Dict containing crossover and mutation probabilities.

    Returns:
        The pymoo.Result and the problem object from running the algorithm.
    """

    constraints = ComplexConstraints() if complex else SimpleConstraints()
    objectives = MetricFunction(
        total_mass,
        TestCollection(loads)
    )
    problem = SubstructureProblem(objectives, invert_goals)

    if complex:
        crossover = ComplexSubstructureCrossover(constraints)
        mutation = ComplexSubstructureMutation(constraints)
    else:
        crossover = SimpleSubstructureCrossover(constraints, probabilities["crossover"])
        mutation = SimpleSubstructureMutation(constraints, probabilities["mutation"])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SubstructureSampler(constraints),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=SubstructureDupElim()
    )

    res = minimize(problem, algorithm, ('n_gen', n_gen), seed=1, verbose=False,
                    save_history=True, callback=PlotCallback(pop_size, n_gen))
    return res, problem

if __name__ == '__main__':
    result = run_algorithm()

