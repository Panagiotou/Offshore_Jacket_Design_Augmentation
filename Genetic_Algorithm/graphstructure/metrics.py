import copy
import numpy as np
from pathlib import Path

from .graph3d import Edge, Material, Node, SubstructureComplex, SubstructureInterface
from .morison_equation import calculate_maximum_forces
from .opensees.model import sub_to_ops, add_edges, add_nodes
from .opensees.structural_test import Collection, MultiNodeTest, StructuralTest
from .simple_substructure import ConnectionBraces, SubstructureSimple

from sklearn.preprocessing import LabelEncoder 
from sklearn.mixture import GaussianMixture

from copy import deepcopy
from light_famd.famd import FAMD as LIGHT_FAMD
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.svm import OneClassSVM
from statsmodels.nonparametric.kernel_density import KDEMultivariate
# from isolation_forest.learn.isolation_forest import IsolationForest 

import sys
sys.path.append("../util")

DISCONNECTED_STABILTY = 100

# Loads in N and Nm (torque) that resemble real life forces on substructures
REAL_LIFE_LOADS = {
    "compression": 6.835e06,
    "tipover": 1.664e04,
    "torque": 1e05
}

def total_mass(sub: SubstructureInterface):
    """ A metric function to estimate the cost of a design.

    Args:
        sub: Design to be measured.

    Returns:
        Sum of all edges of the structure.
    """

    return sum([np.linalg.norm(a - b) for (a, b, _) in sub.edges])

def load_test_to_top(sub: SubstructureInterface, load):
    """ Sets up a test with a load on the top of a structure.

    If the structure does not have a top center node, one will be created with
    infinitely strong connections to the outer nodes.
    In any case, the design will be converted to the opensees model within this
    function.

    Args:
        load (list[int]): The load with 6 values for the df.
        sub: Design to be tested.

    Returns:
        A test that applies force on the top center node.
    """

    if isinstance(sub, SubstructureComplex):
        node = prepare_top_center_complex(sub)
    else: #isinstance(sub, SubstructureSimple):
        node = prepare_top_center_simple(sub)

    sub_to_ops(sub, rotation=np.pi/4)
    return MultiNodeTest([node.idx], load)

def compression_test(sub: SubstructureInterface, load: float, steps=1, execute=True):
    """ Stress test with a vertical load to the top center of the given design.

    This simulates the weight of the windturbine tower pressing downwards.

    Args:
        steps (int): Analysis steps.
        execute (bool): If true, runs the test before returning the object.
        sub: The design to be tested.
        load: Force value in N that is applied on the z-axis towards the ground.

    Returns:
        A test object and the substructure, which might have been changed slightly for the test.
    """

    test = load_test_to_top(sub, [0., 0., -abs(load), 0., 0., 0.])

    if execute:
        test(steps)
        return test
    return test, sub

def tipover_test(sub: SubstructureInterface, load: float, steps=1, execute=True):
    """ Stress test with a horizontal load to the top center of the given design.

    This simulates the pressure of the wind on the windturbine tower.

    Args:
        steps (int): Analysis steps.
        execute (bool): If true, runs the test before returning the object.
        sub: The design to be tested.
        load: Force value in N that is applied on the y-axis.

    Returns:
        A test object and the substructure, which might have been changed slightly for the test.
    """

    test = load_test_to_top(sub, [0., load, 0., 0., 0., 0.])

    if execute:
        test(steps)
        return test
    return test, sub

def torque_test(sub: SubstructureInterface, load: float, steps=1, execute=True):
    """ Stress test with a rotational load to the top center of the given design.

    Args:
        steps (int): Analysis steps.
        execute (bool): If true, runs the test before returning the object.
        sub: The design to be tested.
        load: Moment value in Nm that is applied to the z-rotational value.

    Returns:
        A test object and the substructure, which might have been changed slightly for the test.
    """

    # applying the load on the 6th degree of freedom (z-rotational load)
    test = load_test_to_top(sub, [0., 0., 0., 0., 0., float(load)])

    if execute:
        test(steps)
        return test
    return test, sub


def wave_test(sub: SubstructureInterface, steps=1, execute=True,
              relative_water_depth=0.57136, wave_height=1.1909, wave_period=6.59):
    """Applies the force of a wave on a structure.

    This maximum force is calculated with the Morison equation.
    The force is only applied at the structure height of the biggest force.
    The main legs are modified, so that the elements are interpolated at the
    right height. These inserted nodes are then loaded with the force from
    the Morison equation.

    Args:
        steps (int): Analysis steps.
        execute (bool): Decides if the test is executed before returning.
        relative_water_depth (float): Ratio between water depth and total height
            of the design.
        wave_height (float): Height of a wave in m.
        wave_period (float): Time between two waves.
        sub: Design to be tested.

    Returns:
        A test object and the modified substructure.
    """

    water_depth = relative_water_depth*sub.total_height
    ftmax, heights = calculate_maximum_forces(T=wave_period, d=water_depth, H=wave_height, D=1.2)
    max_impact_idx = np.argmax(ftmax)
    impact_measure_height = water_depth + heights[max_impact_idx]

    # we need a copy here to be able to insert interpolated nodes without changing the passed Substructure
    sub_copy = copy.deepcopy(sub)
    node_offset = len(sub_copy.nodes)
    loaded_nodes = []
    # insert artifical nodes at measure_height
    for lower, upper in zip(sub_copy.layers[:-1], sub_copy.layers[1:]):
        if not lower.height < impact_measure_height < upper.height:
            continue

        percent_between = (impact_measure_height - lower.height) / (upper.height - lower.height)

        # here we assume, that the first x nodes are the main legs nodes
        for i in range(2): # only use first two legs
            l, u = lower.nodes[i], upper.nodes[i]
            matching_edges = [e for e in sub_copy.edges if e.start == l and e.end == u]
            if len(matching_edges) == 1:
                sub_copy.edges.remove(matching_edges[0])

            offset = (u - l) * percent_between

            interpolated = Node(l.x + offset[0], l.y + offset[1], impact_measure_height, idx=node_offset)
            loaded_nodes.append(node_offset)
            node_offset += 1
            sub_copy.nodes.append(interpolated)
            sub_copy.edges.append(Edge(l, interpolated, Material.Strong))
            sub_copy.edges.append(Edge(interpolated, u, Material.Strong))

        break

    sub_to_ops(sub_copy, rotation=45)

    test = MultiNodeTest(loaded_nodes, [0,-len(loaded_nodes)*max(ftmax),0,0,0,0])
    if execute:
        test(steps)
        return test
    return test, sub_copy

def combined_test(sub: SubstructureInterface, load_dict: dict, steps = 1):
    """ Combines four structural tests at the same time.

    This is realized by creating the four tests in series without running them
    during the creation. Just afterwards, the nodes and loads are combined to
    execute the test.

    Args:
        steps (int): Analysis steps.
        sub: Design to be tested.
        load_dict: Dict containing the loads for the different tests.
            See REAL_LIFE_LOADS as an example.

    Returns:
        A test object.
    """

    # sub_to_ops(sub, rotation=45)
    sub = copy.deepcopy(sub)
    # as we add a top center node here, we need to make sure, that we are operating on the same substructure
    comp_test, sub = compression_test(sub, load=load_dict["compression"], execute=False)
    push_test, sub = tipover_test(sub, load=load_dict["tipover"], execute=False)
    torq_test, sub = torque_test(sub, load=load_dict["torque"], execute=False)
    wav_test, sub = wave_test(sub, execute=False)

    sub_to_ops(sub)

    loads, node_groups = zip(*[(t.load, t.nodes) for t in [comp_test, push_test, torq_test, wav_test]])
    test = Collection(loads=loads, node_groups=node_groups)
    test(steps=steps)
    return test


def prepare_top_center_complex(sub: SubstructureComplex) -> Node:
    """ Adds a top center node to the given substructure if none exists yet.

    Also adds infinitely strong edges from that center node to the inner layers'
    nodes.
    WARNING! This method work directly on the opensees model. It won't work if
    one wants to convert to opensees AFTER this function call.
    TODO: Fix this behaviour.

    Args:
        sub: The substructure to be tested.

    Returns:
        The top center node (created or already existing).
    """

    # determine top nodes
    legs = sub.legs
    nodes = sub.layers[-1].nodes

    # if the structure already has a center node, we dont have to add one
    if sub.has_center_node:
        node = nodes[0]
        offset = 1
    else:
        z = nodes[0].z
        offset = 0
        node = Node(0., 0., float(z), idx=len(sub.nodes))
        add_nodes([node])

    # make sure, that the middle node is connected to all inner layer nodes
    new_edges: list[Edge] = []
    for inner_layer_node in nodes[offset:legs+offset]:
        if inner_layer_node.idx < node.idx:
            new_edge = Edge(inner_layer_node, node, Material.Inf)
        else:
            new_edge = Edge(node, inner_layer_node, Material.Inf)

        if new_edge not in sub.edges: # TODO take care of possible reference problem
            new_edges.append(new_edge)
    add_edges(new_edges, offset=len(sub.edges))

    return node

def prepare_top_center_simple(sub: SubstructureSimple) -> Node:
    """ Adds a top center node to the given substructure.

    Also adds infinitely strong edges from that center node to the inner layers'
    nodes.

    Args:
        sub: The substructure to be tested.

    Returns:
        The top center node (created or already existing).
    """

    nodes = [n for n in sub.nodes if n.z == sub.total_height]

    coords = np.array([0, 0, sub.total_height])
    try:
        idx = [np.allclose(n.as_array(), coords) for n in sub.nodes].index(True)
        return sub.nodes[idx]
    except:
        # do nothing if index() fails, as this is expected
        # (in case the center node doesnt exist yet)
        pass

    center_node = Node(*coords, idx=len(sub.nodes))

    sub.nodes.append(center_node)
    for node in nodes:
        sub.edges.append(Edge(node, center_node, Material.Inf))

    return center_node


def get_max_deformation(test_fun, **kwargs):
    """
        Test deformation delegate function.
        kwargs should include all necessary arguments for the test function.
    """
    test: StructuralTest = test_fun(**kwargs)
    return np.abs(test.stretch_percentages).max()

def get_max_stress(test_fun, **kwargs):
    """
        Test stress delegate function.
        kwargs should include all necessary arguments for the test function.
    """
    test: StructuralTest = test_fun(**kwargs)
    return test.max_bottom_stress()

class TestCollection():
    """
        This class wraps different tests and their combinations.
        The __call__ function only takes the substructure to be tested and returns
        the sum of instability values.
    """

    def __init__(self, objective_dict) -> None:
        self.compression_load = objective_dict.get("compression", -1)
        self.tipover_load = objective_dict.get("tipover", -1)
        self.torque_load = objective_dict.get("torque", -1)

    def __call__(self, sub: SubstructureInterface):
        fully_connected = sub.is_fully_connected()
        result = 0

        if fully_connected:
            if self.compression_load != -1:
                compression = compression_test(sub, self.compression_load)
                result += np.abs(compression.stretch_percentages).max()
            if self.tipover_load != -1:
                tipover = tipover_test(sub, self.tipover_load)
                result += np.abs(tipover.stretch_percentages).max()
            if self.torque_load != -1:
                torque = torque_test(sub, self.torque_load)
                result += np.abs(torque.stretch_percentages).max()
        else:
            result = DISCONNECTED_STABILTY

        return result

from .sample import SimpleConstraints
import pandas as pd
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import json
from sklearn.svm import SVC
from .plausibility_metrics import *


def plausibility_svm(sub: SubstructureInterface, svm):
    dct = sub.to_dict()
    svm.instance_df.loc[0] = svm.encode(dct)
    transformed_encoded = svm.ct.transform(svm.instance_df)
    probability = -1*svm.clf.decision_function(transformed_encoded)[0]
    return probability

def plausibility_svm_one_class(sub: SubstructureInterface, svm):
    dct = sub.to_dict()
    svm.instance_df.loc[0] = svm.encode(dct)
    transformed_encoded = svm.ct.transform(svm.instance_df)
    probability = -1*svm.clf.decision_function(transformed_encoded)[0]
    return probability

def plausibility_svm_one_class_famd(sub: SubstructureInterface, svm):
    dct = sub.to_dict()
    transformed_encoded = svm.famd.transform(dct)
    probability = -1*svm.clf.decision_function(transformed_encoded)[0]

    return probability

def plausibility_famd_mahalanobis(sub: SubstructureInterface, famd):
    dct = sub.to_dict()
    dist = famd.mahalanobis(dct)
    return dist

def plausibility_famd_gmm(sub: SubstructureInterface, famd):
    dct = sub.to_dict()
    log_likelihood = famd.gmm_log_likelihood(dct)
    return log_likelihood

def plausibility_kde(sub: SubstructureInterface, kde):
    dct = sub.to_dict()
    kde.instance_df.loc[0] = kde.encode(dct)
    transformed_encoded = kde.ct.transform(kde.instance_df)
    neg_log_like = -1*kde.clf.score_samples(transformed_encoded)[0]
    return neg_log_like

def plausibility_isolation_forest(sub: SubstructureInterface, mif):
    dct = sub.to_dict()
    mif.instance_df.loc[0] = mif.encode(dct)
    transformed_encoded = mif.ct.transform(mif.instance_df)
    neg_anomaly_score = -1*mif.clf.score_samples(transformed_encoded)[0]
    return neg_anomaly_score

def plausibility_local_outlier_factor(sub: SubstructureInterface, lof):
    dct = sub.to_dict()
    lof.instance_df.loc[0] = lof.encode(dct)
    transformed_encoded = lof.ct.transform(lof.instance_df)
    neg_anomaly_score = -1*lof.clf.score_samples(transformed_encoded)[0]
    return neg_anomaly_score


def get_plausibility_metric_name(metric_name, gamma, nu, use_specs=True):
    name = None
    if metric_name.lower() == "famd mahalanobis":
        name = "plausibility (FAMD Mahalanobis)"
    elif metric_name.lower() == "famd gmm":
        name = "plausibility (FAMD GMM)"
    elif metric_name.lower() == "svm":
        name = "plausibility (SVM)"
    elif metric_name.lower() == "one class svm":
        if use_specs:
            name = "plausibility (One Class SVM) gamma={} nu={}".format(gamma, nu)
        else:
            name = "plausibility (One Class SVM)"
    elif metric_name.lower() == "famd one class svm":
        if use_specs:
            name = "plausibility (FAMD One Class SVM) gamma={} nu={}".format(gamma, nu)
        else:
            name = "plausibility (FAMD One Class SVM)"
    elif metric_name.lower() == "kde":
        name = "plausibility (KDE)"
    elif metric_name.lower() == "isolation forest":
        name = "plausibility (Isolation Forest)"
    elif metric_name.lower() == "local outlier factor":
        name = "plausibility (local outlier factor)"
    elif metric_name.lower() == "none":
        return None
    else: 
        print(metric_name.lower(), "Plausibility metric not defined")
        exit(1)
    return name

def get_plausibility_metric(metric_name, n_samples, gamma, nu, use_specs=True):
    func = None
    name = None
    if metric_name == "famd mahalanobis":
        famd = Famd()
        func = lambda x: plausibility_famd_mahalanobis(x, famd)
        name = "plausibility (FAMD Mahalanobis)"
    elif metric_name == "famd gmm":
        famd = Famd()
        func = lambda x: plausibility_famd_gmm(x, famd)
        name = "plausibility (FAMD GMM)"
    elif metric_name == "svm":
        discriminator = Discriminator(n_samples=n_samples)
        func = lambda x: plausibility_svm(x, discriminator)
        name = "plausibility (SVM)"
    elif metric_name == "one class svm":
        discriminator_one_class = DiscriminatorOneClass(gamma=gamma, nu=nu)
        func = lambda x: plausibility_svm_one_class(x, discriminator_one_class)
        if use_specs:
            name = "plausibility (One Class SVM) gamma={} nu={}".format(gamma, nu)
        else:
            name = "plausibility (One Class SVM)"
    elif metric_name == "famd one class svm":
        discriminator_one_class_famd = DiscriminatorOneClassFAMD(gamma=gamma, nu=nu)
        func = lambda x: plausibility_svm_one_class_famd(x, discriminator_one_class_famd)
        if use_specs:
            name = "plausibility (FAMD One Class SVM) gamma={} nu={}".format(gamma, nu)
        else:
            name = "plausibility (FAMD One Class SVM)"
    elif metric_name == "kde":
        discriminator_kde = ClassKDE()
        func = lambda x: plausibility_kde(x, discriminator_kde)
        name = "plausibility (KDE)"
    elif metric_name == "isolation forest":
        discriminator_mif = ClassIsolationForest()
        func = lambda x: plausibility_isolation_forest(x, discriminator_mif)
        name = "plausibility (Isolation Forest)"
    elif metric_name == "local outlier factor":
        discriminator_lof = ClassLocalOutlierFactor()
        func = lambda x: plausibility_local_outlier_factor(x, discriminator_lof)
        name = "plausibility (local outlier factor)"
    elif metric_name == "none":
        return None, None
    else: 
        print("Plausibility metric not defined")
        exit(1)
    return func, name
        
class MetricFunction():
    """
        This class is a template for user-defined metric combinations.

        It is callable and when called with a given Substructure, it will apply
        the given functions to that substructure.
    """

    def __init__(self, *functions, metric_names=[], combination_function=None, n_samples=10000, nu=0.3, gamma="scale"):
        self.functions = functions
        self.n_objectives = len(self.functions)
        self.combination_function = combination_function
        self.metric_names = metric_names


    def __call__(self, sub: SubstructureInterface):
        """Evaluate the metric functions on a given substructure object.

        This calls the metric functions in series and returns the results as list
        and if a combination_function is given also as the combined value.

        Args:
            sub: Substructure object to be evaluated.

        Returns:
            Tuple with
            - the value(s) for the genetic algorithm to work with
                - either a list of values, or
                - whatever the combination_function calculates
            - and with the single values to plot (always a list of results)
        """
        results = []
        for fun in self.functions:
            results.append(fun(sub))

        if (np.inf in results):
            results = [np.inf] * len(results)

        # the evaluation needs the result values and the values for plotting
        if self.combination_function:
            return self.combination_function(results), results
        else:
            return results, results
    
    @staticmethod
    def from_loads(objectives: dict, n_samples=10000, nu=0.5, gamma="scale"):
        if "cost" in objectives and not objectives["cost"]:
            functions = []
            objective_names = []            
        else:
            functions = [total_mass]
            objective_names = ["Cost"]
        if objectives.get("plausibility", False):
            if isinstance(objectives["plausibility"], list):
                for plaus in objectives["plausibility"]:
                    func, name = get_plausibility_metric(plaus.lower(), n_samples=n_samples, nu=nu, gamma=gamma)
                    functions.append(func)
                    objective_names.append(name)
            else:
                func, name = get_plausibility_metric(objectives["plausibility"].lower(), n_samples=n_samples, nu=nu, gamma=gamma)
                if func is not None:
                    functions.append(func)
                    objective_names.append(name)
                

        functions.append(lambda sub: get_max_stress(wave_test, sub=sub, steps=1))
        objective_names.append("wave")

        functions.append(lambda sub: get_max_stress(combined_test, sub=sub, load_dict=objectives, steps=1))
        objective_names.append("combined")

        for fun, name in zip([compression_test, tipover_test, torque_test],
                            ["compression",    "tipover",    "torque"]):
            load = objectives[name]
            if load != 0:
                functions.append(lambda sub, fun=fun, load=load: get_max_stress(fun, sub=sub, load=load, steps=1))
                objective_names.append(name)
        print("Running with objectives:", objective_names)
        return MetricFunction(*functions, metric_names=objective_names, n_samples=n_samples, nu=nu, gamma=gamma)

