from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import math

import numpy as np
from numpy.typing import NDArray


class Material(Enum):
    """ Enum class with different material types. """

    Weak = 1
    Strong = 2
    Truss = 3
    Inf = 4

@dataclass
class Node:
    """ Representation of a node in the 3-dimensional graph. """

    x: float
    y: float
    z: float
    idx: int = -1
    def as_array(self) -> NDArray[np.float64]:
        return np.array([self.x, self.y, self.z])

    def __sub__(self, other) -> NDArray[np.float64]:
        """ Conveniently substract coordinates from each other. """
        return self.as_array() - other.as_array()

    def __add__(self, other) -> NDArray[np.float64]:
        """ Conveniently add coordinates from two nodes. """
        return self.as_array() + other.as_array()

@dataclass
class Edge:
    """ An edge between two nodes in the 3-dimensional graph.

    The edge also has a material parameter telling about its strenght.
    """

    start: Node
    end: Node
    material: Material

    def __iter__(self):
        return iter((self.start, self.end, self.material))

    def as_tuple(self) -> tuple[int, int]:
        """ returns the start and end nodes' indices as a tuple """
        return (self.start.idx, self.end.idx)

@dataclass
class LayerBase:
    """ A base class for different substructure layer implementations.

    By definition, a layer must have a height above the ground, a radius on
    which its nodes are spread out and a count of legs, i.e. corners on the polygon.
    """

    legs: int
    height: float
    radius: float

@dataclass
class Layer(LayerBase):
    """ Layer implementaion for complex substructures.

    This implementation adds layer-wise rotations, center nodes and implements
    horizontal layers as part of a vertical layer.

    Horizontal layers are constructed with the hori_dists parameter, which is a
    list of distances between the horizontal layers. The first value depicts the
    distance between the main (inner) layer and the first horizontal layer.
    The second value depicts the distance between the first to horizontal layers
    and so on.
    An empty list implies, that there are no extra horizontal layers.
    """

    rotation: float # in radians
    has_center_node: bool
    hori_dists: list[int] = field(default_factory=lambda: [])

    def __post_init__(self) -> None:
        self.nodes: list[Node] = []
        self.h_layers = len(self.hori_dists)+1

        if self.has_center_node:
            self.nodes.append(Node(0.,0.,self.height))

        for i in range(self.h_layers):
            if i > 0:
                self.radius += self.hori_dists[i-1]
            xs = self.radius * np.cos((np.arange(self.legs)*2*math.pi) / self.legs + self.rotation)
            ys = self.radius * np.sin((np.arange(self.legs)*2*math.pi) / self.legs + self.rotation)
            for x, y in zip(xs, ys):
                self.nodes.append(Node(x, y, self.height))

    def set_global_idcs(self, offset):
        for i, node in enumerate(self.nodes):
            node.idx = offset + i
        return len(self.nodes)

    def __getitem__(self, idx):
        # This is a little hacky and might lead to bugs...
        if self.has_center_node and idx >= 0:
            return self.nodes[idx+1]
        return self.nodes[idx]


class Connections:
    """ Definition of various connections between layers and inside layers.

    This class is used for the complex substructure architecture, which offers
    many different connection types, which are implemented here.
    Clockwise is the direction of a diagonal connection to the left seen from above the layer.
    """

    def __init__(self, h_layers=1, **kwargs) -> None:
        self.h_layers = h_layers
        self.basic_connections = {
            "h_neighbor":        False, # horizontal neighbor
            "h_opposite":        False, # opposite of node in layer (only for inner layer)
            "v_neighbor":        False, # nodes above each other
            "u_r_neighbor":      False, # upper right neighbor (counter-clockwise)
            "u_l_neighbor":      False, # upper left neighbor (clockwise)
            "u_opposite":        False, # upper opposite (only inner layer)
        }
        self.center_connections = {
            "center_vertical":   False, # connect two center points above each other
            "center_horizontal": False, # connections between center node and inner hori layer; conflicts with h_opposite
            "center_u_out":      False, # from center to upper first hori layer
            "center_u_in":       False, # from upper first hori layer to center
        }
        self.extended_connections = {
            "u_in_twin":         False, # upper inner twin
            "u_out_twin":        False, # upper outer twin
            "u_in_right":        False, # upper right neighbor of inner twin (counter-clockwise)
            "u_in_left":         False, # upper left neighbor of inner twin (clockwise)
            "u_out_right":       False, # upper right neighbor of outer twin (counter-clockwise)
            "u_out_left":        False, # upper left neighbor of outer twin (clockwise)
            "h_twin":            False, # horizontal layer twin (outer to inner)
            "h_out_right":       False, # outward clockwise diagonal of horizontal layers
            "h_out_left":        False, # outward counter-clockwise diagonal of horizontal layers
        }
        self.default_types: defaultdict = defaultdict(lambda: False)
        self.default_types.update(**self.basic_connections)
        self.default_types.update(**self.center_connections)
        if self.h_layers > 1:
            self.default_types.update(**self.extended_connections)

        # overwrite the defaults with what the kwargs contain
        self.types = self.default_types | kwargs # python 3.9 needed for this
        self.edges: list[Edge] = []
        self.has_center_connections = any(self.types[t] for t in self.center_connections.keys())

    @staticmethod
    def random(h_layers=1, n_connection_types=4, allowed_connections: list[str] = []):
        conns = Connections(h_layers=h_layers).types
        allowed_connections = allowed_connections or list(conns.keys())
        for c in np.random.choice(allowed_connections, n_connection_types):
            conns[c] = True
        return Connections(h_layers=h_layers, **conns)

    @staticmethod
    def from_dict(d):
        return Connections(len(d["hori_layer_dists"])+1, **d['connections'])

    def __getitem__(self, key):
        return self.types[key]

    def __setitem__(self, key, value):
        self.types[key] = value

    def __eq__(self, other) -> bool:
        if self.h_layers == other.h_layers == 1:
            for key in self.basic_connections.keys():
                if self[key] != other[key]:
                    return False
            for key in self.center_connections.keys():
                if self[key] != other[key]:
                    return False
            return True
        else:
            return self.types == other.types

    def connect(self, lower: Layer, upper: Layer):
        if self["v_neighbor"]:
            self.c_v_neighbor(lower, upper)

        if self["center_vertical"]:
            self.edges.append(Edge(lower.nodes[0], upper.nodes[0], Material.Strong))

        if self["center_u_in"]:
            for i in range(lower.legs):
                self.edges.append(Edge(lower[i], upper.nodes[0], Material.Strong))

        if self["center_u_out"]:
            for i in range(upper.legs):
                self.edges.append(Edge(lower.nodes[0], upper[i], Material.Strong))

        if self["u_r_neighbor"]:
            self.c_u_r_neighbor(lower, upper)

        if self["u_l_neighbor"]:
            self.c_u_r_neighbor(upper, lower) # switch arguments

        if self["u_in_twin"]:
            self.c_u_out_twin(upper, lower) # switch arguments

        if self["u_out_twin"]:
            self.c_u_out_twin(lower, upper)

        if self["u_opposite"]:
            self.c_u_opposite(lower, upper)

        if self["u_out_right"]:
            self.c_u_h_right(lower, upper)

        if self["u_out_left"]:
            self.c_u_h_right(upper, lower, invert_h_dir=True)

        if self["u_in_right"]:
            self.c_u_h_right(lower, upper, invert_h_dir=True)

        if self["u_in_left"]:
            self.c_u_h_right(upper, lower)

    def self_connect(self, layer: Layer):
        if self["h_neighbor"]:
            self.c_h_neighbor(layer)

        if self["h_twin"]:
            self.c_h_twin(layer)

        if self["h_opposite"]:
            self.c_h_opposite(layer)

        if self["h_out_left"]:
            self.c_h_out(layer)

        if self["h_out_right"]:
            self.c_h_out(layer, clockwise=False)

        if self["center_horizontal"]:
            for i in range(layer.legs):
                self.edges.append(Edge(layer.nodes[0], layer[i], Material.Strong))

    def get_connections(self):
        return self.edges

    def reset(self):
        self.edges = []

    def c_h_neighbor(self, layer: Layer):
        cnt = 0
        for i in range(layer.h_layers):
            for j in range(layer.legs-1):
                self.edges.append(Edge(layer[cnt+j], layer[cnt+j+1], Material.Strong))

            self.edges.append(Edge(layer[cnt], layer[cnt+layer.legs-1], Material.Strong))
            cnt += layer.legs

    def c_h_twin(self, layer: Layer):
        cnt = layer.legs
        for i in range(1, layer.h_layers):
            for j in range(layer.legs):
                self.edges.append(Edge(layer[cnt+j], layer[cnt+j-layer.legs], Material.Strong))
            cnt += layer.legs

    def c_h_opposite(self, layer: Layer):
        if layer.legs < 4:
            return

        half_legs = layer.legs // 2
        if layer.legs % 2 == 0:
            for c in range(half_legs):
                self.edges.append(Edge(layer[c], layer[half_legs+c], Material.Strong))
        else:
            for c in range(layer.legs):
                self.edges.append(Edge(layer[c], layer[(half_legs+c)%layer.legs], Material.Strong))

    def c_h_out(self, layer: Layer, clockwise=True):
        C = layer.legs
        cnt = C
        if clockwise:
            for i in range(1, layer.h_layers):
                for j in range(C-1):
                    self.edges.append(Edge(layer[cnt+j-C+1], layer[cnt+j], Material.Strong))
                self.edges.append(Edge(layer[cnt-C], layer[cnt+C-1], Material.Strong))
                cnt += C
        else:
            for i in range(1, layer.h_layers):
                self.edges.append(Edge(layer[cnt-1], layer[cnt], Material.Strong))
                for j in range(1, C):
                    self.edges.append(Edge(layer[cnt+j-C-1], layer[cnt+j], Material.Strong))
                cnt += C

    def c_v_neighbor(self, lower: Layer, upper: Layer):
        start_idx = 1 if self.has_center_connections else 0

        for a, b in zip(lower.nodes[start_idx:], upper.nodes[start_idx:]):
            self.edges.append(Edge(a, b, Material.Strong))

    def c_u_r_neighbor(self, lower: Layer, upper: Layer):
        for h_layer in range(lower.h_layers):
            offset = h_layer*lower.legs
            for c in range(lower.legs-1):
                self.edges.append(Edge(lower[c+offset], upper[c+offset+1], Material.Strong))

            # wrap around; special case
            c = lower.legs-1
            self.edges.append(Edge(lower[offset+c], upper[offset], Material.Strong))

    def c_u_out_twin(self, lower: Layer, upper: Layer):
        for h_layer in range(lower.h_layers-1):
            l_offset = h_layer*lower.legs
            u_offset = (h_layer+1)*lower.legs
            for c in range(lower.legs):
                self.edges.append(Edge(lower[l_offset+c], upper[u_offset+c], Material.Strong))

    def c_u_opposite(self, lower: Layer, upper: Layer):
        if lower.legs < 4:
            return

        half_legs = lower.legs // 2
        C = lower.legs
        if C % 2 == 0:
            for c in range(C):
                self.edges.append(Edge(lower[c], upper[(half_legs+c)%C], Material.Strong))
        else:
            for c in range(C):
                self.edges.append(Edge(lower[c], upper[(half_legs+c)%C], Material.Strong))

    def c_u_h_right(self, start: Layer, dest: Layer, invert_h_dir: bool = False):
        """ This function can be used for 4 connection types:
                - inwards up left (clockwise)
                - inwards up right (counter-clockwise)
                - outwards up left (clockwise)
                - outwards up right (counter-clockwise)
        """
        legs = start.legs
        for h_layer in range(start.h_layers-1):
            s_offset = h_layer*legs
            d_offset = (h_layer+1)*legs
            if invert_h_dir:
                s_offset, d_offset = d_offset, s_offset
            for c in range(legs-1):
                self.edges.append(Edge(start[s_offset+c], dest[d_offset+c+1], Material.Strong))

            c = legs-1
            self.edges.append(Edge(start[s_offset+c], dest[d_offset], Material.Strong))

@dataclass
class SubstructureInterface:
    """ This is an interface for defining different Substructure architecture classes.

    Many functions like plotting and structural testing can be performed only
    on the sets of nodes and edges. Thus, to avoid writing functions for both
    SubstructureComplex and SubstructureSimple would be unnecessary.

    Args:
        total_height: Height from bottom to top layer.
        legs: Number of main legs. Usually 3-5
        n_layers: Number of layers. The bottom and top are included.
            E.g.: n_layers=4 => bottom, layer1, layer2, top
    """

    total_height:  float
    legs:          int
    n_layers:      int

    def __post_init__(self) -> None:
        """ Predefine members that all inheriting classes need to use. """
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.layers: list[Layer] = []

    @staticmethod
    def from_dict(d: dict):
        """Create an object with the correct implementation of this interface.

        This is not a very good way to build a factory. But if needed in the future,
        it can be factored outside of this interface.

        Args:
            d: Dict containing the information of a substructure design.

        Returns:
            An object of either inheriting class, whichever the dict corresponds to.
        """

        if "hori_layer_dists" in d.keys():
            return SubstructureComplex.from_dict(d)
        else:
            from .simple_substructure import SubstructureSimple
            return SubstructureSimple.from_dict(d)

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def load_json(fname):
        """Creates a substructure object from a design in a json file.

        Args:
            fname (Path): Path to the design json file.

        Returns:
            Substructure object described by the design info in the json file.
        """
        with open(fname, "r") as f:
            d = json.load(f)
            s = SubstructureInterface.from_dict(d)
        return s

    @staticmethod
    def load_json_str(s):
        d = json.loads(s)
        s = SubstructureInterface.from_dict(d)
        return s

    def store_json(self, fname: Path=None):
        """Store the substructure design information in a json file.

        Args:
            fname: Path to the file to write to. Overwrites the contents of that file.

        Returns:
            The json string that is written to the file.
        """

        json_string = json.dumps(self.to_dict())
        if not fname is None:
            with open(fname, "w") as f:
                f.write(json_string)
        return json_string

    def store_graph(self, fname: Path = None):
        """Stores graph information in a json file.

        The json will contain two keys:
            - nodes: A ordered list of the node coordinates.
            - edges: A list of tuples, containing the indices of both nodes and
                a binary variable describing if the edge is of strong material or not.

        Args:
            fname: Path to the graph json to write to. Overwrites contents.

        Returns:
            The dictionary that gets written to the json file.
        """
        def edge_with_binary_material(edge: Edge):
            res = list(edge.as_tuple())
            res.append(0 if edge.material == Material.Weak else 1)
            return res

        d = {
            "nodes": [n.as_array().tolist() for n in self.nodes],
            "edges": [edge_with_binary_material(edge) for edge in self.edges]
        }
        if not fname is None:
            with open(fname, "w") as f:
                json.dump(d, f)

        return d

    def plot(self, method="ops", rotation=0, clean=True, **kwargs):
        """Plot the substructure graphs.

        Args:
            method (str): Currently, only "ops" is implemented.
            rotation (float): z-rotation of the whole structure. In radians.
            clean (bool): If true, leaves out extra info like node labels.
            **kwargs: Extra flags for the actual plot function.
        """

        if method == "ops":
            from .opensees.model import sub_to_ops
            import opsvis

            if clean:
                kwargs = dict(node_labels=False, element_labels=False,
                              axis_off=True, local_axes=False) | kwargs
            sub_to_ops(self, rotation=rotation)
            opsvis.plot_model(**kwargs)
        else:
            print(f"Implementation for {method=} doesnt exist!")

    def is_fully_connected(self):
        """Computes if all nodes are reachable from each other.

        TODO: Could be implemented quicker with some MST algorithm.

        Returns:
            True if fully connected, False otherwise.
        """

        adj = np.eye(len(self.nodes))
        for edge in self.edges:
            a, b = edge.as_tuple()
            adj[a, b] = adj[b, a] = 1

        adj_fully_walked = np.linalg.matrix_power(adj, len(self.nodes))
        return (adj_fully_walked != 0).all()

    def sort_edges(self):
        """ For every edge, put the smaller idx infront. """
        for edge in self.edges:
            if edge.start.idx > edge.end.idx:
                edge.start, edge.end = edge.end, edge.start

@dataclass
class SubstructureComplex(SubstructureInterface):
    """ A complex implementation of Substructures.

    Args:
        init_radius: The layer radius at the bottom layer.
        radius_evolution: Multiplier for the radius for each consecutive layer.
        rotation: Rotation between the bottom and top layers in radians.
            This gets uniformly distributed among the intermediate layers.
        connections: A Connections object describing which braces the structure
            consists of.
        hori_layer_dists: Distances between horizontal layers.
        layer_dist_evolution: Describes how big the distance between two layers is
            compared to the distance between the layers below.
            E.g.: n_layers=3, layer_dist_evolution=0.9:
                dist(layer2, layer3) = dist(layer1, layer2) * 0.9
    """

    init_radius:          int         = 100
    radius_evolution:     float       = 0.9
    rotation:             float       = 0.0 # in degrees. sets rotation of lowest to highest layer
    # distance_evolution: float       = 0.5
    connections:          Connections = field(default_factory=lambda: Connections())
    hori_layer_dists:     list[int]   = field(default_factory=lambda: [])
    layer_dist_evolution: float       = 1.0

    def __post_init__(self) -> None:
        """ Some initialization. Constructs the whole structure already. """

        super().__post_init__()

        # convert degrees to radians
        self.per_layer_rotation = math.pi * self.rotation / 180 / (self.n_layers-1)

        self.has_center_node = self.connections.has_center_connections
        self.h_layers = len(self.hori_layer_dists) + 1

        self.construct()

    def construct(self):
        """ Setup layers, and create all chosen connections between them. """

        self.layers: list[Layer] = []

        relative_layer_dists = np.multiply.accumulate(
                [1.]+[self.layer_dist_evolution]*(self.n_layers-2))
        layer_dists = relative_layer_dists / relative_layer_dists.sum() * self.total_height

        height = 0
        idx_offset = 0
        radius = self.init_radius
        for i in range(self.n_layers):
            new_layer = Layer(legs=self.legs, height=height, radius=radius,
                    rotation=i*self.per_layer_rotation, hori_dists=self.hori_layer_dists,
                    has_center_node=self.has_center_node)
            idx_offset += new_layer.set_global_idcs(idx_offset)
            self.nodes.extend(new_layer.nodes)
            self.layers.append(new_layer)

            if i > 0:
                self.connect_layers(self.layers[-2], self.layers[-1])

            radius *= self.radius_evolution # logarithmic decrease
            height += layer_dists[i] if i != self.n_layers - 1 else 0
        self.connections.self_connect(self.layers[-1])
        self.edges = self.connections.get_connections()
        self.sort_edges()

    def connect_layers(self, lower, upper):
        self.connections.self_connect(lower)
        self.connections.connect(lower, upper)

    @staticmethod
    def from_dict(d: dict):
        """Create a SubstructureComplex object from a design dictionary.

        Args:
            d: Design dict.

        Returns:
            A substructure object.
        """

        conns = Connections(h_layers=len(d["hori_layer_dists"])+1, **d["connections"])
        s = SubstructureComplex(
            n_layers=d["n_layers"],
            legs=d["legs"],
            init_radius=d["init_radius"],
            radius_evolution=d["radius_evolution"],
            rotation=d["rotation"],
            total_height=d["total_height"],
            connections=conns,
            hori_layer_dists=d["hori_layer_dists"],
            layer_dist_evolution=d["layer_dist_evolution"]
        )
        return s

    def to_dict(self):
        """Save all design parameters of the object to a dictionary.

        Returns:
            The design dictionary.
        """

        d = {
            "n_layers":             self.n_layers,
            "legs":                 self.legs,
            "init_radius":          self.init_radius,
            "radius_evolution":     self.radius_evolution,
            "layer_dist_evolution": self.layer_dist_evolution,
            "rotation":             self.rotation,
            "total_height":         self.total_height,
            "connections":          {k:v for (k,v) in self.connections.types.items()},
            "hori_layer_dists":     self.hori_layer_dists
        }
        return d


### Some special cases of connections
BCC    = Connections(h_neighbor=False, v_neighbor=False, u_opposite=True)
BCCZ   = Connections(h_neighbor=False, v_neighbor=True, u_opposite=True)
FCC    = Connections(h_neighbor=False, v_neighbor=False, h_opposite=True, u_r_neighbor=True, u_l_neighbor=True)
FBCC   = Connections(h_neighbor=False, v_neighbor=False, h_opposite=True, u_r_neighbor=True, u_l_neighbor=True, u_opposite=True)
SFCC   = Connections(h_neighbor=False, v_neighbor=False, u_r_neighbor=True, u_l_neighbor=True)
SFCCZ  = Connections(h_neighbor=False, v_neighbor=True, u_r_neighbor=True, u_l_neighbor=True)
SFBCC  = Connections(h_neighbor=False, v_neighbor=False, u_r_neighbor=True, u_l_neighbor=True, u_opposite=True)
SFBCCZ = Connections(h_neighbor=False, v_neighbor=True, u_r_neighbor=True, u_l_neighbor=True, u_opposite=True)
