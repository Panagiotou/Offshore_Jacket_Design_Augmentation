from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np

from .graph3d import Edge, LayerBase, Material, Node, SubstructureInterface


class ConnectionBraces(Enum):
    """ Enum for brace types.

    Each member's value is a dictionary containing information about ordering,
    display/storing name, and if the brace has a horizontal connection.
    """

    NONE = dict(id=0, name="NONE", horizontal=False) # No brace (empty)
    H    = dict(id=1, name="H",    horizontal=True)  # No brace, horizontal connections only
    Z    = dict(id=2, name="Z",    horizontal=False) # z brace (counter-clockwise upwards)
    IZ   = dict(id=3, name="IZ",   horizontal=False) # inverted z brace (clockwise upwards)
    ZH   = dict(id=4, name="ZH",   horizontal=True)  # z brace with horizontal edges
    IZH  = dict(id=5, name="IZH",  horizontal=True)  # inverted z brace with horizontal edges
    K    = dict(id=6, name="K",    horizontal=True)  # k brace
    X    = dict(id=7, name="X",    horizontal=False) # x brace
    XH   = dict(id=8, name="XH",   horizontal=True)  # x brace with horizontal edges

    @staticmethod
    def from_str(label: str):
        """Get the correct enum member from a given string.

        Args:
            label: Name of the brace.

        Returns:
            The correct brace if found, ConnectionBraces.NONE otherwise.
        """

        for b in ConnectionBraces:
            if b.value["name"] == label:
                return b
        return ConnectionBraces.NONE

    @classmethod
    def ordered_dict(cls):
        """Create an ordered_dict assigning a number to each brace name.

        This can be used e.g. for consistent plotting in statistics on braces.

        Returns:
            The ordered dictionary.
        """

        return {brace.value["name"]: brace.value["id"] for brace in cls}

@dataclass
class RadiusWrapper:
    """ A wrapper around bottom and top radius.

    This class handles the radius calculation for given layer heights. Here, the
    radius is uniformly decreased from bottom to top, thus making it easy to interpolate
    the radius for any given layer height.
    """

    bottom: float
    top: float

    def get_radius(self, total_height: float, layer_height: float):
        """Calculates the radius of a layer at specific height in the substructure.

        Args:
            total_height: Total height of the structure.
            layer_height: Height of the layer that the radius gets computed for.

        Returns:
            Radius of the given layer.
        """

        assert total_height > layer_height, f"{layer_height=} not allowed with {total_height=}"

        rad_diff = (self.bottom - self.top)
        return self.bottom - layer_height / total_height * rad_diff


def x_brace(nodes: tuple[Node, Node, Node, Node], material: Material) -> tuple[Node, list[Edge]]:
    """
        Returns the crossover of the two diagonals of a trapezoid.
        The nodes must be given in the order that neighboring nodes are next to
        each other in the tuple:
        3 2
        0 1
    """

    # find out point of intersection of the two diagonals
    a, b, c, d = nodes
    v1, v2 = c - a, d - b

    # using Cramer's rule to solve the linear equation system
    D = np.linalg.det([[v1[0], -v2[0]], [v1[1], -v2[1]]])

    # if dividing determinant is close to zero, just take center
    # this happens, when the 4 nodes make an almost perfect square
    if abs(D) <= 1e-12:
        center = a.as_array() + v1 / 2
    else:
        center = (a.as_array() +
            np.linalg.det([[b.x - a.x, -v2[0]], [b.y - a.y, -v2[1]]]) / D * v1
        )
    center_node = Node(*center)

    # create the four edges
    edges: list[Edge] = []
    for n in nodes:
        edges.append(Edge(n, center_node, material))


    return center_node, edges

@dataclass
class Layer(LayerBase):
    """Layer implementation for the SubstructureSimple architecture.

    The information for how the edges are computed is included in this class.
    """

    def __post_init__(self):
        """ Creates nodes of the layer. """
        xs = self.radius * np.cos((np.arange(self.legs)*2*np.pi) / self.legs)
        ys = self.radius * np.sin((np.arange(self.legs)*2*np.pi) / self.legs)

        self.nodes: list[Node] = []
        for x, y in zip(xs, ys):
            self.nodes.append(Node(x, y, self.height))

    def index(self, offset):
        """ Gives all nodes an index and returns the new offset """
        i = 0
        for i, n in enumerate(self.nodes):
            n.idx = offset + i

        return offset + i + 1

    def connect_vertical(self, rhs, material: Material):
        """Vertical connections between the layer and the layer above.

        Args:
            rhs (Layer): The other layer that gets connected to this one.
            material: Material of the vertical edges.

        Returns:
            The list of created edges.
        """

        edges: list[Edge] = []

        for i in range(self.legs):
            edges.append(Edge(self.nodes[i], rhs.nodes[i], material))

        return edges

    def connect_horizontal_top(self, last_brace, material: Material):
        """Create horizontal connections between the nodes of the top layer.

        Args:
            last_brace (ConnectionBraces): Brace between the second to top and
                top layers. Needed to know, if edges have to be added or not.
            material: Material of the horizontal edges.

        Returns:
            The list of created edges.
        """

        if last_brace.value["horizontal"] and last_brace != ConnectionBraces.K:
            return []

        edges: list[Edge] = []
        for i in range(self.legs):
            next_idx = (i+1)%self.legs
            edges.append(Edge(self.nodes[i], self.nodes[next_idx], material))
        return edges

    def connect_brace(self, rhs, brace, last_brace, material: Material):
        """Create the brace connections between two layers.

        Args:
            rhs (Layer): The other layer.
            brace (ConnectionBraces): Brace between both layers.
            last_brace (ConnectionBraces): Brace below this layer.
            material: Material for new edges.

        Returns:
            The list of created edges, and the list of additional nodes.
        """

        edges, additional_nodes = [], []
        for i in range(self.legs):
            next_idx = (i+1)%self.legs
            if brace in [ConnectionBraces.X, ConnectionBraces.XH]:
                # The nodes' order
                # 3 2
                # 0 1
                nodes: list[Node] = [self.nodes[i], self.nodes[next_idx],
                                    rhs.nodes[next_idx], rhs.nodes[i]]

                n, e = x_brace(tuple(nodes), material)
                additional_nodes.append(n)
                edges.extend(e)
            elif brace == ConnectionBraces.K:
                # create connecting node first
                a, b = self.nodes[i].as_array(), self.nodes[next_idx].as_array()
                middle = (b - a) / 2 + a
                new_node = Node(*middle)

                edges.append(Edge(self.nodes[i], new_node, material))
                edges.append(Edge(new_node, self.nodes[next_idx], material))

                # finally create the diagonal k-brace connections
                edges.append(Edge(new_node, rhs.nodes[i], material))
                edges.append(Edge(new_node, rhs.nodes[next_idx], material))
                additional_nodes.append(new_node)
            elif brace in [ConnectionBraces.Z, ConnectionBraces.ZH]:
                edges.append(Edge(self.nodes[i], rhs.nodes[next_idx], material))
            elif brace in [ConnectionBraces.IZ, ConnectionBraces.IZH]:
                edges.append(Edge(self.nodes[next_idx], rhs.nodes[i], material))

            if brace.value["horizontal"] and not brace == ConnectionBraces.K:
                if not last_brace.value["horizontal"] or last_brace == ConnectionBraces.K:
                    edges.append(Edge(self.nodes[i], self.nodes[next_idx], material))
                edges.append(Edge(rhs.nodes[i], rhs.nodes[next_idx], material))
        return edges, additional_nodes


@dataclass
class SubstructureSimple(SubstructureInterface):
    """A less complex implementation of the SubstructureInterface.

    It is more constrained on which connections are possible and is closer to
    real structures.

    Args:
        radius_wrapper: The RadiusWrapper object used for constructing this design.
        layer_heights: List of heights of the layers. The values are not cumulative.
            E.g.: [20, 40] => layer with height 20m, and layer with 40m above the ground.
        connection_types: List of braces. If a single value is given, it is assumed to
            be used for all connections of the whole structure.
        connect_top: If true, connects the top nodes horizontally.
    """

    radius_wrapper: RadiusWrapper
    layer_heights: list[float] = field(default_factory=lambda: [])
    connection_types: Union[str, list[str], ConnectionBraces, list[ConnectionBraces]] = field(
            default_factory=lambda: ConnectionBraces.NONE)
    connect_top: bool = True

    def __post_init__(self):
        """ This mainly preprocesses and checks if the inputs are fine

        Also constructs the structure with layers and braces.
        """

        super().__post_init__()

        # check and set layer_heights if needed
        if self.n_layers == -1:
            self.n_layers = len(self.layer_heights) + 2
        elif len(self.layer_heights) == 0 and self.n_layers != 2:
            # simply interpolate the layer heights if not given
            self.layer_heights = (np.arange(self.n_layers) * self.total_height / (self.n_layers-1))[1:-1].tolist()
        else:
            assert len(self.layer_heights) == self.n_layers - 2, "Layer heights and number of layers dont match"

        self.layer_heights.sort()

        # check and set braces if needed
        if not isinstance(self.connection_types, list):
            # handles old behaviour to use same brace everywhere
            print(f"Setting all layers' braces to {self.connection_types}!")
            self.connection_types = [self.connection_types] * (self.n_layers - 1)
        elif len(self.connection_types) == 0:
            # choose no braces for all layers then...
            print("Setting all layers' braces to NONE!")
            self.connection_types = [ConnectionBraces.NONE] * (self.n_layers - 1)
        else:
            assert len(self.connection_types) == self.n_layers - 1, "Connection types and number of layers dont match"

        # parse ConnectionBraces as string input
        for i, b in enumerate(self.connection_types):
            if isinstance(b, str):
                self.connection_types[i] = ConnectionBraces.from_str(b)

        self.construct()

    def construct(self):
        self.layers: list[Layer] = []

        # create all layers bottom and top are special cases
        self.layers.append(Layer(legs=self.legs, height=0.0, radius=self.radius_wrapper.bottom))

        for layer_idx in range(self.n_layers - 2):
            height = self.layer_heights[layer_idx]
            radius = self.radius_wrapper.get_radius(self.total_height, height)
            self.layers.append(Layer(legs=self.legs, height=height, radius=radius))

        self.layers.append(Layer(legs=self.legs, height=self.total_height, radius=self.radius_wrapper.top))

        # index all current nodes
        offset = 0
        for l in self.layers:
            offset = l.index(offset)

        self.additional_nodes: list[Node] = []

        # add vertical and brace connections
        for i, (lower, upper) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.edges.extend(lower.connect_vertical(upper, Material.Strong))
            last_brace = self.connection_types[i-1] if i > 0 else ConnectionBraces.NONE
            edges, additional_nodes = lower.connect_brace(upper, self.connection_types[i], last_brace, Material.Weak)
            self.edges.extend(edges)
            self.additional_nodes.extend(additional_nodes)

        # add top horizontal connections
        if self.connect_top:
            self.edges.extend(self.layers[-1].connect_horizontal_top(self.connection_types[-1], Material.Weak))

        # give indices to the additional nodes
        i = 0
        for i, n in enumerate(self.additional_nodes):
            n.idx = offset + i

        self.nodes = self.get_all_nodes()

    def get_all_nodes(self) -> list[Node]:
        nodes = self.additional_nodes

        for layer in self.layers:
            nodes.extend(layer.nodes)

        nodes.sort(key=lambda node: node.idx)

        return nodes

    @staticmethod
    def from_dict(d):
        c = d["connection_types"]
        connection_types = c if isinstance(c, str) else c.copy()
        return SubstructureSimple(
            legs             = d["legs"],
            total_height     = d["total_height"],
            connection_types = connection_types,
            radius_wrapper   = RadiusWrapper(d["radius_bottom"], d["radius_top"]),
            n_layers         = int(d["n_layers"]),
            layer_heights    = d["layer_heights"],
        )

    def to_dict(self):
        return {
            "legs":             self.legs,
            "total_height":     self.total_height,
            "connection_types": [b.value["name"] for b in self.connection_types],
            "radius_bottom":    self.radius_wrapper.bottom,
            "radius_top":       self.radius_wrapper.top,
            "n_layers":         self.n_layers,
            "layer_heights":    self.layer_heights,
        }

