from dataclasses import dataclass, field
import numpy as np

from .graph3d import Connections, SubstructureComplex, SubstructureInterface
from .simple_substructure import ConnectionBraces, RadiusWrapper, SubstructureSimple


def randint_end_inclusive(min: int, max: int, n: int) -> np.ndarray:
    """
        Convenience function for sampling numbers in an interval with the
        right side included.
    """

    if n == 1:
        return np.random.randint(min, max+1, 1).astype(int)[0]
    return np.random.randint(min, max+1, n).astype(int).tolist()

class SubstructureConstraints:
    """ Constraint interface defining some functions for inheriting classes. """

    def sample_fully_connected(self, _) -> list[SubstructureInterface]:
        raise NotImplementedError

    @staticmethod
    def from_dict(d):
        if "h_layers" in d.keys():
            return ComplexConstraints.from_dict(d)
        else:
            return SimpleConstraints.from_dict(d)

    def to_dict(self):
        raise NotImplementedError

@dataclass
class ComplexConstraints(SubstructureConstraints):
    """
        Constraints defined so that a COMPLEX substructure can only be
        generated in a certain space.
        This class does not enforce any constraints, but is rather just used to
        collect, package and read constraints which can later be used by other
        classes or functions.
        Also it defines a sample function to sample designs from constraint space.

        Most of the constraints are defined as tuples of numbers depicting the
        lower and upper bounds in which the parameter can reside.

        The one exception is allowed_connections, which is a list of connections
        that can be used. The sampling is constrained to these connections.
    """

    total_height:         tuple[float, float] = (1000, 1000)
    layers:               tuple[int, int]     = (3, 7)
    h_layers:             tuple[int, int]     = (1, 3)
    h_layer_dist:         tuple[int, int]     = (75, 200)
    legs:                 tuple[int, int]     = (3, 7)
    rotation:             tuple[int, int]     = (-270, 270)
    radius:               tuple[int, int]     = (100, 250)
    radius_evolution:     tuple[float, float] = (0.5, 1.)
    layer_dist_evolution: tuple[float, float] = (0.5, 1.)
    allowed_connections:  dict[str, bool]     = field(
            default_factory=lambda: dict(Connections(h_layers=2).default_types))

    def _get_tuple_constraint_names(self):
        """ Helper function needed for storing and loading from dictionaries. """
        return ["total_height", "layers", "legs", "h_layers", "layer_dist_evolution",
                "h_layer_dist", "rotation", "radius", "radius_evolution"]

    def __post_init__(self):
        """
            We assume, that all constraints are tuples of min/max values.
            If only one value is given, that value is transformed into a tuple
            of (value, value).
        """

        for key in self._get_tuple_constraint_names():
            value = self.__getattribute__(key)
            if isinstance(value, tuple) and len(value) == 1:
                self.__setattr__(key, (value[0], value[0]))
            else:
                self.__setattr__(key, (value, value))

    def to_dict(self, unroll: bool=True):
        """Create a dictionary that can later be parsed to create a constraints object.

        Args:
            unroll: If true, each tuple is split into two values (min and max)
                in the dict as separate key-value-pairs. E.g.:
                    { "total_height_min": 20, "total_height_max": 30 }

        Returns:
            Dictionary with constraints.
        """

        d = {}
        for key in self._get_tuple_constraint_names():
            values = self.__getattribute__(key)
            if unroll:
                d[key + "_min"] = values[0]
                d[key + "_max"] = values[1]
            else:
                d[key] = values
        d["allowed_connections"] = [{"name": key, "allowed": value}
                for key, value in self.allowed_connections.items()]
        return d

    @staticmethod
    def from_dict(d, unrolled=True):
        """ Create a constraint object from a dictionary.

        Args:
            d (dict): Constraint dictionary.
            unrolled (bool): If True, assumes that min-max-tuples are presented
                as separate key-value-pairs. E.g.:
                    { "total_height_min": 20, "total_height_max": 30 }

        Returns:
            [TODO:return]
        """
        sc = ComplexConstraints()
        for key in sc._get_tuple_constraint_names():
            if unrolled:
                sc.__setattr__(key, (float(d[key + "_min"]), float(d[key + "_max"])))
            else:
                sc.__setattr__(key, (d[key]))
        for key in sc.allowed_connections.keys():
            # Have to compare to the string "True", thats how JS sends the information
            sc.allowed_connections[key] = d[key] == "True"
        return sc

    def get_allowed(self) -> list[str]:
        res = [k for k, v in self.allowed_connections.items() if v]
        if not res:
            res = list(self.allowed_connections.keys())
        return res

    def sample_fully_connected(self, n: int=1000):
        """ Only sample designs that are fully connected. """
        return self.sample(n, fully_connected=True)

    def sample(self, n: int=1000, fully_connected=False):
        """Sample n complex designs with constraints.

        Args:
            n: Number of sampled designs.
            fully_connected (bool): If True, only returns designs that are fully connected.

        Returns:
            A list of sampled designs.
        """
        allowed_conns = self.get_allowed()
        subs = []

        while len(subs) < n:
            layers = randint_end_inclusive(*self.layers, 1)
            legs = randint_end_inclusive(*self.legs, 1)
            radius = randint_end_inclusive(*self.radius, 1)
            h_layers = randint_end_inclusive(*self.h_layers, 1)
            n_connections = randint_end_inclusive(3, 10, 1)

            min, max = self.radius_evolution
            radius_evol = float(np.random.rand() * (max-min) + min)
            min, max = self.layer_dist_evolution
            layer_dist_evolution = float(np.random.rand() * (max-min) + min)
            min, max = self.rotation
            rotation = float(np.random.rand() * (max-min) + min)
            min, max = self.total_height
            total_height = np.random.rand() * (max-min) + min

            hori_dists = []
            if h_layers > 1:
                hori_dists = randint_end_inclusive(*self.h_layer_dist, h_layers-1)
                if not isinstance(hori_dists, list):
                    hori_dists = [int(hori_dists)]

            c = Connections.random(h_layers, n_connections, allowed_conns)

            s = SubstructureComplex(
                total_height=total_height,
                n_layers=int(layers),
                legs=int(legs),
                init_radius=int(radius),
                radius_evolution=float(radius_evol),
                rotation=float(rotation),
                connections=c,
                hori_layer_dists=hori_dists,
                layer_dist_evolution=float(layer_dist_evolution)
            )

            # only add if structure is fully connected
            if fully_connected and s.is_fully_connected():
                subs.append(s)

        return subs

    def enforce_on_design(self, design: dict):
        """ Constraints are applied to the given design.

        If a value lies outside of its constraint, it will be pushed
        back to the closest constraint border. The changes are applied
        directly on the passed design, thus there is no returned object.

        Args:
            design: A ComplexSubstructure design dict that constraints are
                enforced on (inplace).
        """

        for c in self._get_tuple_constraint_names():
            min_, max_ = self.__getattribute__(c)
            if c == "h_layers":
                curr_dists = design["hori_layer_dists"]
                new_len = int(max(min(len(curr_dists), max_), min_))
                design["hori_layer_dists"] = curr_dists[:new_len-1]
            if not c in design.keys(): # quick fix for something like 'total_height'
                continue
            design[c] = max(min(design[c], max_), min_)


def sample_from_enum(e_list: list, n: int) -> list:
    """ Helper function that samples members from an enum class.

    Args:
        e_list: List of enum members.
        n: Number of samples.

    Returns:
        Randomly sampled list of the given enum members.
    """
    return np.random.choice(e_list, size=n).tolist()


@dataclass
class SimpleConstraints(SubstructureConstraints):
    """
        Constraints defined so that a SIMPLE substructure can only be
        generated in a certain space.
        This class does not enforce any constraints, but is rather just used to
        collect, package and read constraints which can later be used by other
        classes or functions.
        Also it defines a sample function to sample designs from constraint space.

        Most of the constraints are defined as tuples of numbers depicting the
        lower and upper bounds in which the parameter can reside.

        The one exception is allowed_connections, which is a list of connections
        that can be used. The sampling is constrained to these connections.
    """
    total_height:        tuple[float, float]    = (50, 100)
    n_layers:            tuple[int, int]        = (3, 7)
    legs:                tuple[int, int]        = (3, 4)
    radius_bottom:       tuple[int, int]        = (10, 20)
    radius_top:          tuple[int, int]        = (10, 20)
    allowed_connections: list[ConnectionBraces] = field(
            default_factory=lambda: [b for b in ConnectionBraces])

    def __post_init__(self):
        for i, b in enumerate(self.allowed_connections):
            if isinstance(b, str):
                self.allowed_connections[i] = ConnectionBraces.from_str(b)

    def _get_tuple_constraint_names(self):
        """ Function returning names of all min-max-tuple members. """
        return ["total_height", "n_layers", "radius_bottom", "radius_top", "legs"]

    def to_dict(self):
        """ Save the constraints as a dictionary. """
        d = {}
        for key in self._get_tuple_constraint_names():
            values = self.__getattribute__(key)
            d[key + "_min"] = values[0]
            d[key + "_max"] = values[1]

        d["allowed_connections"] = [{"name": b.value["name"], "allowed": True} for b in self.allowed_connections]
        for b in ConnectionBraces:
            if b not in self.allowed_connections:
                d["allowed_connections"].append({"name": b.value["name"], "allowed": False})
        return d

    @staticmethod
    def from_dict(d):
        """ Build a constraints object from a dictionary. """
        sc = SimpleConstraints()
        for key in sc._get_tuple_constraint_names():
            try:
                sc.__setattr__(key, (int(d[key + "_min"]), int(d[key + "_max"])))
            except:
                sc.__setattr__(key, (float(d[key + "_min"]), float(d[key + "_max"])))

        allowed = []
        for b in ConnectionBraces:
            if d[b.value["name"]] == "True":
                allowed.append(b)
        if allowed:
            sc.allowed_connections = allowed

        return sc

    def sample_fully_connected(self, n: int = 1000):
        """
            For SubstructureSimple objects it is not possible to not be fully connected,
            as all legs are connected either through braces or thorugh the top connection.
        """
        return self.sample(n)

    def sample(self, n: int = 1000):
        """Sample n simple designs with constraints.

        Args:
            n: Number of sampled designs.

        Returns:
            A list of sampled designs.
        """
        layers = randint_end_inclusive(*self.n_layers, n)
        legs = randint_end_inclusive(*self.legs, n)

        min, max = self.radius_top
        radius_top = np.random.rand(n) * (max-min) + min

        # make bottom radius' lower bound the corresponding top radius
        min, max = np.maximum(radius_top, self.radius_bottom[0]), self.radius_bottom[1]
        radius_bottom = np.random.rand(n) * (max-min) + min

        min, max = self.total_height
        heights = (np.random.rand(n)*(max-min)+min).tolist()

        subs: list[SubstructureSimple] = []
        for i in range(n):
            layer_heights = (np.random.rand(layers[i]-2) * heights[i]).tolist()

            subs.append(SubstructureSimple(
                legs            = legs[i],
                total_height    = float(heights[i]),
                n_layers        = int(layers[i]),
                layer_heights   = layer_heights,
                connection_types = sample_from_enum(self.allowed_connections, layers[i]-1),
                radius_wrapper  = RadiusWrapper(float(radius_bottom[i]), float(radius_top[i]))
                ))
        return subs


