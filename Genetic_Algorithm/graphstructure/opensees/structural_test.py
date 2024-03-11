import math
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import opsvis as opsv

from ..graph3d import Node

np.set_printoptions(precision=3, suppress=None)


def get_all_coords():
    tags = ops.getNodeTags()
    return np.array([ops.nodeCoord(node) for node in tags])

def reset_node_disps():
    for node in ops.getNodeTags():
        for dof in range(1, 7): # dofs are counted from 1 to 6
            ops.setNodeDisp(node, dof, 0, '-commit')

def get_model_range(coords=None):
    if coords is None:
        coords = get_all_coords()
    return {dim: (coords[:, x].min(), coords[:, x].max()) for dim, x in zip("xyz", [0,1,2])}

def choose_random_epicenter(model_range):
    center = [0]*3
    for i, dim in enumerate("xyz"):
        center[i] = np.random.randint(*model_range[dim])
    return center

class StructuralTest():
    def __init__(self) -> None:
        self.nodes = np.array([])
        self.load = np.array([])
        self.stretches = []
        self.stretch_percentages = []
        self.init_coords = get_all_coords()
        self.elements = np.array([ops.eleNodes(el) for el in ops.getEleTags()])
        self.ERROR = 0

    def multi_node_test(self, steps, load=None):
        loads = self.load if load is None else load

        try:
            ops.remove('timeSeries', 1)
            ops.remove('loadPattern', 1)
        except:
            print("Currently there hasnt been a timeSeries set")
        ops.timeSeries("Linear", 1) # Load is applied linearly proportional over time
        ops.pattern("Plain", 1, 1)  # Load Pattern with patternTag 1 and timeSeries tag 1

        if loads.ndim == 1:
            loads = loads if len(loads) == 6 else np.hstack((loads, np.zeros(3)))
            for node in self.nodes:
                node = int(node)
                if ops.nodeCoord(node, 3) == 0.0:
                    continue
                ops.load(node, *loads) # Load applied to each node
        else:
            loads = loads if loads.shape[1] == 6 else np.hstack((loads, np.zeros_like(loads)))
            for node, load in zip(self.nodes, loads):
                node = int(node)
                if ops.nodeCoord(node, 3) == 0.0:
                    continue
                ops.load(node, *load) # Load applied to each node

        ops.wipeAnalysis() # remove previous analysis' settings

        #Start of the analysis generation
        ops.system("BandSPD")       # Constructs a Linear equation system and a solver
        ops.numberer("RCM")         # Maps the degress of freedrom of nodes to the equation system
        ops.constraints("Plain")    # creates a constraints handler
        ops.integrator("LoadControl", 1.0) #
        ops.algorithm("Linear")     # linear algorithm to solve the system of equations
        ops.analysis("Static")      # set the analysis type to static analysis
        exec_return = ops.analyze(steps)          # perform the analysis with n steps
        if not exec_return==0:
            self.ERROR = np.inf
        self.analyze_deformation()

    def analyze_deformation(self):
        for element in ops.getEleTags():
            nodes = ops.eleNodes(element)
            init_poss = np.array([ops.nodeCoord(n) for n in nodes])
            disp = np.array([ops.nodeDisp(n)[:3] for n in nodes])
            new_poss = init_poss + disp
            init_length = np.linalg.norm(np.subtract(*init_poss)) # initial distance
            new_length = np.linalg.norm(np.subtract(*new_poss))
            element_stretch = new_length - init_length

            self.stretches.append(element_stretch)
            self.stretch_percentages.append(element_stretch / init_length)

    def analyze_node_reaction(self, nodes=list[int]):
        ops.reactions()
        return [ops.nodeReaction(node) for node in nodes]

    def bottom_stress(self):
        bottom_nodes = [node for node in ops.getNodeTags() if ops.nodeCoord(node)[2] == 0.0]
        reactions = self.analyze_node_reaction(bottom_nodes)
        # one reaction: [Vx, Vy, Vz, Mx, My, Tz]
        # internal force: [-Vx, -Vy, -Vz, -Mx, -My, -Tz]

        # outer and inner diameter of a column in m
        D, d = 1.2, 1.165

        vm_stresses = []
        for r in reactions:
            Vx, Vy, Vz, Mx, My, Tz = r
            shear_stress = (
                    8 * math.sqrt(Vx**2 + Vy**2) / (math.pi * (D**2 - d**2)) +
                    (16*abs(Tz)) / (math.pi * (D+d) * (D**2 - d**2))
                )

            normal_stress = (
                    4 * abs(Vz) / (math.pi * (D**2 - d**2)) +
                    32*D*(abs(Mx) + abs(My)) / (math.pi * (D**4 - d**4))
                )

            von_mises_stress = math.sqrt(normal_stress**2 + 3*shear_stress**2)
            vm_stresses.append(von_mises_stress)
            # TODO: look at opsv.vm_stress()

        return vm_stresses

    def max_bottom_stress(self):
        if not self.ERROR==0:
            return self.ERROR
        return max(self.bottom_stress())

    def get_max_stretch(self):
        max_pos = int(np.argmax(np.abs(self.stretch_percentages)))
        return max_pos
        #({ops.eleNodes(max_pos)}): {self.stretches[max_pos]:.3f}
          #  ({self.stretch_percentages[max_pos]:.1%})")

    def get_max_rotation(self):
        pass # for future analyses

    def get_max_angle_change(self):
        pass # for future analyses

    def plot_defo(self, clean=True, show_max_stretched=False, **kwargs):
        sfac = opsv.plot_defo(fig_wi_he=(30, 30), **kwargs)
        ax = plt.gca()

        if clean:
            ax.axis("off")
            ax.grid(False)

        # plot loaded nodes
        loaded_coordinates = np.array([self.init_coords[node][:3] for node in self.nodes])
        ax.scatter(*(loaded_coordinates.T), color="red")

        # plot most stretched element
        if show_max_stretched:
            max_stretched_nodes = self.elements[self.get_max_stretch()]
            coords = self.init_coords[max_stretched_nodes]
            ax.plot(*coords.T, color="red")
        return ax


class MultiNodeTest(StructuralTest):
    def __init__(self, nodes: list[int], loads) -> None:
        super().__init__()
        self.nodes = nodes
        self.load = np.array(loads)
        if self.load.ndim != 1 and len(self.nodes) != len(self.load):
            raise ValueError(f"Number of loads must either be 1 or {len(nodes)=}."
                            + f"{nodes=}, {loads=}")

    def __call__(self, steps=1):
        self.multi_node_test(steps)


class PlainTest(StructuralTest):
    def __init__(self, load, nodes: list[Node]=None, epicenter=None, radius=0.0):
        super().__init__()
        self.epicenter = np.array(epicenter) if not epicenter is None else None
        self.load = np.array(load)
        self.nodes = [n.idx for n in nodes] if not nodes is None else []
        self.impact_radius = radius

    def __call__(self, *args, **kwds):
        self.perform_test(*args, **kwds)

    def perform_test(self, steps=1):
        # determine nodes to be hit; only take nodes that are NOT at the bottom (z=0)
        if len(self.nodes) == 0:
            self.nodes = [node for node in self.elements_at_epicenter() if self.init_coords[node][2] != 0]
        # perform multiple static analysis with load / len(nodes) as load
        self.multi_node_test(steps, load=self.load / len(self.nodes))

    def elements_at_epicenter(self):
        def angle(yx, yz):
            #yx, yz = x - y, z - y
            cos_angle = np.dot(yx, yz) / (np.linalg.norm(yx) * np.linalg.norm(yz))
            degrees = np.degrees(np.arccos(cos_angle))
            return degrees % 180 # get the inner angle in case x and z are in the wrong order
        def dist(p1, p2):
            p12, p1e, p2e = p1-p2, p1-self.epicenter, p2-self.epicenter             # calculate the vectors
            a, b, c = np.linalg.norm(p1e), np.linalg.norm(p12), np.linalg.norm(p2e) # get their lengths
            if 0 in (sides:=[a,b,c]):
                print(sides)
            if angle(p1e, p12) >= 90:
                return a
            elif angle(-p12, p2e) >= 90:
                return c
            h = 0.5 * np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) / b
            return h

        dists = np.array([dist(self.init_coords[a], self.init_coords[b]) for a, b in self.elements])
        min_dist = min(dists)
        elements = self.elements[np.where(dists <= max(self.impact_radius, min_dist+10))]
        return list(set(np.array(elements).flatten()))

    def nodes_at_epicenter(self):
        distances = np.linalg.norm(self.init_coords - self.epicenter, 2, axis=1)
        closest_distance = distances.min()
        close_nodes = np.argwhere(distances < closest_distance+self.impact_radius)
        return close_nodes.flatten()

    def plot_defo_and_load(self, **kwargs):
        ax = self.plot_defo(**kwargs)
        load_dir = self.load[:3].reshape((3,1))
        plt.quiver(*(self.epicenter.reshape((3,1))), *(load_dir/np.linalg.norm(load_dir)*100), color="orange")

        # plot impact sphere
        if not self.impact_radius is None and self.impact_radius > 0.0:
            n_points = 15
            u = np.linspace(0, np.pi, n_points)
            v = np.linspace(0, 2 * np.pi, n_points)
            x = (np.outer(np.sin(u), np.sin(v))) * self.impact_radius + self.epicenter[0]
            y = (np.outer(np.sin(u), np.cos(v))) * self.impact_radius + self.epicenter[1]
            z = (np.outer(np.cos(u), np.ones_like(v))) * self.impact_radius + self.epicenter[2]
            ax.plot_wireframe(x, y, z, color="red")

        return ax

class Collection(StructuralTest):
    def __init__(self, loads, node_groups):
        super().__init__()
        self.loads = loads
        self.node_groups = node_groups

    def __call__(self, steps=1):
        # prepare loads
        self.nodes: list[int] = []
        loads = []

        for node_group, load in zip(self.node_groups, self.loads):
            single_load = np.array(load) / len(node_group)
            for node in node_group:
                idx = self.nodes.index(node) if node in self.nodes else -1
                if idx == -1:
                    self.nodes.append(node)
                    loads.append(np.array([0.0]*6))
                loads[idx] += single_load

        # execute test
        self.multi_node_test(steps=steps, load=np.array(loads))

