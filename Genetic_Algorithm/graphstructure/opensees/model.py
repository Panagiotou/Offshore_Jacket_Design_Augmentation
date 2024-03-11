import math
import openseespy.opensees as ops

from ..graph3d import Material, SubstructureInterface

# define some mass values.
E = 29500.0
M = 0.0
MASS_TYPE = '-lMass'
MASSES_PER_DOF = [0.49, 0.49, 0.01, *([1e-10]*3)]
E_mod, G_mod = 2.1e11, 8.077e10
# Cross-section parameters
# for main legs (D_out=1.2m, D_inn=1.165m)
M_area, M_Jxx, M_Iy, M_Iz = 6.5011e-2, 2.273141e-2, 1.136570e-2, 1.136570e-2
# for X-braces (D_out=0.8m, D_inn=0.78m)
X_area, X_Jxx, X_Iy, X_Iz = 2.4819e-2, 3.87293e-3, 1.936469e-3, 1.936469e-3
# for the infinitely strong top edges
Inf_area, Inf_Jxx, Inf_Iy, Inf_Iz = 7.854e-3, 9.8175e-6, 4.9087e-6, 4.9087e-6

# yield strength, initial elastic tangent (kg/cm^2), strain-hardening ratio
STEEL_PARAMS = (7e8, 100., 1.3)
STEEL_TAG = 1

def rotate(coords, rotation: float):
    """
        Rotates point around the coordinate center (only x and y).
        The rotation has to be passed in radians.
    """
    if rotation == 0.:
        return coords
    s, c = math.sin(rotation), math.cos(rotation)
    x, y, z = coords
    return [x * c - y * s,
            x * s + y * c,
            z]

def add_nodes(nodes, fix_bottom=True, rotation=0.):
    for n in nodes:
        coords = n.as_array()
        ops.node(n.idx, *rotate(coords, rotation))
        if fix_bottom and coords[2] == 0:
            ops.fix(n.idx, 1, 1, 1, 1, 1, 1)
        ops.mass(n.idx, *MASSES_PER_DOF)
        # print(i, g3d.points3d[i])

def add_edges(edges, offset=0):
    i = offset
    for _, edge in enumerate(edges):
        transform_direction = 1 if edge.start.z != edge.end.z else 2
        if edge.material == Material.Strong:
            ops.element('elasticBeamColumn', i, *edge.as_tuple(), M_area, E_mod, G_mod, M_Jxx, M_Iy, M_Iz, transform_direction, '-mass', M, MASS_TYPE)
        elif edge.material == Material.Weak:
            ops.element('elasticBeamColumn', i, *edge.as_tuple(), X_area, E_mod, G_mod, X_Jxx, X_Iy, X_Iz, transform_direction, '-mass', M, MASS_TYPE)
        elif edge.material == Material.Inf:
            ops.element('elasticBeamColumn', i, *edge.as_tuple(), Inf_area, 1e17, 1e17, Inf_Jxx, Inf_Iy, Inf_Iz, transform_direction, '-mass', 0, MASS_TYPE)
        else:
            ops.element('Truss', i, *edge.as_tuple(), M_area, STEEL_TAG)
        i += 1

def create_ops_model():
    ops.wipe()
    ops.model('Basic', '-ndm', 3, '-ndf', 6)

    # create geometric transformation objects. These are needed to create edges between nodes.
    ops.geomTransf('Linear', 1, 1, 0, 0)
    ops.geomTransf('Linear', 2, 0, 0, 1)
    ops.geomTransf('Linear', 3, 0, 1, 0)

    # create material Steel01
    ops.uniaxialMaterial('Steel01', STEEL_TAG, *STEEL_PARAMS)


def sub_to_ops(sub: SubstructureInterface, rotation: float=0.):
    create_ops_model()

    add_nodes(sub.nodes, rotation=rotation)
    add_edges(sub.edges)

